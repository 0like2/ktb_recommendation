import dgl
import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset
from torchtext.data.functional import numericalize_tokens_from_iterator


def padding(array, yy, val):
    """
    :param array: torch tensor array
    :param yy: desired width
    :param val: padded value
    :return: padded array
    """
    w = array.shape[0]
    b = 0
    bb = yy - b - w

    return torch.nn.functional.pad(
        array, pad=(b, bb), mode="constant", value=val
    )


def compact_and_copy(frontier, seeds):
    block = dgl.to_block(frontier, seeds)
    for col, data in frontier.edata.items():
        if col == dgl.EID:
            continue
        block.edata[col] = data[block.edata[dgl.EID]]  # 엣지 데이터 복사
        print(f"복사된 엣지 데이터: {col} - {block.edata[col]}")  # 엣지 데이터 출력
    return block


class ItemToItemBatchSampler(IterableDataset):
    def __init__(self, g, user_type, item_type, batch_size):
        self.g = g
        self.user_type = user_type
        self.item_type = item_type

        # 엣지 타입을 명확히 지정
        self.user_to_item_etype = "creator_to_item"
        self.item_to_user_etype = "item_to_creator"

        self.batch_size = batch_size

    def __iter__(self):
        while True:
            heads = torch.randint(
                0, self.g.num_nodes(self.item_type), (self.batch_size,)
            )
            # 수정된 메타패스 설정
            tails = dgl.sampling.random_walk(
                self.g,
                heads,
                metapath=[
                    ("item", "item_to_creator", "creator"),
                    ("creator", "creator_to_item", "item")
                ],
            )[0][:, 2]
            neg_tails = torch.randint(
                0, self.g.num_nodes(self.item_type), (self.batch_size,)
            )

            mask = tails != -1
            yield heads[mask], tails[mask], neg_tails[mask]


class RandomWalkSampler:
    def __init__(self, g, metapath, num_walks, walk_length):
        self.g = g
        self.metapath = metapath
        self.num_walks = num_walks
        self.walk_length = walk_length

    def __call__(self, seeds):
        # 메타패스를 따른 random walk 샘플링
        traces, types = dgl.sampling.random_walk(
            self.g,
            seeds,
            metapath=self.metapath
        )

        # 노드와 엣지를 frontier에 포함
        frontier = dgl.heterograph({
            ('item', 'item_to_creator', 'creator'): (traces[0], traces[1]),
            ('creator', 'creator_to_item', 'item'): (traces[1], traces[0])
        })
        return frontier


class RandomWalkSampler:
    def __init__(self, g, metapath, num_walks, walk_length):
        self.g = g
        self.metapath = metapath
        self.num_walks = num_walks
        self.walk_length = walk_length

    def __call__(self, seeds):
        # 메타패스를 따른 random walk 샘플링
        traces, types = dgl.sampling.random_walk(
            self.g,
            seeds,
            metapath=self.metapath
        )

        # 노드와 엣지를 frontier에 포함
        frontier = dgl.heterograph({
            ('item', 'item_to_creator', 'creator'): (traces[0], traces[1]),
            ('creator', 'creator_to_item', 'item'): (traces[1], traces[0])
        })
        return frontier

class NeighborSampler(object):
    def __init__(
        self,
        g,
        user_type,
        item_type,
        random_walk_length,
        random_walk_restart_prob,
        num_random_walks,
        num_neighbors,
        num_layers,
        user_to_item_etype="creator_to_item",
        item_to_user_etype="item_to_creator"
    ):
        self.g = g
        self.user_type = user_type
        self.item_type = item_type
        self.user_to_item_etype = user_to_item_etype
        self.item_to_user_etype = item_to_user_etype

        self.metapath = [
            (self.item_type, "item_to_creator", self.user_type),
            (self.user_type, "creator_to_item", self.item_type)
        ]

        self.samplers = [
            RandomWalkSampler(
                g,
                self.metapath,
                num_walks=num_random_walks,
                walk_length=random_walk_length
            )
            for _ in range(num_layers)
        ]

    def sample_blocks(self, seeds):
        seeds = seeds.view(-1)
        blocks = []

        for sampler in self.samplers:
            # 각 샘플링 단계에서 frontier를 통해 block 생성
            frontier = sampler(seeds)

            # 디버깅 -> 삭제 필요
            print("샘플링된 frontier 노드 타입:", frontier.ntypes)
            print("샘플링된 frontier 엣지 타입:", frontier.etypes)

            # dst_nodes에 item과 creator 노드 모두 포함
            # dst_nodes를 명시적으로 설정하고, seeds는 src_nodes로 설정
            block = dgl.to_block(
                frontier,
                seeds,  # seeds는 src_nodes로만 사용
                dst_nodes=frontier.dstnodes['item'] + frontier.dstnodes['creator'],  # item과 creator 노드 모두 포함
                include_dst_in_src=True
            )

            # 디버깅 -> 삭제 필요
            print("블록 생성 후 노드 타입:", block.ntypes)
            print("블록 srcdata NID 크기:", len(block.srcdata[dgl.NID]))
            print("블록 dstdata NID 크기:", len(block.dstdata[dgl.NID]))

            # 블록에 올바른 엣지 타입 설정
            for etype in frontier.canonical_etypes:
                block.edges[etype].data.update(frontier.edges[etype].data)

            # block의 src, dst 노드에 대해 타입을 명확히 설정
            for ntype in frontier.ntypes:
                block.srcnodes[ntype].data.update(frontier.nodes[ntype].data)
                block.dstnodes[ntype].data.update(frontier.nodes[ntype].data)

            blocks.insert(0, block)
            seeds = block.srcdata[dgl.NID]  # 다음 레이어의 샘플링을 위한 seeds 설정

        return blocks


    def sample_from_item_pairs(self, heads, tails, neg_tails):
        pos_graph = dgl.graph(
            (heads, tails), num_nodes=self.g.num_nodes(self.item_type) + self.g.num_nodes(self.user_type)
        )
        neg_graph = dgl.graph(
            (heads, neg_tails), num_nodes=self.g.num_nodes(self.item_type) + self.g.num_nodes(self.user_type)
        )
        pos_graph, neg_graph = dgl.compact_graphs([pos_graph, neg_graph])
        seeds = pos_graph.ndata[dgl.NID]

        blocks = self.sample_blocks(seeds)
        return pos_graph, neg_graph, blocks





'''
class NeighborSampler(object):
    def __init__(
        self,
        g,
        user_type,
        item_type,
        random_walk_length,
        random_walk_restart_prob,
        num_random_walks,
        num_neighbors,
        num_layers,
        user_to_item_etype="creator_to_item",
        item_to_user_etype="item_to_creator"
    ):
        self.g = g
        self.user_type = user_type
        self.item_type = item_type
        self.user_to_item_etype = user_to_item_etype
        self.item_to_user_etype = item_to_user_etype

        self.metapath = [
            (self.item_type, "item_to_creator", self.user_type),
            (self.user_type, "creator_to_item", self.item_type)
        ]

        self.samplers = [
            dgl.sampling.PinSAGESampler(
                g,
                item_type,
                user_type,
                random_walk_length,
                random_walk_restart_prob,
                num_random_walks,
                num_neighbors
            )
            for _ in range(num_layers)
        ]

    def sample_blocks(self, seeds, heads=None, tails=None, neg_tails=None):
        seeds = seeds.view(-1)
        blocks = []

        for sampler in self.samplers:
            # 각 샘플링 단계에서 frontier를 통해 block 생성
            frontier = sampler(seeds)

            # 디버깅 -> 삭제 필요
            print("샘플링된 frontier 노드 타입:", frontier.ntypes)
            print("샘플링된 frontier 엣지 타입:", frontier.etypes)

            # frontier에 creator 노드와 item_to_creator, creator_to_item 엣지가 포함되어 있는지 확인
            if 'creator' in frontier.ntypes:
                print("creator 노드가 frontier에 포함됨")
            else:
                print("creator 노드가 frontier에 없음")

            if 'item_to_creator' in frontier.etypes:
                print("item_to_creator 엣지가 frontier에 포함됨")
            else:
                print("item_to_creator 엣지가 frontier에 없음")

            if 'creator_to_item' in frontier.etypes:
                print("creator_to_item 엣지가 frontier에 포함됨")
            else:
                print("creator_to_item 엣지가 frontier에 없음")

            block = dgl.to_block(frontier, seeds, include_dst_in_src=True)

            # 디버깅 -> 삭제 필요
            print("블록 생성 후 노드 타입:", block.ntypes)
            print("블록 srcdata NID 크기:", len(block.srcdata[dgl.NID]))
            print("블록 dstdata NID 크기:", len(block.dstdata[dgl.NID]))

            # 올바른 엣지 타입 설정
            for etype in frontier.canonical_etypes:
                block.edges[etype].data.update(frontier.edges[etype].data)

            # block의 src, dst 노드에 대해 타입을 명확히 설정
            for ntype in frontier.ntypes:
                block.srcnodes[ntype].data.update(frontier.nodes[ntype].data)
                block.dstnodes[ntype].data.update(frontier.nodes[ntype].data)

            blocks.insert(0, block)
            seeds = block.srcdata[dgl.NID]


            # 디버깅 -> 삭제 필요
            print(f"=== 블록 디버그 정보 ===")
            print(f"블록 {len(blocks)}의 ntypes: {block.ntypes}")
            print(f"블록 {len(blocks)}의 srcdata[dgl.NID] 크기: {len(block.srcdata[dgl.NID])}")
            print(f"블록 {len(blocks)}의 dstdata[dgl.NID] 크기: {len(block.dstdata[dgl.NID])}")

        return blocks

    def sample_from_item_pairs(self, heads, tails, neg_tails):
        pos_graph = dgl.graph(
            (heads, tails), num_nodes=self.g.num_nodes(self.item_type) + self.g.num_nodes(self.user_type)
        )
        neg_graph = dgl.graph(
            (heads, neg_tails), num_nodes=self.g.num_nodes(self.item_type) + self.g.num_nodes(self.user_type)
        )
        pos_graph, neg_graph = dgl.compact_graphs([pos_graph, neg_graph])
        seeds = pos_graph.ndata[dgl.NID]

        blocks = self.sample_blocks(seeds, heads, tails, neg_tails)
        return pos_graph, neg_graph, blocks
'''

def assign_simple_node_features(ndata, g, ntype, assign_id=False):
    for col in g.nodes[ntype].data.keys():
        if not assign_id and col == dgl.NID:
            continue
        induced_nodes = ndata[dgl.NID]

        # 디버깅 -> 삭제 필요
        print(f"{ntype} 노드 타입 - '{col}' 데이터 크기 비교:")
        print(f"induced_nodes (크기 {len(induced_nodes)}): {induced_nodes}")
        print(f"g.nodes[{ntype}].data[{col}] 크기: {g.nodes[ntype].data[col].shape}")

        try:
            ndata[col] = g.nodes[ntype].data[col][induced_nodes]
        except IndexError:
            raise ValueError(f"인덱스 오류: induced_nodes 크기({len(induced_nodes)})와 "
                             f"g.nodes[{ntype}].data[{col}] 크기({g.nodes[ntype].data[col].shape[0]})가 "
                             "맞지 않습니다.")


def assign_textual_node_features(ndata, textset, ntype):
    node_ids = ndata[dgl.NID].numpy()

    for field_name, field in textset.items():
        textlist, vocab, pad_var, batch_first = field

        examples = [textlist[i] for i in node_ids]
        ids_iter = numericalize_tokens_from_iterator(vocab, examples)

        maxsize = max([len(textlist[i]) for i in node_ids])
        ids = next(ids_iter)
        x = torch.asarray([num for num in ids])
        lengths = torch.tensor([len(x)])
        tokens = padding(x, maxsize, pad_var)

        for ids in ids_iter:
            x = torch.asarray([num for num in ids])
            l = torch.tensor([len(x)])
            y = padding(x, maxsize, pad_var)
            tokens = torch.vstack((tokens, y))
            lengths = torch.cat((lengths, l))

        if not batch_first:
            tokens = tokens.t()

        ndata[field_name] = tokens
        ndata[field_name + "__len"] = lengths


def assign_features_to_blocks(blocks, g, textset, ntype):
    # 디버깅 -> 삭제 필요
    print("블록의 엣지 타입 확인")
    for block in blocks:
        print(f"Block node types: {block.ntypes}")
        print(f"Block srcdata keys: {block.srcdata.keys()}")
        print(f"Block dstdata keys: {block.dstdata.keys()}")
        print(f"Block etypes: {block.etypes}")  # 엣지 타입 출력
        if "creator_to_item" not in block.etypes:
            print("creator_to_item 엣지가 없어요.")
        if "item_to_creator" not in block.etypes:
            print("item_to_creator 엣지가 없어요.")

    # 올바른 키를 사용해 노드 특성 할당
    if ntype == 'creator':
        assign_simple_node_features(blocks[0].srcdata, g, 'creator')  # Creator 노드 데이터 할당
        assign_simple_node_features(blocks[0].dstdata, g, 'creator')  # Creator 노드 데이터 할당
    else:
        assign_simple_node_features(blocks[0].srcdata, g, 'item')  # Item 노드 데이터 할당
        assign_simple_node_features(blocks[0].dstdata, g, 'item')  # Item 노드 데이터 할당



class PinSAGECollator:
    def __init__(self, sampler, g, item_type, user_type, textset):
        self.sampler = sampler
        self.g = g
        self.item_type = item_type
        self.user_type = user_type
        self.textset = textset

    def collate_train(self, batches):
        heads, tails, neg_tails = batches[0]
        pos_graph, neg_graph, blocks = self.sampler.sample_from_item_pairs(
            heads, tails, neg_tails
        )

        # Assign features for both item and creator types to each block
        for block in blocks:
            # Item features
            assign_features_to_blocks(block, self.g, self.textset, self.item_type)
            # Creator features
            assign_features_to_blocks(block, self.g, self.textset, self.user_type)

        return pos_graph, neg_graph, blocks

    def collate_test(self, samples):
        batch = torch.LongTensor(samples)
        blocks = self.sampler.sample_blocks(batch)

        # Assign features for both item and creator types to each block
        for block in blocks:
            assign_features_to_blocks(block, self.g, self.textset, self.item_type)
            assign_features_to_blocks(block, self.g, self.textset, self.user_type)

        return blocks

