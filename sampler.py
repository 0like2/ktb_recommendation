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
        block.edata[col] = data[block.edata[dgl.EID]]
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
        user_to_item_etype="creator_to_item",  # 새 인수 추가
        item_to_user_etype="item_to_creator"   # 새 인수 추가
    ):
        self.g = g
        self.user_type = user_type
        self.item_type = item_type
        self.user_to_item_etype = user_to_item_etype
        self.item_to_user_etype = item_to_user_etype

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
            block = compact_and_copy(frontier, seeds)

            print("블록 생성 후 노드 타입:", block.ntypes)
            print("블록 srcdata NID 크기:", len(block.srcdata[dgl.NID]))

            # 추가 노드 삽입 로직을 제거하고, 중복 방지를 위해 block.srcdata[dgl.NID]에서 고유한 노드만 사용
            unique_seeds = torch.unique(block.srcdata[dgl.NID])

            # 고유 노드 목록을 seeds로 사용하여 다음 샘플링 단계에 전달
            seeds = unique_seeds
            blocks.insert(0, block)

            # 블록의 srcdata와 dstdata의 크기 및 노드 타입 확인
            print(f"=== 블록 디버그 정보 ===")
            print(f"블록 {len(blocks)}의 ntypes: {block.ntypes}")
            print(f"블록 {len(blocks)}의 srcdata[dgl.NID] 크기: {len(block.srcdata[dgl.NID])}")
            print(f"블록 {len(blocks)}의 dstdata[dgl.NID] 크기: {len(block.dstdata[dgl.NID])}")

        return blocks

    def sample_from_item_pairs(self, heads, tails, neg_tails):
        pos_graph = dgl.graph(
            (heads, tails), num_nodes=self.g.num_nodes(self.item_type)
        )
        neg_graph = dgl.graph(
            (heads, neg_tails), num_nodes=self.g.num_nodes(self.item_type)
        )
        pos_graph, neg_graph = dgl.compact_graphs([pos_graph, neg_graph])
        seeds = pos_graph.ndata[dgl.NID]

        blocks = self.sample_blocks(seeds, heads, tails, neg_tails)
        return pos_graph, neg_graph, blocks


def assign_simple_node_features(ndata, g, ntype, assign_id=False):
    for col in g.nodes[ntype].data.keys():
        if not assign_id and col == dgl.NID:
            continue
        induced_nodes = ndata[dgl.NID]

        # 디버깅: induced_nodes 및 g 노드 데이터 크기 출력
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
    # srcdata와 dstdata의 키를 확인
    for block in blocks:
        print(f"Block node types: {block.ntypes}")
        print(f"Block srcdata keys: {block.srcdata.keys()}")
        print(f"Block dstdata keys: {block.dstdata.keys()}")

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

