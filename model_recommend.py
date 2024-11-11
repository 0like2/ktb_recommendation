
'''
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
'''


import argparse
import os
import pickle

import dgl
# import evaluation
import layers
import numpy as np
import sampler as sampler_module
import torch
import torch.nn as nn
import torchtext
import tqdm
from torch.utils.data import DataLoader
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import TensorDataset
from sampler import assign_features_to_blocks  # 추가된 부분


class PinSAGEModel(nn.Module):
    def __init__(self, full_graph, textset, hidden_dims, n_layers, item_ntype, user_ntype):
        super().__init__()
        self.item_proj = layers.LinearProjector(full_graph, item_ntype, textset, hidden_dims)
        self.creator_proj = layers.LinearProjector(full_graph, user_ntype, None, hidden_dims)
        self.sage = layers.SAGENet(hidden_dims, n_layers)
        self.scorer = layers.ItemToItemScorer(full_graph, None)  # None으로 설정해 모든 노드 타입을 지원

    def forward(self, pos_graph, neg_graph, blocks, embedding, ntype):
        h_node = self.get_repr(blocks, embedding, ntype)
        pos_score = self.scorer(pos_graph, h_node)
        neg_score = self.scorer(neg_graph, h_node)
        return (neg_score - pos_score + 1).clamp(min=0)

    def get_repr(self, blocks, embedding, ntype):
        if ntype == "item":
            h_node = self.item_proj({k: v for k, v in blocks[0].srcdata.items() if k in self.item_proj.inputs})
            h_node_dst = self.item_proj({k: v for k, v in blocks[-1].dstdata.items() if k in self.item_proj.inputs})
        else:
            # Debugging output -> 삭제 필요
            print("Available keys in blocks[0].srcdata.items():", blocks[0].srcdata.keys())
            print("Expected keys in creator_proj.inputs:", self.creator_proj.inputs)

            h_node = self.creator_proj({k: v for k, v in blocks[0].srcdata.items() if k in self.creator_proj.inputs})
            h_node_dst = self.creator_proj(
                {k: v for k, v in blocks[-1].dstdata.items() if k in self.creator_proj.inputs})

        # 해당하는 노드 타입의 임베딩 추가
        h_node = h_node + embedding(blocks[0].srcdata[dgl.NID].cpu()).to(h_node)
        h_node_dst = h_node_dst + embedding(blocks[-1].dstdata[dgl.NID].cpu()).to(h_node_dst)

        return h_node_dst + self.sage(blocks, h_node)


def train(dataset, args):
    g = dataset["train-graph"]
    val_matrix = dataset["val-matrix"].tocsr()
    test_matrix = dataset["test-matrix"].tocsr()
    item_texts = dataset["textset"]["item-texts"][0]  # 텍스트 리스트
    creator_texts = dataset["textset"]["creator-texts"][0]  # 텍스트 리스트
    user_ntype = dataset["user-type"]
    item_ntype = dataset["item-type"]

    device = torch.device(args.device)

    # data.pkl에서 저장된 textset 불러오기
    textset = dataset.get("textset", None)
    if textset is None:
        raise ValueError("Textset not found in dataset. Ensure data.pkl includes textset.")

    # item-texts와 creator-texts 각각에 대해 vocab 설정
    item_vocab = textset["item-texts"][1]
    item_vocab.set_default_index(item_vocab["<unk>"])
    item_pad_var = textset["item-texts"][2]
    batch_first = textset["item-texts"][3]

    creator_vocab = textset["creator-texts"][1]
    creator_vocab.set_default_index(creator_vocab["<unk>"])
    creator_pad_var = textset["creator-texts"][2]

    print("Loaded vocabulary size for item-texts:", len(item_vocab))
    print("Loaded vocabulary size for creator-texts:", len(creator_vocab))

    # textset 업데이트: item-texts와 creator-texts를 각각 포함
    textset = {
        "item-texts": (item_texts, item_vocab, item_pad_var, batch_first),
        "creator-texts": (creator_texts, creator_vocab, creator_pad_var, batch_first)
    }


    # Sampler
    batch_sampler = sampler_module.ItemToItemBatchSampler(
        g, user_ntype, item_ntype, args.batch_size
    )
    neighbor_sampler = sampler_module.NeighborSampler(
        g,
        user_ntype,
        item_ntype,
        args.random_walk_length,
        args.random_walk_restart_prob,
        args.num_random_walks,
        args.num_neighbors,
        args.num_layers,
        user_to_item_etype= "creator_to_item",  # 새로운 엣지 타입 반영
        item_to_user_etype= "item_to_creator"   # 새로운 엣지 타입 반영
    )
    collator = sampler_module.PinSAGECollator(
        neighbor_sampler, g, item_ntype, user_ntype, textset
    )

    dataloader = DataLoader(
        batch_sampler,
        collate_fn=collator.collate_train,
        num_workers=args.num_workers,
    )

    # ** 디버깅 코드 추가 ** -> 삭제 필요
    for pos_graph, neg_graph, blocks in dataloader:
        print("NeighborSampler sampled blocks node types:", [block.ntypes for block in blocks])
        print("blocks[0].srcdata keys:", blocks[0].srcdata.keys())
        print("blocks[0].dstdata keys:", blocks[0].dstdata.keys())
        break  # 첫 번째 배치만 출력

    # `torch.arange(g.num_nodes(item_ntype))`을 Dataset으로 감싸기
    item_tensor = torch.arange(g.num_nodes(item_ntype))
    item_dataset = TensorDataset(item_tensor)
    
    # `creator_tensor` 생성 -> 수정 부분
    creator_tensor = torch.arange(g.num_nodes(user_ntype))
    creator_dataset = TensorDataset(creator_tensor)


    dataloader_test = DataLoader(
        item_dataset,
        batch_size=args.batch_size,
        collate_fn=collator.collate_test,
        num_workers=args.num_workers,
    )

    creator_dataloader_test = DataLoader(
        creator_dataset,
        batch_size=args.batch_size,
        collate_fn=collator.collate_test,
        num_workers=args.num_workers,
    )

    dataloader_it = iter(dataloader)

    # Model
    model = PinSAGEModel(
        g, textset, args.hidden_dims, args.num_layers, item_ntype, user_ntype
    ).to(device)

    embedding = nn.Embedding(g.num_nodes(), args.hidden_dims, sparse=True).to(device)

    item_emb = nn.Embedding(
        g.num_nodes(item_ntype), args.hidden_dims, sparse=True
    )

    creator_emb = nn.Embedding(
        g.num_nodes(user_ntype), args.hidden_dims, sparse=True
    )

    # Optimizer
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    opt_emb = torch.optim.SparseAdam(embedding.parameters(), lr=args.lr)

    # 학습 루프
    for epoch_id in range(args.num_epochs):
        model.train()
        for batch_id, (pos_graph, neg_graph, blocks) in enumerate(dataloader):
            blocks = [block.to(device) for block in blocks]
            pos_graph = pos_graph.to(device)
            neg_graph = neg_graph.to(device)

            # 노드 타입에 따라 적절한 ntype을 설정
            ntype = "item" if "item" in pos_graph.ntypes else "creator"
            # assign_features_to_blocks에서 creator와 item 노드에 특성을 할당하도록 보장

            for block in blocks:
                # src, dst에서 creator와 item 노드 모두에 대해 특성 할당
                assign_features_to_blocks([block], g, textset, 'creator')
                assign_features_to_blocks([block], g, textset, 'item')

            # 통합 모델의 손실 계산
            loss = model(pos_graph, neg_graph, blocks, embedding, ntype).mean()
            opt.zero_grad()
            opt_emb.zero_grad()
            loss.backward()
            opt.step()
            opt_emb.step()




    # 학습이 완료된 모델과 임베딩 저장
    print("Saving model state_dict to saved_model.pth...")
    torch.save(model.state_dict(), os.path.join(args.output_dir, "saved_model.pth"))
    # item_emb를 state_dict 형식으로 저장하기 전 확인용 출력문 추가

    # item_emb 저장 방식 확인
    item_emb_state_dict = item_emb.state_dict()
    print("Type of item_emb state_dict before saving:", type(item_emb_state_dict))
    # craetor_emb 저장 방식 확인
    creator_emb_state_dict = creator_emb.state_dict()
    print("Type of item_emb state_dict before saving:", type(creator_emb_state_dict))

    print("Saving item_emb state_dict to item_embedding.pth...")
    torch.save(item_emb.state_dict(), os.path.join(args.output_dir, "item_embedding.pth"))
    print("Saving creator_emb state_dict to creator_embedding.pth...")
    torch.save(creator_emb.state_dict(), os.path.join(args.output_dir, "creator_embedding.pth"))

    # 통합 embedding 저장 방식 확인
    embedding_state_dict = embedding.state_dict()
    print("Type of embedding state_dict before saving:", type(embedding_state_dict))
    torch.save(embedding.state_dict(), os.path.join(args.output_dir, "embedding.pth"))

    return model, item_emb, creator_emb, embedding

