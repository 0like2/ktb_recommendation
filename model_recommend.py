
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

class PinSAGEModel(nn.Module):
    def __init__(self, full_graph, ntype, textset, hidden_dims, n_layers):
        super().__init__()
        self.ntype = ntype
        self.proj = layers.LinearProjector(
            full_graph, ntype, textset, hidden_dims
        )
        self.sage = layers.SAGENet(hidden_dims, n_layers)
        self.scorer = layers.ItemToItemScorer(full_graph, ntype)

    def forward(self, pos_graph, neg_graph, blocks, embedding):
        h_node = self.get_repr(blocks, embedding)
        pos_score = self.scorer(pos_graph, h_node)
        neg_score = self.scorer(neg_graph, h_node)
        return (neg_score - pos_score + 1).clamp(min=0)

    def get_repr(self, blocks, embedding):
        # ntype에 따라 feature projection을 달리 적용
        if self.ntype == "item":
            h_node = self.proj({k: v for k, v in blocks[0].srcdata.items() if k in self.proj.inputs})
            h_node_dst = self.proj({k: v for k, v in blocks[-1].dstdata.items() if k in self.proj.inputs})
        elif self.ntype == "creator":
            h_node = self.proj({k: v for k, v in blocks[0].srcdata.items() if k in self.proj.inputs})
            h_node_dst = self.proj({k: v for k, v in blocks[-1].dstdata.items() if k in self.proj.inputs})

        # add to the embedding itself
        h_node = h_node + embedding(blocks[0].srcdata[dgl.NID].cpu()).to(h_node)
        h_node_dst = h_node_dst + embedding(blocks[-1].dstdata[dgl.NID].cpu()).to(h_node_dst)

        return h_node_dst + self.sage(blocks, h_node)


def train(dataset, args):
    g = dataset["train-graph"]
    val_matrix = dataset["val-matrix"].tocsr()
    test_matrix = dataset["test-matrix"].tocsr()
    item_texts = dataset["item-texts"]
    user_ntype = dataset["user-type"]
    item_ntype = dataset["item-type"]

    device = torch.device(args.device)

    # data.pkl에서 저장된 textset 불러오기
    textset = dataset.get("textset", None)
    if textset is None:
        raise ValueError("Textset not found in dataset. Ensure data.pkl includes textset.")

    vocab = textset["item-texts"][1]
    vocab.set_default_index(vocab["<unk>"])
    pad_var = textset["item-texts"][2]
    batch_first = textset["item-texts"][3]

    print("Loaded vocabulary size from data.pkl:", len(vocab))

    # textset 그대로 사용
    textset = {"item-texts": (item_texts, vocab, pad_var, batch_first)}


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
        neighbor_sampler, g, item_ntype, textset
    )
    dataloader = DataLoader(
        batch_sampler,
        collate_fn=collator.collate_train,
        num_workers=args.num_workers,
    )

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

    # Model 정의
    item_model = PinSAGEModel(
        g, item_ntype, textset, args.hidden_dims, args.num_layers
    ).to(device)

    creator_model = PinSAGEModel(
        g, user_ntype, None, args.hidden_dims, args.num_layers
    ).to(device)

    item_emb = nn.Embedding(
        g.num_nodes(item_ntype), args.hidden_dims, sparse=True
    )

    creator_emb = nn.Embedding(
        g.num_nodes(user_ntype), args.hidden_dims, sparse=True
    )

    # Optimizer
    opt_item_model = torch.optim.Adam(item_model.parameters(), lr=args.lr)
    opt_creator_model = torch.optim.Adam(creator_model.parameters(), lr=args.lr)
    opt_item_emb = torch.optim.SparseAdam(item_emb.parameters(), lr=args.lr)
    opt_creator_emb = torch.optim.SparseAdam(creator_emb.parameters(), lr=args.lr)


    # 학습 시작
    for epoch_id in range(args.num_epochs):
        # item 모델 학습
        item_model.train()
        dataloader_it = iter(dataloader)  # item_model 학습용 반복자 초기화
        for batch_id in tqdm.trange(args.batches_per_epoch, desc=f"Item Model Epoch {epoch_id+1}"):
            pos_graph, neg_graph, blocks = next(dataloader_it)
            blocks = [block.to(device) for block in blocks]
            pos_graph = pos_graph.to(device)
            neg_graph = neg_graph.to(device)
            loss = item_model(pos_graph, neg_graph, blocks, item_emb).mean()
            opt_item_model.zero_grad()
            opt_item_emb.zero_grad()
            loss.backward()
            opt_item_model.step()
            opt_item_emb.step()

        # creator 모델 학습
        creator_model.train()
        dataloader_it = iter(dataloader)  # creator_model 학습용 반복자 초기화
        for batch_id in tqdm.trange(args.batches_per_epoch, desc=f"Creator Model Epoch {epoch_id+1}"):
            pos_graph, neg_graph, blocks = next(dataloader_it)
            blocks = [block.to(device) for block in blocks]
            pos_graph = pos_graph.to(device)
            neg_graph = neg_graph.to(device)
            loss = creator_model(pos_graph, neg_graph, blocks, creator_emb).mean()
            opt_creator_model.zero_grad()
            opt_creator_emb.zero_grad()
            loss.backward()
            opt_creator_model.step()
            opt_creator_emb.step()


    # 학습이 완료된 모델과 임베딩 저장
    print("Saving item_model and creator_model state_dicts...")

    # 저장 경로 설정
    item_model_save_path = os.path.join(args.output_dir, "item_model.pth")
    creator_model_save_path = os.path.join(args.output_dir, "creator_model.pth")
    item_embedding_save_path = os.path.join(args.output_dir, "item_embedding.pth")
    creator_embedding_save_path = os.path.join(args.output_dir, "creator_embedding.pth")

    # item_model과 creator_model 저장
    torch.save(item_model.state_dict(), item_model_save_path)
    torch.save(creator_model.state_dict(), creator_model_save_path)
    print(f"Item model state_dict saved to {item_model_save_path}")
    print(f"Creator model state_dict saved to {creator_model_save_path}")

    # item_emb의 state_dict 저장 및 확인
    item_emb_state_dict = item_emb.state_dict()
    print("Type of item_emb state_dict before saving:", type(item_emb_state_dict))
    torch.save(item_emb_state_dict, item_embedding_save_path)
    print(f"Item embedding saved to {item_embedding_save_path}")

    # creator_emb의 state_dict 저장 및 확인
    creator_emb_state_dict = creator_emb.state_dict()
    print("Type of creator_emb state_dict before saving:", type(creator_emb_state_dict))
    torch.save(creator_emb_state_dict, creator_embedding_save_path)
    print(f"Creator embedding saved to {creator_embedding_save_path}")

    # 학습된 모델과 임베딩 반환
    return item_model, creator_model, item_emb, creator_emb

