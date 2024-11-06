
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
    def __init__(self, full_graph, ntype, textsets, hidden_dims, n_layers):
        super().__init__()

        self.proj = layers.LinearProjector(
            full_graph, ntype, textsets, hidden_dims
        )
        self.sage = layers.SAGENet(hidden_dims, n_layers)
        self.scorer = layers.ItemToItemScorer(full_graph, ntype)

    def forward(self, pos_graph, neg_graph, blocks, item_emb):
        h_item = self.get_repr(blocks, item_emb)
        pos_score = self.scorer(pos_graph, h_item)
        neg_score = self.scorer(neg_graph, h_item)
        return (neg_score - pos_score + 1).clamp(min=0)

    def get_repr(self, blocks, item_emb):
        # project features
        h_item = self.proj(blocks[0].srcdata)
        h_item_dst = self.proj(blocks[-1].dstdata)

        # add to the item embedding itself
        h_item = h_item + item_emb(blocks[0].srcdata[dgl.NID].cpu()).to(h_item)
        h_item_dst = h_item_dst + item_emb(
            blocks[-1].dstdata[dgl.NID].cpu()
        ).to(h_item_dst)

        return h_item_dst + self.sage(blocks, h_item)


def train(dataset, args):
    g = dataset["train-graph"]
    val_matrix = dataset["val-matrix"].tocsr()
    test_matrix = dataset["test-matrix"].tocsr()

    # item-texts 구조 언팩
    item_texts_data = dataset.get("item-texts", {})
    item_texts_list, vocab, pad_token, batch_first = item_texts_data["item-texts"]

    user_ntype = dataset["user-type"]
    item_ntype = dataset["item-type"]

    device = torch.device(args.device)

    # Prepare torchtext dataset and vocabulary
    textset = {}
    tokenizer = get_tokenizer("basic_english")  # 기본 토크나이저를 사용하도록 설정

    # 이미 토큰화된 item_texts_list 사용
    textset["item-texts"] = (
        item_texts_list,
        vocab,
        pad_token,
        batch_first,
    )

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
        user_to_item_etype="creator_to_item",  # 새로운 엣지 타입 반영
        item_to_user_etype="item_to_creator"  # 새로운 엣지 타입 반영
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

    dataloader_test = DataLoader(
        item_dataset,
        batch_size=args.batch_size,
        collate_fn=collator.collate_test,
        num_workers=args.num_workers,
    )

    dataloader_it = iter(dataloader)

    # Model
    model = PinSAGEModel(
        g, item_ntype, textset, args.hidden_dims, args.num_layers
    ).to(device)
    item_emb = nn.Embedding(
        g.num_nodes(item_ntype), args.hidden_dims, sparse=True
    )
    # Optimizer
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    opt_emb = torch.optim.SparseAdam(item_emb.parameters(), lr=args.lr)

    # For each batch of head-tail-negative triplets...
    for epoch_id in range(args.num_epochs):
        model.train()
        for batch_id in tqdm.trange(args.batches_per_epoch):
            pos_graph, neg_graph, blocks = next(dataloader_it)
            for i in range(len(blocks)):
                blocks[i] = blocks[i].to(device)
            pos_graph = pos_graph.to(device)
            neg_graph = neg_graph.to(device)

            loss = model(pos_graph, neg_graph, blocks, item_emb).mean()
            opt.zero_grad()
            opt_emb.zero_grad()
            loss.backward()
            opt.step()
            opt_emb.step()

    return model, item_emb







if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_path", type=str)
    parser.add_argument("--random-walk-length", type=int, default=2)
    parser.add_argument("--random-walk-restart-prob", type=float, default=0.5)
    parser.add_argument("--num-random-walks", type=int, default=10)
    parser.add_argument("--num-neighbors", type=int, default=3)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--hidden-dims", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--device", type=str, default="cpu"
    )  # can also be "cuda:0"
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--batches-per-epoch", type=int, default=20000)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("-k", type=int, default=10)
    args = parser.parse_args()

    # Load dataset
    data_info_path = os.path.join(args.dataset_path, "data.pkl")
    with open(data_info_path, "rb") as f:
        dataset = pickle.load(f)
    train_g_path = os.path.join(args.dataset_path, "train_g.bin")
    g_list, _ = dgl.load_graphs(train_g_path)
    dataset["train-graph"] = g_list[0]
    train(dataset, args)