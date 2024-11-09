
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

        #testset 관련 부분
        self.proj = layers.LinearProjector(
            full_graph, ntype, textset, hidden_dims
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


        # Evaluate
        '''
        model.eval()
        with torch.no_grad():
            item_batches = torch.arange(g.num_nodes(item_ntype)).split(
                args.batch_size
            )
            h_item_batches = []
            for blocks in tqdm.tqdm(dataloader_test):
                for i in range(len(blocks)):
                    blocks[i] = blocks[i].to(device)
                h_item_batches.append(model.get_repr(blocks, item_emb))
            h_item = torch.cat(h_item_batches, 0)

            print(
                evaluation.evaluate_nn(dataset, h_item, args.k, args.batch_size)
            )
        '''

    '''
    dataloader_test = DataLoader(
        torch.arange(g.num_nodes(item_ntype)),
        batch_size=args.batch_size,
        collate_fn=collator.collate_test,
        num_workers=args.num_workers,
    )
    '''
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
            # Copy to GPU
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

    # 학습이 완료된 모델과 임베딩 저장
    print("Saving model state_dict to saved_model.pth...")
    torch.save(model.state_dict(), os.path.join(args.output_dir, "saved_model.pth"))
    # item_emb를 state_dict 형식으로 저장하기 전 확인용 출력문 추가

    # item_emb 저장 방식 확인
    item_emb_state_dict = item_emb.state_dict()
    print("Type of item_emb state_dict before saving:", type(item_emb_state_dict))
    print("Saving item_emb state_dict to item_embedding.pth...")
    torch.save(item_emb.state_dict(), os.path.join(args.output_dir, "item_embedding.pth"))

    return model, item_emb  # 학습이 완료된 모델과 임베딩을 반환

    '''
        # Evaluate
        model.eval()
        with torch.no_grad():
            item_batches = torch.arange(g.num_nodes(item_ntype)).split(
                args.batch_size
            )
            h_item_batches = []
            for blocks in tqdm.tqdm(dataloader_test):
                for i in range(len(blocks)):
                    blocks[i] = blocks[i].to(device)

                h_item_batches.append(model.get_repr(blocks, item_emb))
            h_item = torch.cat(h_item_batches, 0)
          
            print(
                evaluation.evaluate_nn(dataset, h_item, args.k, args.batch_size)
            )
          
    '''
