# test_sampler.py
import os
import dgl
import torch
import pickle
from sampler import ItemToItemBatchSampler, NeighborSampler
from sampler import assign_features_to_blocks

# 그래프와 데이터 파일 경로 설정
output_dir = "../output"
graph_path = os.path.join(output_dir, "train_g.bin")
data_path = os.path.join(output_dir, "data.pkl")

# 1. 저장된 그래프 로드
print("=== 저장된 그래프 로드 ===")
graphs, _ = dgl.load_graphs(graph_path)
g = graphs[0]  # 첫 번째 그래프를 사용
print("불러온 그래프의 노드 타입:", g.ntypes)
print("불러온 그래프의 엣지 타입:", g.etypes)

# 1-1. 노드 타입 정보를 수동으로 추가
g.ndata[dgl.NTYPE] = {
    "creator": torch.full((g.num_nodes("creator"),), g.get_ntype_id("creator"), dtype=torch.long),
    "item": torch.full((g.num_nodes("item"),), g.get_ntype_id("item"), dtype=torch.long),
}

# 2. 저장된 데이터 로드
print("\n=== 저장된 데이터 로드 ===")
with open(data_path, "rb") as f:
    dataset = pickle.load(f)
print("데이터셋 키:", dataset.keys())
print("검증 행렬:", dataset["val-matrix"])
print("테스트 행렬:", dataset["test-matrix"])

# 3. ItemToItemBatchSampler 테스트
print("\n=== ItemToItemBatchSampler 테스트 ===")
item_sampler = ItemToItemBatchSampler(g, 'creator', 'item', batch_size=2)
heads, tails, neg_tails = next(iter(item_sampler))
print("샘플링된 heads:", heads)
print("샘플링된 tails:", tails)
print("샘플링된 neg_tails:", neg_tails)


# NeighborSampler 테스트
print("\n=== NeighborSampler 테스트 ===")
neighbor_sampler = NeighborSampler(
    g, 'creator', 'item', random_walk_length=2, random_walk_restart_prob=0.5,
    num_random_walks=1, num_neighbors=2, num_layers=2
)
seeds = torch.tensor([0, 1])
blocks = neighbor_sampler.sample_blocks(seeds)

# 각 블록에 노드 특성 추가
for block in blocks:
    assign_features_to_blocks([block], g, dataset["textset"], 'creator')
    assign_features_to_blocks([block], g, dataset["textset"], 'item')

print("생성된 블록 수:", len(blocks))
for i, block in enumerate(blocks):
    print(f"블록 {i}의 노드 수:", block.number_of_nodes())
    print(f"블록 {i}의 엣지 수:", block.number_of_edges())
    print(f"블록 {i}의 srcdata 키: {block.srcdata.keys()}")
    print(f"블록 {i}의 dstdata 키: {block.dstdata.keys()}")

