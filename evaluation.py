import argparse
import pickle

import dgl
import numpy as np
import torch


def prec(recommendations, ground_truth):
    n_users, n_items = ground_truth.shape
    K = recommendations.shape[1]

    # recommendations의 사용자 수가 ground_truth와 맞는지 확인
    if recommendations.shape[0] != n_users:
        raise ValueError(f"Mismatch in number of users: expected {n_users}, got {recommendations.shape[0]}")

    user_idx = np.repeat(np.arange(n_users), K)
    item_idx = recommendations.flatten()

    # item_idx가 ground_truth의 열 크기를 초과하지 않도록 제한
    item_idx = np.clip(item_idx, 0, n_items - 1)

    relevance = ground_truth[user_idx, item_idx].reshape((n_users, K))
    hit = relevance.any(axis=1).mean()
    return hit


class LatestNNRecommender:
    def __init__(self, user_ntype, item_ntype, user_to_item_etype, batch_size):
        self.user_ntype = user_ntype
        self.item_ntype = item_ntype
        self.user_to_item_etype = user_to_item_etype
        self.batch_size = batch_size

    def recommend(self, full_graph, K, h_user, h_item):
        graph_slice = full_graph.edge_type_subgraph([self.user_to_item_etype])
        n_users = full_graph.num_nodes(self.user_ntype)

        # 그래프의 모든 유저가 포함되어 있는지 확인
        user, latest_items = graph_slice.edges(form="uv")
        unique_user = torch.unique(user)

        # 모든 유저가 포함되어 있지 않은 경우, 기본값을 사용하여 채우기
        if len(unique_user) < n_users:
            missing_users = torch.tensor([u for u in range(n_users) if u not in unique_user])
            latest_items = torch.cat([latest_items, torch.zeros(len(missing_users), dtype=latest_items.dtype)])
            unique_user = torch.cat([unique_user, missing_users])

        # 추천 수행
        recommended_batches = []
        user_batches = unique_user.split(self.batch_size)

        for user_batch in user_batches:
            latest_item_batch = latest_items[user_batch].to(device=h_item.device)
            dist = h_item[latest_item_batch] @ h_item.t()

            for i, u in enumerate(user_batch.tolist()):
                interacted_items = full_graph.successors(u, etype=self.user_to_item_etype)
                dist[i, interacted_items] = -np.inf

            recommended_batches.append(dist.topk(K, 1)[1])

        recommendations = torch.cat(recommended_batches, 0)
        return recommendations[:n_users]


def evaluate_nn(dataset, h_item, k, batch_size):
    g = dataset.get("train-graph")
    val_matrix = dataset.get("val-matrix").tocsr()
    test_matrix = dataset.get("test-matrix").tocsr()
    item_texts = dataset.get("item-texts")
    user_ntype = dataset.get("user-type", "user")  # 기본값 "user"
    item_ntype = dataset.get("item-type", "item")  # 기본값 "item"
    user_to_item_etype = dataset.get("user-to-item-type", "user_to_item")  # 기본값 설정

    rec_engine = LatestNNRecommender(
        user_ntype, item_ntype, user_to_item_etype, batch_size
    )

    recommendations = rec_engine.recommend(g, k, None, h_item).cpu().numpy()
    return prec(recommendations, val_matrix)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_path", type=str)
    parser.add_argument("item_embedding_path", type=str)
    parser.add_argument("-k", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    with open(args.dataset_path, "rb") as f:
        dataset = pickle.load(f)
    with open(args.item_embedding_path, "rb") as f:
        emb = torch.FloatTensor(pickle.load(f))
    print(evaluate_nn(dataset, emb, args.k, args.batch_size))
