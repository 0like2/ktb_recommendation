import pickle

from recommend import load_model_and_embeddings, recommend_for_new_creator, recommend_for_new_item
import torch
import os
import random
from typing import List
import pandas as pd


new_creator_data = {
    'channel_category': "게임",
    'channel_name': "최마태의 POST IT",
    'subscribers': "300000"
}

# 새로운 item 데이터 예시
new_item_data = {
    'title': "서머 시즌의 T1과 월즈에서의 T1이 달랐던 이유",
    'item_category': '게임',
    'score': 95,
    'item_content': '게임에 대한 상세한 분석과 플레이 전략 소개'
}

def factorize_data(data, mappings=None):
    transformed_data = {}
    if mappings is None:
        mappings = {}
    for key, value in data.items():
        if key not in mappings:
            mappings[key] = {}  # 새 키의 매핑 딕셔너리 생성
        # 기존 매핑이 있는 경우 활용
        if value not in mappings[key]:
            # 새로운 값을 발견하면 자동으로 인덱스를 추가
            new_index = len(mappings[key])
            mappings[key][value] = new_index
        transformed_data[key] = torch.tensor([mappings[key][value]]).long()
    return transformed_data, mappings




def recommend_for_creator(mappings=None):
    output_dir = "/Users/iyeonglag/PycharmProjects/ktb_recommendation/output"  # 경로 수정 필요 시 수정
    model_path = os.path.join(output_dir, "saved_model.pth")
    item_emb_path = os.path.join(output_dir, "item_embedding.pth")
    creator_emb_path = os.path.join(output_dir, "creator_embedding.pth")  # creator_emb 경로 추가
    graph_path = os.path.join(output_dir, "train_g.bin")
    data_path = os.path.join(output_dir, "data.pkl")

    # 모델, item_emb, creator_emb, graph 로드
    model, item_emb, creator_emb, graph = load_model_and_embeddings(model_path, item_emb_path, creator_emb_path, graph_path, data_path)
    print("모델, 임베딩, 그래프를 로드 완료 했습니다")

    # 아이템 임베딩 계산
    h_item = item_emb.weight.detach()
    print("아이템 임베딩 계산 완료 했습니다")

    # `factorize_data`에서 생성된 mappings 반환
    transformed_creator_data, mappings = factorize_data(new_creator_data, mappings)

    # 추천 수행
    recommended_item_ids = recommend_for_new_creator(model, item_emb, transformed_creator_data, h_item, top_k=10)

    # item_data 로드
    with open(data_path, "rb") as f:
        dataset = pickle.load(f)
    item_data = dataset.get("item_data")

    # 추천 결과를 딕셔너리로 반환
    return [
        {
            "item_id": item_id,
            "title": item_data[item_id]["title"],
            "item_category": item_data[item_id]["item_category"],
            "media_type": item_data[item_id]["media_type"],
            "score": item_data[item_id]["score"]
        }
        for item_id in recommended_item_ids
    ]


def recommend_for_item(mappings=None):
    output_dir = "./output"  # 모델과 데이터가 저장된 경로
    model_path = os.path.join(output_dir, "saved_model.pth")
    item_emb_path = os.path.join(output_dir, "item_embedding.pth")
    creator_emb_path = os.path.join(output_dir, "creator_embedding.pth")  # creator_emb 경로 추가
    graph_path = os.path.join(output_dir, "train_g.bin")
    data_path = os.path.join(output_dir, "data.pkl")

    # 모델, item_emb, creator_emb, graph 로드
    model, item_emb, creator_emb, graph = load_model_and_embeddings(model_path, item_emb_path, creator_emb_path, graph_path, data_path)
    h_creator = creator_emb.weight.detach()

    # Item 데이터 factorize 방식으로 변환
    transformed_item_data, _ = factorize_data(new_item_data)
    # `factorize_data`에서 생성된 mappings 반환
    transformed_item_data, mappings = factorize_data(new_item_data, mappings)

    # 추천 수행
    recommended_creator_ids = recommend_for_new_item(model, h_creator, transformed_item_data, h_creator, top_k=10)

    # creator_data 로드
    with open(data_path, "rb") as f:
        dataset = pickle.load(f)
    creator_data = dataset.get("creator_data")

    # 추천 결과를 딕셔너리로 반환
    return [
        {
            "user_id": creator_id,
            "channel_name": creator_data[creator_id]["channel_name"],
            "channel_category": creator_data[creator_id]["channel_category"],
            "subscribers": creator_data[creator_id]["subscribers"]
        }
        for creator_id in recommended_creator_ids
    ]

if __name__ == "__main__":
    mappings = None  # 초기 매핑 정의

    # 새로운 creator에 대한 추천
    print("=== Creator 추천 결과 ===")
    creator_recommendations, mappings = recommend_for_creator(mappings)
    for recommendation in creator_recommendations:
        print(recommendation)

    # 새로운 item에 대한 추천
    print("\n=== Item 추천 결과 ===")
    item_recommendations, mappings = recommend_for_item(mappings)
    for recommendation in item_recommendations:
        print(recommendation)
