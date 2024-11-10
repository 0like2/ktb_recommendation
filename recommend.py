import torch
import pickle
import os
import dgl
from model_recommend import PinSAGEModel

def load_model_and_embeddings(item_model_path, creator_model_path, item_emb_path, creator_emb_path, graph_path, data_path):
    # 학습된 모델, 임베딩 및 그래프 로드
    g_list, _ = dgl.load_graphs(graph_path)
    graph = g_list[0]

    # data.pkl에서 textset 불러오기
    with open(data_path, "rb") as f:
        dataset = pickle.load(f)
    textset = dataset.get("textset", None)

    # 모델 로드 시 각 노드 타입에 대해 별도로 모델 인스턴스 생성
    item_model = PinSAGEModel(graph, "item", textset=textset, hidden_dims=16, n_layers=2)
    creator_model = PinSAGEModel(graph, "creator", textset=None, hidden_dims=16, n_layers=2)

    item_emb = torch.nn.Embedding(graph.num_nodes("item"), 16, sparse=True)
    creator_emb = torch.nn.Embedding(graph.num_nodes("creator"), 16, sparse=True)

    # item_model과 creator_model의 state_dict 로드
    item_model_state_dict = torch.load(item_model_path)
    creator_model_state_dict = torch.load(creator_model_path)
    item_model.load_state_dict(item_model_state_dict)
    creator_model.load_state_dict(creator_model_state_dict)

    # item_emb의 state_dict 로드 및 확인
    item_emb_state_dict = torch.load(item_emb_path)
    item_emb.load_state_dict(item_emb_state_dict)

    # creator_emb의 state_dict 로드 및 확인
    creator_emb_state_dict = torch.load(creator_emb_path)
    creator_emb.load_state_dict(creator_emb_state_dict)

    return item_model, creator_model, item_emb, creator_emb, graph


def is_creator_or_item(new_data):
    # 데이터의 특징에 따라 item인지 creator인지 구분하는 함수
    if 'channel_category' in new_data and 'subscribers' in new_data:
        return 'creator'
    elif 'title' in new_data and 'item_category' in new_data:
        return 'item'
    else:
        raise ValueError("데이터가 'item'인지 'creator'인지 확인할 수 없습니다.")


def recommend_for_new_item(item_model, creator_emb, new_item_data, creator_data, top_k=5):
    device = next(item_model.parameters()).device  # 모델과 동일한 장치로 설정
    item_model.eval()  # 평가 모드 전환

    # 새로운 item의 세부 정보 추출 및 임베딩 생성
    item_data_input = {
        'item_category': torch.tensor([new_item_data.get('item_category')]).to(device),
        'title': torch.tensor([new_item_data.get('title')]).to(device),
        'score': torch.tensor([new_item_data.get('score')]).float().to(device)
    }

    with torch.no_grad():
        item_embedding = item_model.proj(item_data_input).to(device)

    # 모든 creator 임베딩과의 유사도 계산
    creator_embeddings = creator_emb.weight.detach().to(device)
    similarities = torch.matmul(creator_embeddings, item_embedding.T).squeeze()
    top_k_indices = torch.topk(similarities, top_k).indices
    recommended_creators = [creator_data[idx] for idx in top_k_indices]

    return recommended_creators


def recommend_for_new_creator(creator_model, item_emb, new_creator_data, item_data, top_k=5):
    device = next(creator_model.parameters()).device  # creator_model과 동일한 장치로 설정
    creator_model.eval()

    # 새로운 creator의 세부 정보 추출 및 임베딩 생성
    creator_data_input = {
        'channel_category': torch.tensor([new_creator_data.get('channel_category')]).to(device),
        'channel_name': torch.tensor([new_creator_data.get('channel_name')]).to(device),
        'subscribers': torch.tensor([new_creator_data.get('subscribers')]).float().to(device)
    }

    with torch.no_grad():
        creator_embedding = creator_model.proj(creator_data_input).to(device)

    # 모든 item 임베딩과의 유사도 계산
    item_embeddings = item_emb.weight.detach().to(device)
    similarities = torch.matmul(item_embeddings, creator_embedding.T).squeeze()
    top_k_indices = torch.topk(similarities, top_k).indices
    recommended_items = [item_data[idx] for idx in top_k_indices]

    return recommended_items


def recommend_based_on_new_data(new_data, item_model, creator_model, item_emb, creator_emb, item_data, creator_data, top_k=5):
    data_type = is_creator_or_item(new_data)

    if data_type == 'item':
        return recommend_for_new_item(item_model, creator_emb, new_data, creator_data, top_k)
    elif data_type == 'creator':
        return recommend_for_new_creator(creator_model, item_emb, new_data, item_data, top_k)


if __name__ == "__main__":
    # 경로 설정
    output_dir = "./output"
    item_model_path = os.path.join(output_dir, "item_model.pth")
    creator_model_path = os.path.join(output_dir, "creator_model.pth")
    item_emb_path = os.path.join(output_dir, "item_embedding.pth")
    creator_emb_path = os.path.join(output_dir, "creator_embedding.pth")
    graph_path = os.path.join(output_dir, "train_g.bin")
    data_path = os.path.join(output_dir, "data.pkl")

    # 모델, 임베딩, 그래프 로드
    item_model, creator_model, item_emb, creator_emb, graph = load_model_and_embeddings(
        item_model_path, creator_model_path, item_emb_path, creator_emb_path, graph_path, data_path
    )

    # 새로운 item 데이터 예시
    new_item_data = {
        'title': "서머 시즌의 T1과 월즈에서의 T1이 달랐던 이유",
        'item_category': '게임',
        'score': 95
    }

    # 새로운 creator 데이터 예시
    new_creator_data = {
        'channel_category': "게임",
        'channel_name': "최마태의 POST IT",
        'subscribers': 300000
    }

    # 선택한 데이터에 따라 추천 수행
    item_recommendations = recommend_for_new_item(item_model, creator_emb, new_item_data, creator_data={}, top_k=5)
    print("Recommended Creators for New Item:")
    for recommendation in item_recommendations:
        print(recommendation)

    creator_recommendations = recommend_for_new_creator(creator_model, item_emb, new_creator_data, item_data={}, top_k=5)
    print("\nRecommended Items for New Creator:")
    for recommendation in creator_recommendations:
        print(recommendation)


