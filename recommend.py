import torch
import pickle
import os
import dgl
from model_recommend import PinSAGEModel

def load_model_and_embeddings(model_path, item_emb_path, creator_emb_path, graph_path, data_path):
    # 학습된 모델, 임베딩 및 그래프 로드
    g_list, _ = dgl.load_graphs(graph_path)
    graph = g_list[0]

    # data.pkl에서 textset 불러오기
    with open(data_path, "rb") as f:
        dataset = pickle.load(f)
    textset = dataset.get("textset", None)

    # 모델과 임베딩 초기화
    model = PinSAGEModel(graph, "item", textset=textset, hidden_dims=16, n_layers=2)
    item_emb = torch.nn.Embedding(graph.num_nodes("item"), 16, sparse=True)
    creator_emb = torch.nn.Embedding(graph.num_nodes("creator"), 16, sparse=True)

    # 모델 state_dict 로드 및 확인
    model_state_dict = torch.load(model_path)
    print("Type of loaded model state_dict:", type(model_state_dict))
    model.load_state_dict(model_state_dict)
    print("Loaded model state_dict keys:")
    for key in model_state_dict.keys():
        print("  -", key)

    # item_emb의 state_dict 로드 및 확인
    item_emb_state_dict = torch.load(item_emb_path)
    print("\nType of loaded item_emb state_dict:", type(item_emb_state_dict))

    item_emb.load_state_dict(item_emb_state_dict)
    print("Loaded item_emb state_dict keys:")
    for key in item_emb_state_dict.keys():
        print("  -", key)

    # creator_emb의 state_dict 로드 및 확인
    creator_emb_state_dict = torch.load(creator_emb_path)
    print("\nType of loaded creator_emb state_dict:", type(creator_emb_state_dict))

    # creator_emb가 올바른 형식인지 검사
    if isinstance(creator_emb_state_dict, dict):
        creator_emb.load_state_dict(creator_emb_state_dict)
        print("Loaded creator_emb state_dict keys:")
        for key in creator_emb_state_dict.keys():
            print("  -", key)
    else:
        print("Error: creator_emb is not in state_dict format.")

    return model, item_emb, creator_emb, graph


def is_creator_or_item(new_data):
    # 데이터의 특징에 따라 item인지 creator인지 구분하는 함수
    if 'category' in new_data and 'subscriber_count' in new_data:
        return 'creator'
    elif 'title' in new_data and 'description' in new_data:
        return 'item'
    else:
        raise ValueError("데이터가 'item'인지 'creator'인지 확인할 수 없습니다.")





def recommend_for_new_item(model, creator_emb, new_item_data, creator_data, top_k=5):
    device = next(model.parameters()).device  # 모델과 동일한 장치로 설정
    model.eval()  # 평가 모드 전환

    # 새로운 item의 세부 정보 추출
    category = new_item_data.get('item_category')
    title = new_item_data.get('title')
    score = int(new_item_data.get('score', 0))

    # item 임베딩 생성
    item_embedding = torch.zeros(creator_emb.embedding_dim).to(device)

    with torch.no_grad():
        # 모델에 정의된 proj 레이어로 각 특성을 임베딩
        if hasattr(model, 'proj'):
            category_emb = model.proj({'category': category}).to(device)
            title_emb = model.proj({'title': title}).to(device)
            score_emb = model.proj({'score': torch.tensor(score).float()}).to(device)
            item_embedding = category_emb + title_emb + score_emb  # 최종 item 임베딩

    # 모든 creator 임베딩 가져오기
    creator_embeddings = creator_emb.weight.detach().to(device)

    # 유사도 계산: item 임베딩과 모든 creator 임베딩 간 내적 계산
    similarities = torch.matmul(creator_embeddings, item_embedding.T).squeeze()

    # 상위 top_k creator 추출
    top_k_indices = torch.topk(similarities, top_k).indices
    recommended_creators = [creator_data[idx] for idx in top_k_indices]

    return recommended_creators


def recommend_for_new_creator(model, item_emb, new_creator_data, item_data, top_k=5):
    device = next(model.parameters()).device
    model.eval()

    # 새로운 creator의 세부 정보
    category = new_creator_data.get('channel_category')
    channel_name = new_creator_data.get('channel_name')
    subscribers = int(new_creator_data.get('subscribers', 0))

    creator_embedding = torch.zeros(item_emb.embedding_dim).to(device)

    with torch.no_grad():
        if hasattr(model, 'proj'):
            # 필요한 특성을 한 번에 묶어 `model.proj`에 전달
            creator_data = {
                'category': torch.tensor([category]).to(device),  # 변환된 키 사용
                'name': torch.tensor([channel_name]).to(device),
                'subscribers': torch.tensor([subscribers]).float().to(device)
            }
            # creator_data 확인
            print("creator_data keys:", creator_data.keys())
            print("creator_data:", creator_data)

            # `model.proj`에 묶인 데이터를 한 번에 전달하여 임베딩 계산
            creator_embedding = model.proj(creator_data).to(device)

    item_embeddings = item_emb.weight.detach().to(device)
    similarities = torch.matmul(item_embeddings, creator_embedding.T).squeeze()
    top_k_indices = torch.topk(similarities, top_k).indices
    recommended_items = [item_data[idx] for idx in top_k_indices]

    return recommended_items


def recommend_based_on_new_data(new_data, model, h_item, item_data):
    data_type = is_creator_or_item(new_data)

    if data_type == 'item':
        return recommend_for_new_item(new_data, model, h_item, k=10)
    elif data_type == 'creator':
        return recommend_for_new_creator(new_data, h_item, item_data, k=10)


if __name__ == "__main__":
    # 경로 설정
    output_dir = "./output"
    model_path = os.path.join(output_dir, "saved_model.pth")
    item_emb_path = os.path.join(output_dir, "item_embedding.pth")
    creator_emb_path = os.path.join(output_dir, "creator_embedding.pth")
    graph_path = os.path.join(output_dir, "train_g.bin")
    data_path = os.path.join(output_dir, "data.pkl")  # data_path 추가


    # 모델, 임베딩, 그래프 로드
    model, item_emb, graph = load_model_and_embeddings(model_path, item_emb_path, creator_emb_path, graph_path)

    # 아이템 임베딩 계산
    h_item = item_emb.weight.detach()

    # 새로운 데이터 예시
    new_data = {
        'category': 'Technology',
        'subscriber_count': 10000,
    }

