import torch
import pickle
import os
import dgl
from model_recommend import PinSAGEModel


def load_model_and_embeddings(model_path, item_emb_path, graph_path, data_path):
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

    # item_emb가 올바른 형식인지 검사
    if isinstance(item_emb_state_dict, dict):
        item_emb.load_state_dict(item_emb_state_dict)
        print("Loaded item_emb state_dict keys:")
        for key in item_emb_state_dict.keys():
            print("  -", key)
    else:
        print("Error: item_emb is not in state_dict format.")

    item_emb.load_state_dict(item_emb_state_dict)
    print("Loaded item_emb state_dict keys:")
    for key in item_emb_state_dict.keys():
        print("  -", key)

    return model, item_emb, graph

if __name__ == "__main__":
    output_dir = "/Users/iyeonglag/PycharmProjects/ktb_recommendation/output"  # 경로 추후 수정
    model_path = os.path.join(output_dir, "saved_model.pth")
    item_emb_path = os.path.join(output_dir, "item_embedding.pth")
    graph_path = os.path.join(output_dir, "train_g.bin")
    data_path = os.path.join(output_dir, "data.pkl")

    # 경로 존재 여부 확인
    print("Model Path Exists:", os.path.exists(model_path))
    print("Item Embedding Path Exists:", os.path.exists(item_emb_path))
    print("Graph Path Exists:", os.path.exists(graph_path))

    # 모델의 state_dict 불러오기
    state_dict = torch.load(model_path)
    print("Keys in the loaded state_dict:")
    for key in state_dict.keys():
        print("  -", key)

    # data.pkl 파일에서 vocab 불러오기
    with open(data_path, "rb") as f:
        dataset = pickle.load(f)

    # 현재 vocab 크기 확인
    current_vocab = dataset["textset"]["item-texts"][1]
    print("Current vocab size:", len(current_vocab))

    # 저장된 vocab 크기 확인
    saved_vocab_size = state_dict["proj.inputs.item-texts.emb.weight"].shape[0]
    print("Saved vocab size:", saved_vocab_size)

    # 모델, 임베딩, 그래프 로드
    model, item_emb, graph = load_model_and_embeddings(model_path, item_emb_path, graph_path, data_path)
    print("모델, 임베딩, 그래프를 로드 완료 했습니다")

    # 아이템 임베딩 계산
    h_item = item_emb.weight.detach()
    print("아이템 임베딩 계산 완료 했습니다")

    # 새로운 데이터 예시
    new_data = {
        'category': 'Technology',
        'subscriber_count': 10000,
    }
