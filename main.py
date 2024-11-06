import argparse
import os
import pickle
import dgl
from process_data import process_data
from model_recommend import train
import torch

def main(args):
    # 1. 데이터 전처리 및 그래프 생성
    print("=== 데이터 전처리 및 그래프 생성 ===")
    process_data(args.data_dir, args.output_dir)

    # 2. 학습을 위한 데이터 로드
    print("=== 데이터 로드 및 학습 시작 ===")
    dataset_path = os.path.join(args.output_dir, "data.pkl")
    with open(dataset_path, "rb") as f:
        dataset = pickle.load(f)

    # 학습 그래프 로드
    train_graph_path = os.path.join(args.output_dir, "train_g.bin")
    g_list, _ = dgl.load_graphs(train_graph_path)
    dataset["train-graph"] = g_list[0]

    # 3. 모델 학습
    model, item_emb = train(dataset, args)
    '''
    # 4. 학습된 모델 및 임베딩 저장
    model_path = os.path.join(args.output_dir, "saved_model.pth")
    item_emb_path = os.path.join(args.output_dir, "item_embedding.pth")
    torch.save(model.state_dict(), model_path)
    torch.save(item_emb.state_dict(), item_emb_path)
    '''
    # 4. 학습 완료 후 모델과 임베딩 저장
    model_save_path = os.path.join(args.output_dir, "saved_model.pth")
    item_embedding_save_path = os.path.join(args.output_dir, "item_embedding.pth")
    torch.save(model.state_dict(), model_save_path)
    torch.save(item_emb.weight.data, item_embedding_save_path)
    print(f"모델이 {model_save_path}에 저장되었습니다.")
    print(f"아이템 임베딩이 {item_embedding_save_path}에 저장되었습니다.")

if __name__ == "__main__":
    # Argument 파서 설정
    parser = argparse.ArgumentParser(description="크리에이터-기획서 추천 시스템")
    parser.add_argument("--data-dir", type=str, required=True, help="데이터 파일이 있는 디렉터리")
    parser.add_argument("--output-dir", type=str, required=True, help="출력 파일 저장 디렉터리")
    parser.add_argument("--random-walk-length", type=int, default=2)
    parser.add_argument("--random-walk-restart-prob", type=float, default=0.5)
    parser.add_argument("--num-random-walks", type=int, default=10)
    parser.add_argument("--num-neighbors", type=int, default=3)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--hidden-dims", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--num-epochs", type=int, default=10)
    parser.add_argument("--batches-per-epoch", type=int, default=5000)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--k", type=int, default=10)

    args = parser.parse_args()
    main(args)
