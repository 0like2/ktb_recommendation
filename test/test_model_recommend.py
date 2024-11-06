import argparse
import os
import pickle
import dgl
from model_recommend import train  # model_recommend에서 train 함수 임포트

# 기본 설정을 위한 argparse 인수를 설정합니다.
parser = argparse.ArgumentParser()
parser.add_argument("--dataset_path", type=str, default="./output", help="데이터셋이 저장된 경로")
parser.add_argument("--random-walk-length", type=int, default=2, help="랜덤 워크의 길이")
parser.add_argument("--random-walk-restart-prob", type=float, default=0.5, help="랜덤 워크 재시작 확률")
parser.add_argument("--num-random-walks", type=int, default=10, help="랜덤 워크 횟수")
parser.add_argument("--num-neighbors", type=int, default=3, help="각 노드의 이웃 수")
parser.add_argument("--num-layers", type=int, default=2, help="SAGE 네트워크의 레이어 수")
parser.add_argument("--hidden-dims", type=int, default=16, help="히든 레이어의 차원")
parser.add_argument("--batch-size", type=int, default=32, help="배치 크기")
parser.add_argument("--device", type=str, default="cpu", help="cpu 또는 cuda:0")
parser.add_argument("--num-epochs", type=int, default=1, help="에포크 수")
parser.add_argument("--batches-per-epoch", type=int, default=10, help="에포크 당 배치 수")
parser.add_argument("--num-workers", type=int, default=0, help="데이터 로더의 워커 수")
parser.add_argument("--lr", type=float, default=3e-5, help="학습률")
parser.add_argument("--k", type=int, default=10, help="평가 시 top-k 값")
args = parser.parse_args()

# 데이터 로드
print("=== 데이터 로드 중 ===")
data_info_path = os.path.join(args.dataset_path, "data.pkl")
train_g_path = os.path.join(args.dataset_path, "train_g.bin")

# `data.pkl` 파일 로드
with open(data_info_path, "rb") as f:
    dataset = pickle.load(f)

# `train_g.bin` 파일 로드
g_list, _ = dgl.load_graphs(train_g_path)
dataset["train-graph"] = g_list[0]

# 데이터셋과 그래프 구조 확인
print("=== 데이터셋 정보 ===")
print(f"데이터셋 키: {dataset.keys()}")
print(f"노드 타입: {dataset['train-graph'].ntypes}")
print(f"엣지 타입: {dataset['train-graph'].etypes}")
print(f"메타그래프: {dataset['train-graph'].metagraph().edges()}")

# 모델 학습 및 평가 실행
print("=== 모델 학습 및 평가 시작 ===")
train(dataset, args)
print("=== 모델 학습 및 평가 완료 ===")