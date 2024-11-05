
import argparse
import os
import pickle
import pandas as pd

from sklearn.model_selection import train_test_split
from builder import PandasGraphBuilder
from data_utils import *
import dgl
from scipy.sparse import csr_matrix


# 유사도 함수 수정해야함 -> 유사도 어떻게 계산할지 정하기
def cal_similarity(creator_s,item_s):
    if creator_s == item_s:  # 수정
        return 1.0  # 유사도가 완전히 일치하면 1
    else:
        return 0.0  # 일치하지 않으면 0


# Load data
def process_data(directory,out_directory):

    """
    크리에이터와 기획서 데이터를 처리하여 DGL 그래프를 생성하는 함수.
    :param directory: 입력 데이터가 있는 디렉터리
    :param out_directory: 출력 그래프 파일을 저장할 디렉터리
    """

    # pandas data load
    creator_df = pd.read_csv(os.path.join(directory, "Creator_random.csv"))
    item_df = pd.read_csv(os.path.join(directory, "Item_random.csv"))

    # Build graph
    graph_builder = PandasGraphBuilder()

    # 1. Add Creator Node
    graph_builder.add_entities(creator_df, 'creator_id', 'creator')

    # 2. Add item Node
    graph_builder.add_entities(item_df, 'item_id', 'item')

    # 3. Add similarity base edge

    edges_src =[]
    edges_dst = []
    similarities = []

    # calculate similarity between creator & item
    for i, creator_row in creator_df.iterrows():
        for j, proposal_row in item_df.iterrows():
            similarity = cal_similarity(
                creator_row['channel_category'], proposal_row['item_category']
            )
            if similarity > 0:  # 유사도가 0 이상일 때만 엣지 추가
                edges_src.append(creator_row['creator_id'])
                edges_dst.append(proposal_row['item_id'])
                similarities.append(similarity)

    edge_df = pd.DataFrame({
        'creator_id': edges_src,
        'item_id': edges_dst,
        'similarity': similarities
    })



    # Add binary relations to the graph
    # creator -> item 방향의 엣지
    graph_builder.add_binary_relations(edge_df, 'creator_id', 'item_id', 'creator_to_item')
    # item -> creator 방향의 엣지
    graph_builder.add_binary_relations(edge_df, 'item_id', 'creator_id', 'item_to_creator')

    # 4. build Graph
    g = graph_builder.build()

    print("노드 타입:", g.ntypes)  # 노드 타입 확인
    print("엣지 타입:", g.etypes)  # 엣지 타입 확인
    print("메타그래프:", g.metagraph().edges)  # 메타그래프 엣지 확인

    # 5. Assign features to Node
    for feature in ["channel_name", "channel_category", "max_views", "min_views", "media_type", "subscribers",
                    "comments"]:
        if creator_df[feature].dtype == 'int64':
            g.nodes["creator"].data[feature] = torch.LongTensor(creator_df[feature].values)
        else:
            g.nodes["creator"].data[feature] = torch.LongTensor(pd.factorize(creator_df[feature])[0])  # Embedding 수정해야할 수도 있음

    for feature in ["title", "item_category", "media_type", "score", "item_content"] :
        if item_df[feature].dtype == 'int64':
            g.nodes["item"].data[feature] = torch.LongTensor(item_df[feature].values)
        else:
            g.nodes["item"].data[feature] = torch.LongTensor(pd.factorize(item_df[feature])[0])

    # 6. Assign feature to Edge
    g.edges[("creator", "creator_to_item", "item")].data['similarity'] = torch.FloatTensor(similarities)
    g.edges[("item", "item_to_creator", "creator")].data['similarity'] = torch.FloatTensor(similarities)

    # 7. Train-validation-test split
    # This is a little bit tricky as we want to select the last interaction for test, and the
    # second-to-last interaction for validation.

    # Train-Validation-Test Split (무작위 분할)
    train_indices, temp_indices = train_test_split(edge_df.index, test_size=0.3, random_state=42)
    val_indices, test_indices = train_test_split(temp_indices, test_size=0.5, random_state=42)


    # Train graph generation
    # Train graph generation
    train_g = dgl.edge_subgraph(
        g,
        {
            ("creator", "creator_to_item", "item"): train_indices,
            ("item", "item_to_creator", "creator"): train_indices,
        },
    )
    # 검증 및 테스트용 Sparse Matrix 생성
    # Validation and Test Sparse Matrix generation
    val_matrix = csr_matrix(
        edge_df.iloc[val_indices].pivot(index='creator_id', columns='item_id', values='similarity').fillna(0).values)
    test_matrix = csr_matrix(
        edge_df.iloc[test_indices].pivot(index='creator_id', columns='item_id', values='similarity').fillna(0).values)

    # 그래프 및 데이터셋 저장
    os.makedirs(out_directory, exist_ok=True)
    dgl.save_graphs(os.path.join(out_directory, "train_g.bin"), train_g)

    dataset = {
        "val-matrix": val_matrix,
        "test-matrix": test_matrix,
        "user-type": "creator",
        "item-type": "item",
        "user-to-item-type": "creator_to_item",
        "item-to-user-type": "item_to_creator",
    }

    # item-texts 추가
    item_texts = item_df['item_content'].tolist()  # 'item_content' 열을 'item-texts'로 추가
    dataset["item-texts"] = item_texts

    with open(os.path.join(out_directory, "data.pkl"), "wb") as f:
        pickle.dump(dataset, f)

    return "Graph and dataset successfully saved!"


# 메인 함수
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=str)
    parser.add_argument("out_directory", type=str)
    args = parser.parse_args()

    process_data(args.directory, args.out_directory)

'''
DGL은 torch 2.4에서 미구현 되어 2.3으로 다운 그레이드
!pip uninstall torch -y
!pip install torch==2.1.2 torchvisio n==0.18.1 torchaudio==2.3.1


!pip install  dgl -f https://data.dgl.ai/wheels/torch-2.3/cu121/repo.html


pip install  dgl -f https://data.dgl.ai/wheels/torch-2.3/repo.html
pip install dgl -f https://data.dgl.ai/wheels/repo.html
pip install torch==2.3.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
'''

"""
output
train_g.bin: 학습용 그래프가 저장된 DGL 그래프 파일.
data.pkl: 검증 및 테스트 행렬과 메타 정보를 포함한 데이터셋 파일.
"""