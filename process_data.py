import numpy as np
import dgl
import argparse
import os
import pickle
import pandas as pd

from sklearn.model_selection import train_test_split
from builder import PandasGraphBuilder
from data_utils import *
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

# 유사도 함수 (원-핫 인코딩 및 코사인 유사도)
categories = ["게임", "과학기술", "교육", "노하우/스타일", "뉴스/정치", "스포츠",
              "비영리/사회운동", "애완동물/동물", "엔터테인먼트", "여행/이벤트",
              "영화/애니메이션", "음악", "인물/블로그", "자동차/교통", "코미디", "기타"]

category_to_index = {cat: idx for idx, cat in enumerate(categories)}
num_categories = len(categories)

def one_hot_vector(category):
    vec = np.zeros(num_categories)
    if category in category_to_index:
        vec[category_to_index[category]] = 1
    return vec

def cal_similarity(creator_s, item_s):
    creator_vec = one_hot_vector(creator_s)
    item_vec = one_hot_vector(item_s)
    similarity = cosine_similarity([creator_vec], [item_vec])[0][0]
    return similarity

# Load data
def process_data(directory, out_directory):
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
    edges_src = []
    edges_dst = []
    similarities = []

    for i, creator_row in creator_df.iterrows():
        for j, item_row in item_df.iterrows():
            similarity = cal_similarity(creator_row['channel_category'], item_row['item_category'])
            if similarity > 0:
                edges_src.append(creator_row['creator_id'])
                edges_dst.append(item_row['item_id'])
                similarities.append(similarity)

    edge_df = pd.DataFrame({
        'creator_id': edges_src,
        'item_id': edges_dst,
        'similarity': similarities
    })

    if edge_df.empty:
        print("edge_df가 비어 있습니다. 유사도를 만족하는 엣지가 없습니다.")
        return
    else:
        print(edge_df.head())  # edge_df 내용 확인

    # Add binary relations to the graph
    # creator -> item 방향의 엣지
    graph_builder.add_binary_relations(edge_df, 'creator_id', 'item_id', 'creator_to_item')
    # item -> creator 방향의 엣지
    graph_builder.add_binary_relations(edge_df, 'item_id', 'creator_id', 'item_to_creator')

    # 5. build Graph
    g = graph_builder.build()

    print("노드 타입:", g.ntypes)  # 노드 타입 확인
    print("엣지 타입:", g.etypes)  # 엣지 타입 확인
    print("메타그래프:", g.metagraph().edges)  # 메타그래프 엣지 확인

    # 6. Assign features to Node
    for feature in ["channel_name", "channel_category", "max_views", "min_views", "media_type", "subscribers", "comments"]:
        if creator_df[feature].dtype == 'int64':
            g.nodes["creator"].data[feature] = torch.LongTensor(creator_df[feature].values)
        else:
            g.nodes["creator"].data[feature] = torch.LongTensor(pd.factorize(creator_df[feature])[0])

    for feature in ["title", "item_category", "media_type", "score", "item_content"]:
        if item_df[feature].dtype == 'int64':
            g.nodes["item"].data[feature] = torch.LongTensor(item_df[feature].values)
        else:
            g.nodes["item"].data[feature] = torch.LongTensor(pd.factorize(item_df[feature])[0])

    # 7. 엣지에 유사도 특징 할당
    g.edges[("creator", "creator_to_item", "item")].data['similarity'] = torch.FloatTensor(similarities)
    g.edges[("item", "item_to_creator", "creator")].data['similarity'] = torch.FloatTensor(similarities)

    # 8. 학습-검증-테스트 분할
    train_indices, temp_indices = train_test_split(edge_df.index, test_size=0.3, random_state=42)
    val_indices, test_indices = train_test_split(temp_indices, test_size=0.5, random_state=42)

    train_g = dgl.edge_subgraph(
        g,
        {
            ("creator", "creator_to_item", "item"): train_indices,
            ("item", "item_to_creator", "creator"): train_indices,
        },
    )

    val_matrix = csr_matrix(
        edge_df.iloc[val_indices].pivot(index='creator_id', columns='item_id', values='similarity').fillna(0).values)
    test_matrix = csr_matrix(
        edge_df.iloc[test_indices].pivot(index='creator_id', columns='item_id', values='similarity').fillna(0).values)

    # 텍스트 전처리 및 단어 집합 생성
    tokenizer = get_tokenizer("basic_english")
    textlist = [tokenizer(text.lower()) for text in item_df['item_content']]
    vocab2 = build_vocab_from_iterator(textlist, specials=["<unk>", "<pad>"])
    pad_token = vocab2["<pad>"]
    batch_first = True

    # textsets 저장 구조 생성
    item_texts = {
        "item-texts": (textlist, vocab2, pad_token, batch_first)
    }

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
        "item-texts": item_texts  #
    }

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
