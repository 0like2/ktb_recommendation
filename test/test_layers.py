import dgl
import torch
from layers import LinearProjector, WeightedSAGEConv, SAGENet, ItemToItemScorer

# 1. 테스트용 그래프 생성
def create_test_graph():
    num_nodes = 10
    edges = (torch.tensor([0, 1, 2, 3]), torch.tensor([1, 2, 3, 4]))
    g = dgl.heterograph({('node_type', 'follows', 'node_type'): edges}, num_nodes_dict={'node_type': num_nodes})
    g.edata['weights'] = torch.rand(g.num_edges())  # 엣지 가중치 추가
    g.nodes['node_type'].data['feature'] = torch.randn(num_nodes, 8)  # 노드 특징 추가
    g.nodes['node_type'].data[dgl.NID] = torch.arange(num_nodes)
    return g

# 2. LinearProjector 테스트
def test_linear_projector():
    g = create_test_graph()
    hidden_dims = 8
    ntype = "node_type"  # 노드 타입 지정
    textset = None  # 텍스트 데이터는 테스트에서 생략
    projector = LinearProjector(g, ntype, textset, hidden_dims)
    projected_features = projector(g.nodes[ntype].data)
    print("LinearProjector Output Shape:", projected_features.shape)

# 3. WeightedSAGEConv 테스트
def test_weighted_sage_conv():
    g = create_test_graph()
    hidden_dims = 8
    output_dims = 8
    h = (g.nodes['node_type'].data['feature'], g.nodes['node_type'].data['feature'])
    conv = WeightedSAGEConv(hidden_dims, hidden_dims, output_dims)
    output = conv(g, h, g.edata['weights'])
    print("WeightedSAGEConv Output Shape:", output.shape)

# 4. SAGENet 테스트
def test_sagenet():
    g = create_test_graph()
    hidden_dims = 8
    n_layers = 2

    # 노드 특징을 사용해 블록 생성
    blocks = []
    for _ in range(n_layers):
        block = dgl.to_block(g)  # g를 블록으로 변환
        block.edata['weights'] = g.edata['weights']  # 엣지 가중치 복사
        blocks.append(block)

    sagenet = SAGENet(hidden_dims, n_layers)

    # 각 블록 내에서 src와 dst 노드에 맞춘 특징을 전달
    h = g.nodes['node_type'].data['feature']
    for block in blocks:
        h_src = h[block.srcdata[dgl.NID]]  # 블록의 src 노드 특징
        h_dst = h[block.dstdata[dgl.NID]]  # 블록의 dst 노드 특징
        output = sagenet([block], h_src)  # 블록을 하나씩 전달하며 테스트
        print("SAGENet Output Shape:", output.shape)


# 5. ItemToItemScorer 테스트
def test_item_to_item_scorer():
    g = create_test_graph()
    ntype = "node_type"  # 노드 타입 지정
    scorer = ItemToItemScorer(g, ntype)

    # item_item_graph 생성 및 노드 ID 추가
    item_item_graph = dgl.heterograph({('node_type', 'similar', 'node_type'): g.edges()},
                                      num_nodes_dict={'node_type': g.num_nodes('node_type')})
    item_item_graph.ndata[dgl.NID] = torch.arange(item_item_graph.num_nodes())  # 노드 ID 추가

    # 임의의 숨겨진 상태 생성
    h = torch.randn(g.num_nodes('node_type'), 8)

    # 스코어 계산
    pair_score = scorer(item_item_graph, h)
    print("ItemToItemScorer Pair Score Shape:", pair_score.shape)


# 모든 테스트 실행
if __name__ == "__main__":
    print("Testing LinearProjector...")
    test_linear_projector()
    print("\nTesting WeightedSAGEConv...")
    test_weighted_sage_conv()
    print("\nTesting SAGENet...")
    test_sagenet()
    print("\nTesting ItemToItemScorer...")
    test_item_to_item_scorer()
