import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GATConv, global_mean_pool, GraphNorm

# from torch_geometric.transforms import NormalizeFeatures


class GATModel(torch.nn.Module):
    def __init__(self, node_features_dim, output_features_dim, hidden_dim=16, num_heads=4):
        super().__init__()
        # [ ] 调整模型结构
        self.model = nn.Sequential(
            [
                GATConv(node_features_dim, hidden_dim, num_heads),
                GraphNorm(hidden_dim),
                GATConv(hidden_dim, output_features_dim, num_heads),
                GraphNorm(output_features_dim),
            ]
        )
        return

    def forward(self, batch: Batch):
        x, edge_index, batch_index = batch.x, batch.edge_index, batch.batch
        x = self.model(x, edge_index)
        x = global_mean_pool(x, batch_index)  # 使用全局平均池化以保持每个图的独立性,这将对每个图的节点进行平均
        return x


if __name__ == "__main__":
    # 假设node_features_dim为特征的维度，output_features_dim为输出的特征维度
    node_features_dim = 3  # 节点特征的维数
    output_features_dim = 10  # 输出特征的维数

    model = GATModel(node_features_dim, output_features_dim)

    # 构造一批图数据
    graphs = []
    for _ in range(5):  # 假设有5个图
        num_nodes = torch.randint(3, 10, (1,)).item()  # 随机节点数
        # 随机节点特征
        x = torch.rand((num_nodes, node_features_dim))
        # 随机边
        edge_index = torch.randint(0, num_nodes, (2, num_nodes * 2))
        graphs.append(Data(x=x, edge_index=edge_index))

    batch = Batch.from_data_list(graphs)

    # 前向传播
    model.eval()
    output_features = model(batch)
    print(output_features.shape)  # 输出应该是 (5, output_features_dim)
