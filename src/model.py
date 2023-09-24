import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import SAGEConv
from torch_geometric.nn import aggr


class Model(nn.Module):
    NUM_NODE_FEATURES = 140
    NUM_OPCODES = 120

    def __init__(self, node_config_feat_size: int = 18):
        super().__init__()

        node_feat_emb_size = 20
        node_opcode_emb_size = 12

        self.node_feat_embedding = nn.Linear(self.NUM_NODE_FEATURES,
                                             node_feat_emb_size)
        self.node_opcode_embedding = nn.Embedding(self.NUM_OPCODES,
                                                  node_opcode_emb_size)
        
        concat_node_feat_size = (node_feat_emb_size +
                                 node_opcode_emb_size +
                                 node_config_feat_size)
        
        in_channels = 32
        channel_config = [64, 64, 128, 128, 256, 256]
        assert len(channel_config) > 0

        self.input_shaping = nn.Linear(concat_node_feat_size, in_channels)

        self.convs = nn.ModuleList()
        in_ch = in_channels
        for out_ch in channel_config:
            conv = SAGEConv(in_ch, out_ch)
            self.convs.append(conv)
            in_ch = out_ch

        REGRESSION_SIZE = 1
        self.output_shaping = nn.Linear(channel_config[-1], REGRESSION_SIZE)

        self.aggr_sum = aggr.SumAggregation()
    
    def forward(self,
                node_feat: torch.Tensor,
                node_opcode: torch.Tensor,
                node_config_feat: torch.Tensor,
                node_config_ids: torch.Tensor,
                edge_index: torch.Tensor,
                batch: torch.Tensor,
                ) -> torch.Tensor:
        """
        DataBatch(
            edge_index=[2, 2388],
            node_feat=[1488, 140],
            node_opcode=[1488],
            node_config_feat=[104, 18],
            node_config_ids=[104],
            config_runtime=[4],
            batch=[1488])
        """

        num_nodes = node_feat.shape[-2]
        config_feat_size = node_config_feat.shape[-1]

        node_feat_abs = torch.relu(node_feat) # discard negative numbers
        node_feat_log = torch.log1p(node_feat_abs)
        node_feat_emb = self.node_feat_embedding(node_feat_log)
        node_opcode_emb = self.node_opcode_embedding(node_opcode.long())
        config_feat_all = torch.zeros(size=(num_nodes, config_feat_size),
                                      dtype=torch.float, device=node_feat.device)
        # maybe there is a bug here, TODO make a test
        config_feat_all.scatter_(-2, node_config_ids.unsqueeze(-1), node_config_feat)

        node_feat_all = torch.cat((node_feat_emb,
                                   node_opcode_emb,
                                   config_feat_all), dim=-1)
        
        x = F.relu(self.input_shaping(node_feat_all))
    
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))

        x = self.output_shaping(x)

        x = self.aggr_sum(x, batch)

        x = x.squeeze(-1)

        SCALE_WELL = 1e-3
        x = SCALE_WELL * x

        return x
