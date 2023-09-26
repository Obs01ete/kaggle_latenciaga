import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import SAGEConv
from torch_geometric.nn import aggr


class Model(nn.Module):
    NUM_NODE_FEATURES = 140
    NUM_OPCODES = 120

    def __init__(self,
                 full_batch_size: int,
                 microbatch_size: int,
                 node_config_feat_size: int = 18,
                 ):

        super().__init__()

        # self.full_batch_size = full_batch_size # does not work with the remainder batch
        self.microbatch_size = microbatch_size

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
                batch: torch.Tensor,
                ptr: torch.Tensor,

                node_config_feat: torch.Tensor,
                node_config_ids: torch.Tensor,
                node_config_batch: torch.Tensor,
                node_config_ptr: torch.Tensor,
                
                edge_index: torch.Tensor
                ) -> torch.Tensor:
        """
            DataBatch(
                node_feat=[525076, 140],
                node_opcode=[525076],
                batch=[525076],
                ptr=[41],

                node_config_feat=[35496, 18],
                node_config_ids=[35496],
                node_config_batch=[35496],
                node_config_ptr=[41],

                edge_index=[2, 896088],
                )
        """

        batch_size = ptr.shape[0] - 1
        if batch_size % self.microbatch_size != 0:
            print(f"Warning: batch size {batch_size} not divisible "
                  f"by microbatch size {self.microbatch_size}. "
                  f"Fine for val, error for train.")
        num_nodes = node_feat.shape[0]
        # num_configs = node_config_feat.shape[0]
        config_feat_size = node_config_feat.shape[1]

        node_feat_abs = torch.relu(node_feat) # discard negative numbers
        node_feat_log = torch.log1p(node_feat_abs)
        node_feat_emb = self.node_feat_embedding(node_feat_log)
        node_opcode_emb = self.node_opcode_embedding(node_opcode.long())
        config_feat_all = torch.zeros(size=(num_nodes, config_feat_size),
                                      dtype=torch.float32, device=node_feat.device)

        for ib in range(batch_size):
            config_slice = slice(node_config_ptr[ib],
                                 node_config_ptr[ib+1])
            sample_config_ids = node_config_ids[config_slice]
            sample_config_feat = node_config_feat[config_slice]
            
            global_config_ids = sample_config_ids + ptr[ib]
            config_feat_all[global_config_ids, :] = sample_config_feat

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
