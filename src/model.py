from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import SAGEConv
from torch_geometric.nn import aggr

from src.utils import make_diff_matrix, triu_vector


class Model(nn.Module):
    NUM_NODE_FEATURES = 140
    NUM_OPCODES = 120

    def __init__(self,
                 is_tile: bool,
                 is_nlp: bool, # unused for now
                 is_default: bool, # unused for now
                 wider_config: bool,
                 node_config_feat_size: int = 18,
                 tile_config_feat_size: int = 24,
                 ):
        """
        Args:
            is_tile (bool): False: layout, True: tile
            is_nlp (bool): False: xla, True: nlp
            is_default (bool): False: random, True: default
        """

        super().__init__()

        node_feat_emb_size = 20
        node_opcode_emb_size = 12

        self.node_feat_embedding = nn.Linear(self.NUM_NODE_FEATURES,
                                             node_feat_emb_size)
        self.node_opcode_embedding = nn.Embedding(self.NUM_OPCODES,
                                                  node_opcode_emb_size)
        
        config_feat_size = tile_config_feat_size if is_tile else node_config_feat_size
        concat_node_feat_size = (node_feat_emb_size +
                                 node_opcode_emb_size +
                                 config_feat_size)
        
        if is_tile or is_nlp or wider_config: # enable wider config for tile and for nlp by default
            in_channels = 64
            channel_config = [256, 256, 256, 256, 512, 512, 512, 512]
        else:
            in_channels = 32
            channel_config = [64, 64, 128, 128, 256, 256]
        assert len(channel_config) > 0

        self.add_residuals: bool
        if is_nlp:
            self.add_residuals = True
        else:
            self.add_residuals = False

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
                node_config_ptr: torch.Tensor,

                config_feat: torch.Tensor,
                config_feat_ptr: torch.Tensor,
                
                edge_index: torch.Tensor,

                ub_size: int, # microbatch_size

                ) -> Tuple[torch.Tensor, torch.Tensor]:
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

        is_tile = config_feat is not None

        SCALE_MS_TO_SEC = 1e-3

        batch_size = ptr.shape[0] - 1
        if batch_size % ub_size != 0:
            print(f"Warning: batch size {batch_size} not divisible "
                  f"by microbatch size {ub_size}. "
                  f"Fine for val, error for train.")
        num_nodes = node_feat.shape[0]
        if is_tile:
            config_feat_size = config_feat.shape[0] // batch_size
        else:
            config_feat_size = node_config_feat.shape[1]

        node_feat_abs = torch.relu(node_feat) # discard negative numbers
        node_feat_log = torch.log1p(node_feat_abs)
        node_feat_emb = self.node_feat_embedding(node_feat_log)
        node_opcode_emb = self.node_opcode_embedding(node_opcode.long())

        if is_tile:
            graph_config_list = []
            for ib in range(batch_size):
                config_slice = slice(config_feat_ptr[ib],
                                     config_feat_ptr[ib+1])
                num_nodes_in_graph = ptr[ib+1] - ptr[ib]
                graph_config = config_feat[config_slice]
                graph_config_tiled = torch.tile(graph_config.unsqueeze(0),
                                                (num_nodes_in_graph, 1))
                graph_config_list.append(graph_config_tiled)
            config_feat_all = torch.concat(graph_config_list)
        else:
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
        
        feat = F.relu(self.input_shaping(node_feat_all))
    
        for conv in self.convs:
            feat_out = conv(feat, edge_index)
            if self.add_residuals and (feat_out.shape[1] == feat.shape[1]):
                feat = feat_out + feat # resudual connection
            else:
                feat = feat_out
            feat = F.relu(feat)

        per_node_latencies_unsq = self.output_shaping(feat)

        # branch for MAPE
        per_graph_latenies_ms = self.aggr_sum(per_node_latencies_unsq, batch)
        per_graph_latenies_ms_sq = per_graph_latenies_ms.squeeze(-1)
        if is_tile:
            per_graph_latenies = per_graph_latenies_ms_sq
        else:
            per_graph_latenies = SCALE_MS_TO_SEC * per_graph_latenies_ms_sq

        # branch for diff matrix
        assert batch_size % ub_size == 0
        num_microbatches = batch_size // ub_size
        diff_triu_vector_list = []
        for iub in range(num_microbatches):
            ub_slice = slice(iub*ub_size,
                             (iub+1)*ub_size)
            # per_ub_latencies [ub_size]
            per_ub_latencies = per_graph_latenies[ub_slice]
            # diff_matrix [ub_size, ub_size]
            diff_matrix = make_diff_matrix(per_ub_latencies)
            # triu_len = ub_size*(ub_size-1)/2. Ex triu_len=6 for ub_size=4.
            # diff_triu_vector [triu_len]
            diff_triu_vector = triu_vector(diff_matrix)
            diff_triu_vector_list.append(diff_triu_vector)

        # diff_triu_vector_stack [num_microbatches, triu_len]
        diff_triu_vector_stack = torch.stack(diff_triu_vector_list)

        return per_graph_latenies, diff_triu_vector_stack
