import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import MessagePassing

class PEConv(MessagePassing):
    def __init__(self):
        super().__init__(aggr='mean')

    def forward(self, edge_index, x):
        return self.propagate(edge_index, x=x)

    def message(self, x_j):
        return x_j

class DDE(nn.Module):
    def __init__(
        self,
        num_rounds,
        num_reverse_rounds
    ):
        super().__init__()
        
        self.layers = nn.ModuleList()
        for _ in range(num_rounds):
            self.layers.append(PEConv())
        
        self.reverse_layers = nn.ModuleList()
        for _ in range(num_reverse_rounds):
            self.reverse_layers.append(PEConv())
    
    def forward(
        self,
        topic_entity_one_hot,
        edge_index,
        reverse_edge_index
    ):
        result_list = []
        
        h_pe = topic_entity_one_hot
        for layer in self.layers:
            h_pe = layer(edge_index, h_pe)
            result_list.append(h_pe)
        
        h_pe_rev = topic_entity_one_hot
        for layer in self.reverse_layers:
            h_pe_rev = layer(reverse_edge_index, h_pe_rev)
            result_list.append(h_pe_rev)
        
        return result_list

class Retriever(nn.Module):
    def __init__(
        self,
        emb_size,
        topic_pe,
        DDE_kwargs,
        hidden_dim=256,
        motif=None,
    ):
        super().__init__()
        
        self.non_text_entity_emb = nn.Embedding(1, emb_size)
        self.topic_pe = topic_pe
        self.dde = DDE(**DDE_kwargs)
        self.motif_cfg = motif or {}
        self.motif_enabled = self.motif_cfg.get('enabled', False)
        self.hidden_dim = hidden_dim

        if self.motif_enabled:
            motif_vocab_size = self.motif_cfg.get('vocab_size', 17)
            motif_emb_dim = self.motif_cfg.get('motif_emb_dim', 64)
            self.motif_emb = nn.Embedding(motif_vocab_size, motif_emb_dim, padding_idx=0)
            self.query_cross_attn_enabled = self.motif_cfg.get('query_cross_attn_enabled', False)
            self.motif_residual_blend_enabled = self.motif_cfg.get('motif_residual_blend_enabled', False)
            if self.query_cross_attn_enabled or self.motif_residual_blend_enabled:
                self.q_to_motif = nn.Linear(emb_size, motif_emb_dim)
            if self.motif_residual_blend_enabled:
                init_alpha = self.motif_cfg.get('motif_residual_init_alpha', 0.2)
                self.motif_residual_alpha = nn.Parameter(torch.tensor(float(init_alpha)))

            pos_node_dim = (2 if topic_pe else 0) + 2 * (
                DDE_kwargs['num_rounds'] + DDE_kwargs['num_reverse_rounds']
            )
            neighborhood_in = 2 * emb_size + 2 * motif_emb_dim
            position_in = 2 * pos_node_dim
            structure_in = emb_size + motif_emb_dim

            self.neighborhood_head = nn.Sequential(
                nn.Linear(neighborhood_in, hidden_dim),
                nn.ReLU(),
            )
            self.position_head = nn.Sequential(
                nn.Linear(position_in, hidden_dim),
                nn.ReLU(),
            )
            self.structure_head = nn.Sequential(
                nn.Linear(structure_in, hidden_dim),
                nn.ReLU(),
            )
            self.channel_gate = nn.Linear(emb_size, 3)
            self.pred = nn.Sequential(
                nn.Linear(emb_size + hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )
        else:
            self.query_cross_attn_enabled = False
            self.motif_residual_blend_enabled = False
            pred_in_size = 4 * emb_size
            if topic_pe:
                pred_in_size += 2 * 2
            pred_in_size += 2 * 2 * (DDE_kwargs['num_rounds'] + DDE_kwargs['num_reverse_rounds'])
            self.pred = nn.Sequential(
                nn.Linear(pred_in_size, emb_size),
                nn.ReLU(),
                nn.Linear(emb_size, 1)
            )

    def _aggregate_motif_emb(self, token_ids, token_wts):
        token_emb = self.motif_emb(token_ids)
        return (token_emb * token_wts.unsqueeze(-1)).sum(dim=1)

    def _aggregate_motif_emb_query_conditioned(self, token_ids, token_wts, q_emb):
        token_emb = self.motif_emb(token_ids)  # [N, K, D]
        q_proj = self.q_to_motif(q_emb).squeeze(0)  # [D]
        attn_logits = (token_emb * q_proj.unsqueeze(0).unsqueeze(0)).sum(dim=-1)  # [N, K]

        # Respect precomputed motif weights while making them query-dependent.
        attn_logits = attn_logits + torch.log(token_wts + 1e-8)
        pad_mask = token_ids.eq(0)
        attn_logits = attn_logits.masked_fill(pad_mask, float('-inf'))
        all_pad = pad_mask.all(dim=1)
        attn_logits = torch.where(
            all_pad.unsqueeze(1),
            torch.zeros_like(attn_logits),
            attn_logits,
        )
        attn = F.softmax(attn_logits, dim=1)
        attn = torch.where(all_pad.unsqueeze(1), torch.zeros_like(attn), attn)
        return (token_emb * attn.unsqueeze(-1)).sum(dim=1)

    def forward(
        self,
        h_id_tensor,
        r_id_tensor,
        t_id_tensor,
        q_emb,
        entity_embs,
        num_non_text_entities,
        relation_embs,
        topic_entity_one_hot,
        node_motif_token_ids=None,
        node_motif_token_wts=None,
        triple_motif_token_ids=None,
        triple_motif_token_wts=None,
        return_aux=False,
    ):
        device = entity_embs.device
        
        h_e = torch.cat(
            [
                entity_embs,
                self.non_text_entity_emb(
                    torch.LongTensor([0]).to(device)).expand(num_non_text_entities, -1)
            ]
        , dim=0)
        h_e_list = [h_e]
        if self.topic_pe:
            h_e_list.append(topic_entity_one_hot)

        edge_index = torch.stack([
            h_id_tensor,
            t_id_tensor
        ], dim=0)
        reverse_edge_index = torch.stack([
            t_id_tensor,
            h_id_tensor
        ], dim=0)
        dde_list = self.dde(topic_entity_one_hot, edge_index, reverse_edge_index)
        h_e_list.extend(dde_list)
        h_e_full = torch.cat(h_e_list, dim=1)

        h_q = q_emb
        # Potentially memory-wise problematic
        h_r = relation_embs[r_id_tensor]

        if not self.motif_enabled:
            h_triple = torch.cat([
                h_q.expand(len(h_r), -1),
                h_e_full[h_id_tensor],
                h_r,
                h_e_full[t_id_tensor]
            ], dim=1)
            pred = self.pred(h_triple)
            if return_aux:
                return pred, {}
            return pred

        if self.query_cross_attn_enabled:
            node_motif_emb = self._aggregate_motif_emb_query_conditioned(
                node_motif_token_ids, node_motif_token_wts, h_q
            )
            triple_motif_emb = self._aggregate_motif_emb_query_conditioned(
                triple_motif_token_ids, triple_motif_token_wts, h_q
            )
        elif self.motif_residual_blend_enabled:
            node_motif_emb_static = self._aggregate_motif_emb(node_motif_token_ids, node_motif_token_wts)
            triple_motif_emb_static = self._aggregate_motif_emb(triple_motif_token_ids, triple_motif_token_wts)
            node_motif_emb_query = self._aggregate_motif_emb_query_conditioned(
                node_motif_token_ids, node_motif_token_wts, h_q
            )
            triple_motif_emb_query = self._aggregate_motif_emb_query_conditioned(
                triple_motif_token_ids, triple_motif_token_wts, h_q
            )
            node_motif_emb = node_motif_emb_static + self.motif_residual_alpha * node_motif_emb_query
            triple_motif_emb = triple_motif_emb_static + self.motif_residual_alpha * triple_motif_emb_query
        else:
            node_motif_emb = self._aggregate_motif_emb(node_motif_token_ids, node_motif_token_wts)
            triple_motif_emb = self._aggregate_motif_emb(triple_motif_token_ids, triple_motif_token_wts)

        h_h = h_e[h_id_tensor]
        h_t = h_e[t_id_tensor]
        m_h = node_motif_emb[h_id_tensor]
        m_t = node_motif_emb[t_id_tensor]
        h_neighborhood = self.neighborhood_head(torch.cat([h_h, h_t, m_h, m_t], dim=1))

        pos_node_features = []
        if self.topic_pe:
            pos_node_features.append(topic_entity_one_hot)
        pos_node_features.extend(dde_list)
        pos_node_features = torch.cat(pos_node_features, dim=1)
        h_position = self.position_head(torch.cat(
            [pos_node_features[h_id_tensor], pos_node_features[t_id_tensor]], dim=1
        ))

        h_structure = self.structure_head(torch.cat([h_r, triple_motif_emb], dim=1))

        gate_logits = self.channel_gate(h_q).squeeze(0)
        gate = F.softmax(gate_logits, dim=0)
        h_fused = gate[0] * h_neighborhood + gate[1] * h_position + gate[2] * h_structure

        pred_in = torch.cat([h_q.expand(len(h_r), -1), h_fused], dim=1)
        pred = self.pred(pred_in)

        if return_aux:
            return pred, {'gate': gate.detach()}
        return pred
