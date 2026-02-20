import numpy as np
import os
import pandas as pd
import time
import torch
import torch.nn.functional as F
import wandb

from collections import defaultdict
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config.retriever import load_yaml
from src.dataset.retriever import RetrieverDataset, collate_retriever
from src.dataset.motifs import MOTIF_VOCAB_SIZE
from src.model.retriever import Retriever
from src.setup import set_seed, prepare_sample

@torch.no_grad()
def eval_epoch(config, device, data_loader, model):
    model.eval()
    
    metric_dict = defaultdict(list)
    
    for sample in tqdm(data_loader):
        h_id_tensor, r_id_tensor, t_id_tensor, q_emb, entity_embs,\
        num_non_text_entities, relation_embs, topic_entity_one_hot,\
        target_triple_probs, a_entity_id_list, node_motif_token_ids,\
        node_motif_token_wts, triple_motif_token_ids, triple_motif_token_wts = prepare_sample(device, sample)

        pred_triple_logits = model(
            h_id_tensor, r_id_tensor, t_id_tensor, q_emb, entity_embs,
            num_non_text_entities, relation_embs, topic_entity_one_hot,
            node_motif_token_ids, node_motif_token_wts,
            triple_motif_token_ids, triple_motif_token_wts).reshape(-1)
        
        # Triple ranking
        sorted_triple_ids_pred = torch.argsort(
            pred_triple_logits, descending=True).cpu()
        triple_ranks_pred = torch.empty_like(sorted_triple_ids_pred)
        triple_ranks_pred[sorted_triple_ids_pred] = torch.arange(
            len(triple_ranks_pred))
        
        target_triple_ids = target_triple_probs.nonzero().squeeze(-1)
        num_target_triples = len(target_triple_ids)
        
        if num_target_triples == 0:
            continue

        num_total_entities = len(entity_embs) + num_non_text_entities
        for k in config['eval']['k_list']:
            recall_k_sample = (
                triple_ranks_pred[target_triple_ids] < k).sum().item()
            metric_dict[f'triple_recall@{k}'].append(
                recall_k_sample / num_target_triples)
            
            triple_mask_k = triple_ranks_pred < k
            entity_mask_k = torch.zeros(num_total_entities)
            entity_mask_k[h_id_tensor[triple_mask_k]] = 1.
            entity_mask_k[t_id_tensor[triple_mask_k]] = 1.
            recall_k_sample_ans = entity_mask_k[a_entity_id_list].sum().item()
            metric_dict[f'ans_recall@{k}'].append(
                recall_k_sample_ans / len(a_entity_id_list))

    for key, val in metric_dict.items():
        metric_dict[key] = np.mean(val)
    
    return metric_dict


def get_motif_dist(triple_scores, triple_motif_token_ids, triple_motif_token_wts, vocab_size, eps=1e-8):
    coeff = triple_scores.unsqueeze(-1).expand_as(triple_motif_token_wts)
    values = (coeff * triple_motif_token_wts).reshape(-1)
    ids = triple_motif_token_ids.reshape(-1)
    motif_mass = torch.zeros(vocab_size, device=triple_scores.device)
    motif_mass = motif_mass.scatter_add(0, ids, values)
    motif_mass[0] = 0.0
    norm = motif_mass.sum()
    if norm <= eps:
        return None
    return motif_mass / norm

def train_epoch(device, train_loader, model, optimizer):
    model.train()
    epoch_loss = 0
    epoch_loss_bce = 0
    epoch_loss_kl = 0
    gate_sum = None
    gate_cnt = 0
    motif_kl_weight = model.motif_cfg.get('motif_kl_weight', 0.1) if model.motif_enabled else 0.0
    motif_vocab_size = model.motif_cfg.get('vocab_size', MOTIF_VOCAB_SIZE) if model.motif_enabled else MOTIF_VOCAB_SIZE

    for sample in tqdm(train_loader):
        h_id_tensor, r_id_tensor, t_id_tensor, q_emb, entity_embs,\
        num_non_text_entities, relation_embs, topic_entity_one_hot,\
        target_triple_probs, a_entity_id_list, node_motif_token_ids,\
        node_motif_token_wts, triple_motif_token_ids, triple_motif_token_wts = prepare_sample(device, sample)
            
        if len(h_id_tensor) == 0:
            continue

        model_out = model(
            h_id_tensor, r_id_tensor, t_id_tensor, q_emb, entity_embs,
            num_non_text_entities, relation_embs, topic_entity_one_hot,
            node_motif_token_ids, node_motif_token_wts,
            triple_motif_token_ids, triple_motif_token_wts,
            return_aux=True)
        if isinstance(model_out, tuple):
            pred_triple_logits, aux = model_out
        else:
            pred_triple_logits = model_out
            aux = {}

        target_triple_probs = target_triple_probs.to(device).unsqueeze(-1)
        loss_bce = F.binary_cross_entropy_with_logits(
            pred_triple_logits, target_triple_probs)
        loss_kl = torch.tensor(0.0, device=device)

        if model.motif_enabled and motif_kl_weight > 0:
            pred_probs = torch.sigmoid(pred_triple_logits).reshape(-1)
            target_probs = target_triple_probs.reshape(-1)
            target_dist = get_motif_dist(
                target_probs,
                triple_motif_token_ids,
                triple_motif_token_wts,
                motif_vocab_size,
            )
            pred_dist = get_motif_dist(
                pred_probs,
                triple_motif_token_ids,
                triple_motif_token_wts,
                motif_vocab_size,
            )
            if (target_dist is not None) and (pred_dist is not None):
                loss_kl = F.kl_div(torch.log(pred_dist + 1e-8), target_dist, reduction='batchmean')

        loss = loss_bce + motif_kl_weight * loss_kl
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_loss_bce += loss_bce.item()
        epoch_loss_kl += loss_kl.item()
        gate = aux.get('gate', None)
        if gate is not None:
            gate_sum = gate if gate_sum is None else gate_sum + gate
            gate_cnt += 1
    
    epoch_loss /= len(train_loader)
    epoch_loss_bce /= len(train_loader)
    epoch_loss_kl /= len(train_loader)
    
    log_dict = {
        'loss': epoch_loss,
        'loss_bce': epoch_loss_bce,
        'loss_motif_kl': epoch_loss_kl,
    }
    if gate_sum is not None and gate_cnt > 0:
        gate_mean = (gate_sum / gate_cnt).detach().cpu().tolist()
        log_dict.update({
            'gate_neighborhood': gate_mean[0],
            'gate_position': gate_mean[1],
            'gate_structure': gate_mean[2],
        })
    return log_dict

def main(args):
    # Modify the config file for advanced settings and extensions.
    config_file = f'configs/retriever/{args.dataset}.yaml'
    config = load_yaml(config_file)
    
    device = torch.device('cuda:0')
    torch.set_num_threads(config['env']['num_threads'])
    set_seed(config['env']['seed'])

    ts = time.strftime('%b%d-%H:%M:%S', time.gmtime())
    config_df = pd.json_normalize(config, sep='/')
    exp_prefix = config['train']['save_prefix']
    exp_name = f'{exp_prefix}_{ts}'
    wandb.init(
        project=f'{args.dataset}',
        name=exp_name,
        config=config_df.to_dict(orient='records')[0]
    )
    os.makedirs(exp_name, exist_ok=True)

    train_set = RetrieverDataset(config=config, split='train')
    val_set = RetrieverDataset(config=config, split='val')

    train_loader = DataLoader(
        train_set, batch_size=1, shuffle=True, collate_fn=collate_retriever)
    val_loader = DataLoader(
        val_set, batch_size=1, collate_fn=collate_retriever)
    
    emb_size = train_set[0]['q_emb'].shape[-1]
    motif_cfg = config.get('motif', {})
    if 'vocab_size' not in motif_cfg:
        motif_cfg['vocab_size'] = MOTIF_VOCAB_SIZE
    motif_cfg['motif_kl_weight'] = config.get('loss', {}).get('motif_kl_weight', 0.1)
    model = Retriever(emb_size, motif=motif_cfg, **config['retriever']).to(device)
    optimizer = Adam(model.parameters(), **config['optimizer'])

    num_patient_epochs = 0
    best_val_metric = 0
    for epoch in range(config['train']['num_epochs']):
        num_patient_epochs += 1
        
        val_eval_dict = eval_epoch(config, device, val_loader, model)
        target_val_metric = val_eval_dict['triple_recall@100']
        
        if target_val_metric > best_val_metric:
            num_patient_epochs = 0
            best_val_metric = target_val_metric
            best_state_dict = {
                'config': config,
                'model_state_dict': model.state_dict()
            }
            torch.save(best_state_dict, os.path.join(exp_name, f'cpt.pth'))

            val_log = {'val/epoch': epoch}
            for key, val in val_eval_dict.items():
                val_log[f'val/{key}'] = val
            wandb.log(val_log)

        train_log_dict = train_epoch(device, train_loader, model, optimizer)
        
        train_log_dict.update({
            'num_patient_epochs': num_patient_epochs,
            'epoch': epoch
        })
        wandb.log(train_log_dict)
        if num_patient_epochs == config['train']['patience']:
            break

if __name__ == '__main__':
    from argparse import ArgumentParser
    
    parser = ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, required=True, 
                        choices=['webqsp', 'cwq'], help='Dataset name')
    args = parser.parse_args()
    
    main(args)
