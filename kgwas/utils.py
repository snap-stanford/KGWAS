import os, sys
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, precision_score
import torch
from torch.nn import functional as F 
from torch import nn
from multiprocessing import Pool

from .params import main_data_path, cohort_data_path, kinship_path, withdraw_path


def evaluate_minibatch_clean(loader, model, device):    
    model.eval()
    pred_all = []
    truth = []
    results = {}
    for step, batch in enumerate(tqdm(loader)):        
        batch = batch.to(device)
        bs_batch = batch['SNP'].batch_size
        
        out = model(batch.x_dict, batch.edge_index_dict, bs_batch)
        pred = out.reshape(-1)
        y_batch = batch['SNP'].y[:bs_batch]
        
        pred_all.extend(pred.detach().cpu().numpy())
        truth.extend(y_batch.detach().cpu().numpy())
        del y_batch, pred, batch, out
        
    results['pred'] = np.hstack(pred_all)
    results['truth'] = np.hstack(truth)
    return results

def compute_metrics(results, binary, coverage = None, uncertainty_reg = 1, loss_fct = None):
    metrics = {}
    metrics['mse'] = mean_squared_error(results['pred'], results['truth'])
    metrics['pearsonr'] = pearsonr(results['pred'], results['truth'])[0]
    return metrics


'''
requires to modify the pyg source code since it does not support heterogeneous graph attention

miniconda3/envs/a100_env/lib/python3.8/site-packages/torch_geometric/nn/conv/hgt_conv.py

def group(xs: List[Tensor], aggr: Optional[str]) -> Optional[Tensor]:
    if len(xs) == 0:
        return None
    elif aggr is None:
        return torch.stack(xs, dim=1)
    elif len(xs) == 1:
        return xs[0]
    elif isinstance(xs, list) and isinstance(xs[0], tuple):
        xs_old = [i[0] for i in xs]
        out = torch.stack(xs_old, dim=0)
        out = getattr(torch, aggr)(out, dim=0)
        out = out[0] if isinstance(out, tuple) else out        
        att = [i[1] for i in xs]
        return (out, att)
    else:
        out = torch.stack(xs, dim=0)
        out = getattr(torch, aggr)(out, dim=0)
        out = out[0] if isinstance(out, tuple) else out
        return out

'''


def get_attention_weight(model, x_dict, edge_index_dict):
    attention_all_layers = []
    for conv in model.convs:
        out = conv(x_dict, edge_index_dict, return_attention_weights_dict = dict(zip(list(data.edge_index_dict.keys()), [True] * len(list(data.edge_index_dict.keys())))))
        x_dict = {i: j[0] for i,j in out.items()}
        attention_layer = {i: j[1] for i,j in out.items()}
        attention_all_layers.append(attention_layer)
        x_dict = {key: x.relu() for key, x in x_dict.items()}    
    idx2n_id = {}
    for i in batch.node_types:
        idx2n_id[i] = dict(zip(range(len(batch[i].n_id)), batch[i].n_id.numpy()))
        
    node_type = 'SNP'
    edge2weight_l1 = {}
    edge2weight_l2 = {}

    edge_type_node = [i for i,j in batch.edge_index_dict.items() if i[2] == node_type]
    edge_type_node_len = [j.shape[1] for i,j in batch.edge_index_dict.items() if i[2] == node_type]

    for idx, edge_type in enumerate(edge_type_node):
        edge2weight_l1[edge_type] = attention_all_layers[0][node_type][idx]
        assert edge_type_node_len[idx] == edge2weight_l1[edge_type][0].shape[1]

        edge2weight_l2[edge_type] = attention_all_layers[1][node_type][idx]
        assert edge_type_node_len[idx] == edge2weight_l2[edge_type][0].shape[1]

        edge2weight_l1[edge_type][0][0] = torch.LongTensor([idx2n_id[edge_type[0]][ent] for ent in edge2weight_l1[edge_type][0][0].detach().cpu().numpy()])
        edge2weight_l1[edge_type][0][1] = torch.LongTensor([idx2n_id[edge_type[2]][ent] for ent in edge2weight_l1[edge_type][0][1].detach().cpu().numpy()])
        
    return edge2weight_l1, edge2weight_l2
    

def get_fields(all_field_ids, main_data_path):
    headers = pd.read_csv(main_data_path, nrows = 1).columns
    relevant_headers = [i for i, header in enumerate(headers) if header == 'eid' or \
            any([header.startswith('%d-' % field_id) for field_id in all_field_ids])]
    return pd.read_csv(main_data_path, usecols = relevant_headers)


def get_row_last_values(df):
    
    result = pd.Series(np.nan, index = df.index)

    for column in df.columns[::-1]:
        result = result.where(pd.notnull(result), df[column])

    return result

def remove_kinships(eid, verbose = True):

    '''
    Determines which samples need to be removed such that the remaining samples will have no kinship connections whatsoever (according to the
    kinship table provided by the UKBB). In order to determine that, kinship groups will first be determined (@see get_kinship_groups), and 
    only one sample will remain within each of the groups. For the sake of determinism, the sample with the lowest eid will be selected within
    each kinship group, and the rest will be discarded.
    @param eid (pd.Series): A series whose values are UKBB sample IDs, from which kinships should be removed.
    @param verbose (bool): Whether to log details of the operation of this function.
    @return: A mask of samples to keep (pd.Series with index corresponding to the eid input, and boolean values).
    '''
    
    all_eids = set(eid)
    kinship_groups = get_kinship_groups()
    
    relevant_kinship_groups = [kinship_group & all_eids for kinship_group in kinship_groups]
    relevant_kinship_groups = [kinship_group for kinship_group in relevant_kinship_groups if len(kinship_group) >= 2]
    unchosen_kinship_representatives = set.union(*[set(sorted(kinship_group)[1:]) for kinship_group in relevant_kinship_groups])
    no_kinship_mask = ~eid.isin(unchosen_kinship_representatives)
    
    if verbose:
        print_sys(('Constructed %d kinship groups (%d samples), of which %d (%d samples) are relevant for the dataset (i.e. containing at least 2 ' + \
                'samples in the dataset). Picking only one representative of each group and removing the %d other samples in those groups ' + \
                'has reduced the dataset from %d to %d samples.') % (len(kinship_groups), len(set.union(*kinship_groups)), \
                len(relevant_kinship_groups), len(set.union(*relevant_kinship_groups)), len(unchosen_kinship_representatives), len(no_kinship_mask), \
                no_kinship_mask.sum()))
    
    return no_kinship_mask
    
def get_kinship_groups():

    '''
    Uses the kinship table provided by the UKBB (as specified by the KINSHIP_TABLE_FILE_PATH configuration) in order to determine kinship groups.
    Each kinship group is a connected component of samples in the graph of kinships (where each node is a UKBB sample, and an edge exists between
    each pair of samples reported in the kinship table).
    @return: A list of sets of strings (the strings are the sample IDs, i.e. eid). Each set of samples is a kinship group.
    '''
    
    kinship_table = pd.read_csv(kinship_path, sep = ' ')
    kinship_ids = np.array(sorted(set(kinship_table['ID1']) | set(kinship_table['ID2'])))
    n_kinship_ids = len(kinship_ids)
    kinship_id_to_index = pd.Series(np.arange(n_kinship_ids), index = kinship_ids)

    kinship_index1 = kinship_table['ID1'].map(kinship_id_to_index).values
    kinship_index2 = kinship_table['ID2'].map(kinship_id_to_index).values

    symmetric_kinship_index1 = np.concatenate([kinship_index1, kinship_index2])
    symmetric_kinship_index2 = np.concatenate([kinship_index2, kinship_index1])

    kinship_matrix = csr_matrix((np.ones(len(symmetric_kinship_index1), dtype = bool), (symmetric_kinship_index1, \
            symmetric_kinship_index2)), shape = (n_kinship_ids, n_kinship_ids), dtype = bool)

    _, kinship_labels = connected_components(kinship_matrix, directed = False)
    kinship_labels = pd.Series(kinship_labels, index = kinship_ids)
    return [set(group_kinship_labels.index) for _, group_kinship_labels in kinship_labels.groupby(kinship_labels)]
    

def save_dict(path, obj):
    """save an object to a pickle file

    Args:
        path (str): the path to save the pickle file
        obj (object): any file
    """
    with open(path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_dict(path):
    """load an object from a path

    Args:
        path (str): the path where the pickle file locates

    Returns:
        object: loaded pickle file
    """
    with open(path, 'rb') as f:
        return pickle.load(f)
    
def save_model(model, config, path_dir):
    if not os.path.exists(path_dir):
        os.makedirs(path_dir)
    torch.save(model.state_dict(), path_dir + '/model.pt')
    save_dict(path_dir + '/config.pkl', config)

def load_pretrained(path, model):
    state_dict = torch.load(os.path.join(path, 'model.pt'), map_location = torch.device('cpu'))
    # to support training from multi-gpus data-parallel:
    if next(iter(state_dict))[:7] == 'module.':
        # the pretrained model is from data-parallel module
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] # remove `module.`
            new_state_dict[name] = v
        state_dict = new_state_dict

    model.load_state_dict(state_dict)
    return model

def get_args(path):
    return load_dict(os.path.join(path, 'config.pkl'))
    
def print_sys(s):
    """system print

    Args:
        s (str): the string to print
    """
    print(s, flush = True, file = sys.stderr)
    
    
def get_plink_QC_fam():
    fam_path = '/dfs/project/datasets/20220524-ukbiobank/data/genetics/ukb_all.fam'
    data = ukbb_cohort(main_data_path, cohort_data_path, withdraw_path, keep_relatives=True).cohort
    df_fam = pd.read_csv(fam_path, sep = ' ', header = None)
    df_fam[df_fam[0].isin(data)].reset_index(drop = True).to_csv('/dfs/project/datasets/20220524-ukbiobank/data/cohort/qc_cohort.txt', header = None, index = False, sep = ' ')

    
def get_plink_no_rel_fam():
    fam_path = '/dfs/project/datasets/20220524-ukbiobank/data/genetics/ukb_all.fam'
    data = ukbb_cohort(main_data_path, cohort_data_path, withdraw_path, keep_relatives=False).cohort
    df_fam = pd.read_csv(fam_path, sep = ' ', header = None)
    df_fam[df_fam[0].isin(data)].reset_index(drop = True).to_csv('/dfs/project/datasets/20220524-ukbiobank/data/cohort/no_rel.fam', header = None, index = False, sep = ' ')

def get_precision_recall_at_N(res, hits_all, input_dim, N, column_rsid = 'ID', thres = 5e-8):
    eval_dict = {}
    hits_sub = res[res.P < thres][column_rsid].values
    p_sorted = res.sort_values('P')[column_rsid].values
    
    for K in range(1, input_dim, 10000):
        topK_true = np.intersect1d(hits_all, p_sorted[:K])
        recall = len(topK_true)/len(hits_all)
        if recall > N:
            break
    
    for K in range(K-10000, K, 1000):
        topK_true = np.intersect1d(hits_all, p_sorted[:K])
        recall = len(topK_true)/len(hits_all)
        if recall > N:
            break

    for K in range(K-1000, K, 100):
        topK_true = np.intersect1d(hits_all, p_sorted[:K])
        recall = len(topK_true)/len(hits_all)
        if recall > N:
            break

    for K in range(K-100, K, 10):
        topK_true = np.intersect1d(hits_all, p_sorted[:K])
        recall = len(topK_true)/len(hits_all)
        if recall > N:
            break
            
    for K in range(K-10, K):
        topK_true = np.intersect1d(hits_all, p_sorted[:K])
        recall = len(topK_true)/len(hits_all)
        if recall > N:
            break
            
    print_sys('PR@' + str(int(N * 100)) + ' is achieved when K = ' + str(K))
    eval_dict['PR@' + str(int(N * 100)) + '_K'] = K
    topK_true = [1 if i in hits_all else 0 for i in p_sorted[:K]]
    precision = precision_score(topK_true, [1] * K)        
    eval_dict['PR@' + str(int(N * 100))] = precision
    
    return eval_dict

def get_gwas_results(res, hits_all, input_dim, column_rsid = 'ID', thres = 5e-8):
    eval_dict = {}
    hits_sub = res[res.P < thres][column_rsid].values
    eval_dict['overall_recall'] = len(np.intersect1d(hits_sub, hits_all))/len(hits_all)
    if len(hits_sub) == 0:
        eval_dict['overall_precision'] = 0
        eval_dict['overall_f1'] = 0
    else:
        eval_dict['overall_precision'] = len(np.intersect1d(hits_sub, hits_all))/len(hits_sub)
        eval_dict['overall_f1'] = 2 * eval_dict['overall_recall'] * eval_dict['overall_precision']/(eval_dict['overall_recall'] + eval_dict['overall_precision'])
    for K in [100, 500, 1000, 5000]:
        topK_true = [1 if i in hits_all else 0 for i in res.sort_values('P').iloc[:K][column_rsid].values]
        eval_dict['precision_' + str(K)] = precision_score(topK_true, [1] * K)
        eval_dict['recall_' + str(K)] = sum(topK_true)/len(hits_all)
    
    eval_dict.update(get_precision_recall_at_N(res, hits_all, input_dim, 0.8, column_rsid, thres))
    eval_dict.update(get_precision_recall_at_N(res, hits_all, input_dim, 0.9, column_rsid, thres))
    eval_dict.update(get_precision_recall_at_N(res, hits_all, input_dim, 0.95, column_rsid, thres))
    return eval_dict


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def get_preds(logits, multi_label):
    if multi_label:
        preds = (logits.sigmoid() > 0.5).float()
    elif logits.shape[1] > 1:  # multi-class
        preds = logits.argmax(dim=1).float()
    else:  # binary
        preds = (logits.sigmoid() > 0.5).float()
    return preds

def process_data(data, use_edge_attr):
    if not use_edge_attr:
        data.edge_attr = None
    if data.get('edge_label', None) is None:
        data.edge_label = {i: torch.zeros(j.shape[1]) for i, j in data.edge_index_dict.items()}
    return data


def load_checkpoint(model, model_dir, model_name, map_location=None):
    checkpoint = torch.load(model_dir / (model_name + '.pt'), map_location=map_location)
    model.load_state_dict(checkpoint['model_state_dict'])


def save_checkpoint(model, model_dir, model_name):
    torch.save({'model_state_dict': model.state_dict()}, model_dir / (model_name + '.pt'))


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
    
def flatten(list_of_lists):
    return [item for sublist in list_of_lists for item in sublist]


def find_connected_components_details(edges):
    graph = {}
    for u, v in edges:
        if u not in graph:
            graph[u] = []
        if v not in graph:
            graph[v] = []
        graph[u].append(v)
        graph[v].append(u)

    def dfs(vertex):
        visited_nodes = set()
        visited_edges = set()
        stack = [vertex]
        
        while stack:
            current = stack.pop()
            if current not in visited_nodes:
                visited_nodes.add(current)
                for neighbor in graph[current]:
                    stack.append(neighbor)
                    if (current, neighbor) not in visited_edges and (neighbor, current) not in visited_edges:
                        visited_edges.add((current, neighbor))
        return list(visited_nodes), list(visited_edges)

    visited = set()
    components = []

    for vertex in tqdm(graph):
        if vertex not in visited:
            nodes, edges = dfs(vertex)
            components.append({
                'nodes': nodes,
                'edges': edges
            })
            visited.update(nodes)

    return components

def flatten(lst):
    return [item for sublist in lst for item in sublist]



def ldsc_regression_weights(ld, w_ld, N, M, hsq, intercept=None, ii=None):
    '''
    Regression weights.

    Parameters
    ----------
    ld : np.matrix with shape (n_snp, 1)
        LD Scores (non-partitioned).
    w_ld : np.matrix with shape (n_snp, 1)
        LD Scores (non-partitioned) computed with sum r^2 taken over only those SNPs included
        in the regression.
    N :  np.matrix of ints > 0 with shape (n_snp, 1)
        Number of individuals sampled for each SNP.
    M : float > 0
        Number of SNPs used for estimating LD Score (need not equal number of SNPs included in
        the regression).
    hsq : float in [0,1]
        Heritability estimate.

    Returns
    -------
    w : np.matrix with shape (n_snp, 1)
        Regression weights. Approx equal to reciprocal of conditional variance function.

    '''
    M = float(M)
    if intercept is None:
        intercept = 1

    hsq = max(hsq, 0.0)
    hsq = min(hsq, 1.0)
    ld = np.fmax(ld, 1.0)
    w_ld = np.fmax(w_ld, 1.0)
    c = hsq * N / M
    het_w = 1.0 / (2 * np.square(intercept + np.multiply(c, ld)))
    oc_w = 1.0 / w_ld
    w = np.multiply(het_w, oc_w)
    return w