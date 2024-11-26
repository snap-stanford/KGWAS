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
from tqdm import tqdm
from functools import partial

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


def get_network_weight(run, data):
    model = run.best_model
    model = model.to('cpu')
    graph_data = data.data.to('cpu')

    x_dict, edge_index_dict = graph_data.x_dict, graph_data.edge_index_dict
    attention_all_layers = []
    print('Retrieving weights...')

    x_dict['SNP'] = model.snp_feat_mlp(x_dict['SNP'])
    x_dict['Gene'] = model.gene_feat_mlp(x_dict['Gene'])
    x_dict['CellularComponent'] = model.go_feat_mlp(x_dict['CellularComponent'])
    x_dict['BiologicalProcess'] = model.go_feat_mlp(x_dict['BiologicalProcess'])
    x_dict['MolecularFunction'] = model.go_feat_mlp(x_dict['MolecularFunction'])

    for conv in model.convs:
        x_dict = conv(x_dict, edge_index_dict, 
                    return_attention_weights_dict = dict(zip(list(graph_data.edge_index_dict.keys()), 
                                                            [True] * len(list(graph_data.edge_index_dict.keys())))),
                    return_raw_attention_weights_dict = dict(zip(list(graph_data.edge_index_dict.keys()), 
                                                            [True] * len(list(graph_data.edge_index_dict.keys())))),
                    )
        attention_layer = {i: j[1] for i,j in x_dict.items()}
        attention_all_layers.append(attention_layer)
        x_dict = {i: j[0] for i,j in x_dict.items()}

    layer2rel2att = {
        'l1': {},
        'l2': {}
    }

    print('Aggregating across node types...')

    for node_type in graph_data.x_dict.keys():
        edge_type_node = [i for i,j in graph_data.edge_index_dict.items() if i[2] == node_type]
        for idx, i in enumerate(attention_all_layers[0][node_type]):
            layer2rel2att['l1'][edge_type_node[idx]] = np.vstack((i[0].detach().cpu().numpy(), i[1].detach().cpu().numpy().reshape(-1)))
        for idx, i in enumerate(attention_all_layers[1][node_type]):
            layer2rel2att['l2'][edge_type_node[idx]] = np.vstack((i[0].detach().cpu().numpy(), i[1].detach().cpu().numpy().reshape(-1)))
    df_val_all = pd.DataFrame()
    for rel, value in layer2rel2att['l1'].items():
        df_val = pd.DataFrame(value).T.rename(columns = {0: 'h_idx', 1: 't_idx', 2: 'weight'})
        df_val['h_type'] = rel[0] 
        df_val['rel_type'] = rel[1] 
        df_val['t_type'] = rel[2] 
        df_val['layer'] = 'l1'
        df_val_all = df_val_all.append(df_val)

    for rel, value in layer2rel2att['l2'].items():
        df_val = pd.DataFrame(value).T.rename(columns = {0: 'h_idx', 1: 't_idx', 2: 'weight'})
        df_val['h_type'] = rel[0] 
        df_val['rel_type'] = rel[1] 
        df_val['t_type'] = rel[2] 
        df_val['layer'] = 'l2'
        df_val_all = df_val_all.append(df_val)

    df_val_all = df_val_all.drop_duplicates(['h_idx', 't_idx', 'rel_type', 'layer'])
    return df_val_all

def get_local_interpretation(query_snp, v2g, g2g, g2p, g2v, id2idx, K_neighbors):
    try:
        snp2gene_around_snp = v2g[v2g.t_idx == id2idx['SNP'][query_snp]]
        snp2gene_around_snp = snp2gene_around_snp.sort_values('importance')[::-1]
        gene_hit = snp2gene_around_snp.iloc[:K_neighbors]
        gene_hit.loc[:, 'rel_type'] = gene_hit.rel_type.apply(lambda x: x[4:])

        g2g_focal = pd.DataFrame()
        for gene in gene_hit.h_id.values:
            g2g_focal = g2g_focal.append(g2g[g2g.t_id == gene].sort_values('importance')[::-1].iloc[:K_neighbors])
        g2g_focal.loc[:,'rel_type'] = g2g_focal.rel_type.apply(lambda x: x.split('-')[1])

        g2p_focal = pd.DataFrame()
        for gene in gene_hit.h_id.values:
            g2p_focal = g2p_focal.append(g2p[g2p.t_id == gene].sort_values('importance')[::-1].iloc[:K_neighbors])

        g2p_focal.loc[:,'rel_type'] = g2p_focal.rel_type.apply(lambda x: x.split('-')[1])

        g2v_focal = pd.DataFrame()
        for gene in gene_hit.h_id.values:
            g2v_focal = g2v_focal.append(g2v[g2v.t_id == gene].sort_values('importance')[::-1].iloc[:K_neighbors])
        local_neighborhood_around_snp = pd.concat((gene_hit, g2g_focal, g2p_focal, g2v_focal))
        local_neighborhood_around_snp.loc[:,'QUERY_SNP'] = query_snp
        return local_neighborhood_around_snp
    except:
        return None

def generate_viz(run, df_network, data_path, variant_threshold = 5e-8, 
                magma_path = None, magma_threshold = 0.05, program_threshold = 0.05,
                K_neighbors = 3, num_cpus = 1):
    gwas = run.kgwas_res
    idx2id = run.data.idx2id
    id2idx = run.data.id2idx
    print('Start generating disease critical network...')

    gene_sets = load_dict(os.path.join(data_path, 'misc_data/gene_set_bp.pkl'))
    with open(os.path.join(data_path, 'misc_data/go2name.pkl'), 'rb') as f:
        go2name = pickle.load(f)
    
    df_network = df_network[~df_network.rel_type.isin(['TSS', 'rev_TSS'])]

    snp2genes = df_network[(df_network.t_type == 'SNP') 
                       & (df_network.h_type == 'Gene')]
    gene2gene = df_network[(df_network.t_type == 'Gene') 
                           & (df_network.h_type == 'Gene')]
    gene2go = df_network[(df_network.t_type == 'Gene') 
                               & (df_network.h_type.isin(['BiologicalProcess']))]

    if 'SNP' not in gwas.columns.values:
        gwas.loc[:, 'SNP'] = gwas['ID']
    hit_snps = gwas[gwas.P < 5e-8].SNP.values
    hit_snps_idx = [id2idx['SNP'][i] for i in hit_snps]
    
    if magma_path is not None:
        # use magma genes and GSEA programs
        print('Using MAGMA genes to filter...')
        gwas_gene = pd.read_csv(magma_path, sep = '\s+')
        id2gene = dict(pd.read_csv(os.path.join(data_path, 'misc_data/NCBI37.3.gene.loc'), sep = '\t', header = None)[[0,5]].values)
        gwas_gene.loc[:,'GENE'] = gwas_gene['GENE'].apply(lambda x: id2gene[x])

        import statsmodels.api as sm
        p_values = gwas_gene['P']
        corrected_p_values = sm.stats.multipletests(p_values, alpha=magma_threshold, method='bonferroni')[1]
        gwas_gene.loc[:,'corrected_p_value'] = corrected_p_values
        df_gene_hits = gwas_gene[gwas_gene['corrected_p_value'] < magma_threshold]
        rnk = df_gene_hits[['GENE', 'ZSTAT']].set_index('GENE')
        gene_hit_idx = [id2idx['Gene'][i] for i in df_gene_hits.GENE.values if i in id2idx['Gene']]

        try:
            gsea_results_BP = gp.prerank(rnk=rnk, gene_sets=gene_sets, 
                                        outdir=None, permutation_num=100, 
                                        min_size=2, max_size=1000, seed = 42)
            gsea_results_BP = gsea_results_BP.res2d
            go_hits = gsea_results_BP[gsea_results_BP['NOM p-val'] < program_threshold].Term.values
            if len(go_hits) <= 5:
                go_hits = gsea_results_BP.sort_values('NOM p-val')[:5].Term.values
            go_hits_idx = [id2idx['BiologicalProcess'][x] for x in go_hits]
            print('Using GSEA gene programs to filter...')
        except:
            print('No significant gene programs found...')
            go_hits_idx = []
    else:
        # use all genes and gene programs
        print('No filters... Using all genes and gene programs...')
        gene_hit_idx = list(id2idx['Gene'].values())
        go_hits_idx = list(id2idx['BiologicalProcess'].values())
    

    snp2genes_hit = snp2genes[snp2genes.t_idx.isin(hit_snps_idx) & snp2genes.h_idx.isin(gene_hit_idx)]
    rel2mean = snp2genes_hit.groupby('rel_type').weight.mean().reset_index().rename(columns = {'weight': 'rel_type_mean'})
    rel2std = snp2genes_hit.groupby('rel_type').weight.agg(np.std).reset_index().rename(columns = {'weight': 'rel_type_std'})

    snp2genes_hit = snp2genes_hit.merge(rel2std)
    snp2genes_hit = snp2genes_hit.merge(rel2mean)
    snp2genes_hit.loc[:,'z_rel'] = (snp2genes_hit['weight'] - snp2genes_hit['rel_type_mean'])/snp2genes_hit['rel_type_std']
    
    v2g_hit = snp2genes_hit.groupby(['h_idx', 't_idx']).z_rel.max().reset_index().rename(columns={'z_rel': 'importance'})
    v2g_hit_with_rel_type = pd.merge(v2g_hit, snp2genes_hit, left_on=['h_idx', 't_idx', 'importance'], right_on=['h_idx', 't_idx', 'z_rel'], how='left')
    v2g_hit = v2g_hit_with_rel_type[['h_idx', 't_idx', 'importance', 'h_type', 't_type', 'rel_type']]
    v2g_hit.loc[:,'rel_type'] = v2g_hit.rel_type.apply(lambda x: x[4:])
    v2g_hit.loc[:,'Category'] = 'V2G'

    v2g_hit.loc[:,'h_id'] = v2g_hit['h_idx'].apply(lambda x: idx2id['Gene'][x])
    v2g_hit.loc[:,'t_id'] = v2g_hit['t_idx'].apply(lambda x: idx2id['SNP'][x])

    gene2gene_hit = gene2gene[gene2gene.h_idx.isin(gene_hit_idx) & gene2gene.t_idx.isin(gene_hit_idx)]
    rel2mean = gene2gene_hit.groupby('rel_type').weight.mean().reset_index().rename(columns = {'weight': 'rel_type_mean'})
    rel2std = gene2gene_hit.groupby('rel_type').weight.agg(np.std).reset_index().rename(columns = {'weight': 'rel_type_std'})

    gene2gene_hit = gene2gene_hit.merge(rel2std)
    gene2gene_hit = gene2gene_hit.merge(rel2mean)
    gene2gene_hit.loc[:,'z_rel'] = (gene2gene_hit['weight'] - gene2gene_hit['rel_type_mean'])/gene2gene_hit['rel_type_std']

    g2g_hit = gene2gene_hit.groupby(['h_idx', 't_idx']).z_rel.max().reset_index().rename(columns={'z_rel': 'importance'})
    g2g_hit_with_rel_type = pd.merge(g2g_hit, gene2gene_hit, left_on=['h_idx', 't_idx', 'importance'], right_on=['h_idx', 't_idx', 'z_rel'], how='left')
    g2g_hit = g2g_hit_with_rel_type[['h_idx', 't_idx', 'importance', 'h_type', 't_type', 'rel_type']]
    g2g_hit.loc[:,'rel_type'] = g2g_hit.rel_type.apply(lambda x: x.split('-')[1])
    g2g_hit.loc[:,'Category'] = 'G2G'

    g2g_hit.loc[:,'h_id'] = g2g_hit['h_idx'].apply(lambda x: idx2id['Gene'][x])
    g2g_hit.loc[:,'t_id'] = g2g_hit['t_idx'].apply(lambda x: idx2id['Gene'][x])

    gene2program_hit = gene2go[gene2go.t_idx.isin(gene_hit_idx) & gene2go.h_idx.isin(go_hits_idx)]
    rel2mean = gene2program_hit.groupby('rel_type').weight.mean().reset_index().rename(columns = {'weight': 'rel_type_mean'})
    rel2std = gene2program_hit.groupby('rel_type').weight.agg(np.std).reset_index().rename(columns = {'weight': 'rel_type_std'})

    gene2program_hit = gene2program_hit.merge(rel2std)
    gene2program_hit = gene2program_hit.merge(rel2mean)
    gene2program_hit.loc[:,'z_rel'] = (gene2program_hit['weight'] - gene2program_hit['rel_type_mean'])/gene2program_hit['rel_type_std']

    g2p_hit = gene2program_hit.groupby(['h_idx', 't_idx']).z_rel.max().reset_index().rename(columns={'z_rel': 'importance'})

    g2p_hit_with_rel_type = pd.merge(g2p_hit, gene2program_hit, left_on=['h_idx', 't_idx', 'importance'], right_on=['h_idx', 't_idx', 'z_rel'], how='left')
    g2p_hit = g2p_hit_with_rel_type[['h_idx', 't_idx', 'importance', 'h_type', 't_type', 'rel_type']]
    g2p_hit.loc[:,'rel_type'] = g2p_hit.rel_type.apply(lambda x: x.split('-')[1])
    g2p_hit.loc[:,'Category'] = 'G2P'
    g2p_hit.loc[:,'h_id'] = g2p_hit['h_idx'].apply(lambda x: idx2id['BiologicalProcess'][x])
    g2p_hit.loc[:,'t_id'] = g2p_hit['t_idx'].apply(lambda x: idx2id['Gene'][x])
    g2p_hit.loc[:,'h_id'] = g2p_hit.h_id.apply(lambda x: go2name[x].capitalize() if x in go2name else x)
    disease_critical_network = pd.concat((v2g_hit, g2g_hit, g2p_hit)).reset_index(drop = True)

    print('Disease critical network finished generating...')
    print('Generating variant interpretation networks...')

    #### get for variant interpretation -> since we are looking at top K neighbors, we don't filter
    
    # V2G
    rel2mean = snp2genes_hit.groupby('rel_type').weight.mean().reset_index().rename(columns = {'weight': 'rel_type_mean'})
    rel2std = snp2genes_hit.groupby('rel_type').weight.agg(np.std).reset_index().rename(columns = {'weight': 'rel_type_std'})

    snp2genes = snp2genes.merge(rel2std)
    snp2genes = snp2genes.merge(rel2mean)
    snp2genes.loc[:,'z_rel'] = (snp2genes['weight'] - snp2genes['rel_type_mean'])/snp2genes['rel_type_std']
    snp2genes = snp2genes.rename(columns={'z_rel': 'importance'})
    v2g = snp2genes.groupby(['h_idx', 't_idx']).importance.max().reset_index()
    v2g_with_rel_type = pd.merge(v2g, snp2genes, left_on=['h_idx', 't_idx', 'importance'], right_on=['h_idx', 't_idx', 'importance'], how='left')
    v2g = v2g_with_rel_type[['h_idx', 't_idx', 'importance', 'h_type', 't_type', 'rel_type']]

    v2g.loc[:,'h_id'] = v2g['h_idx'].apply(lambda x: idx2id['Gene'][x])
    v2g.loc[:,'t_id'] = v2g['t_idx'].apply(lambda x: idx2id['SNP'][x])

    ## G2G

    rel2mean = gene2gene_hit.groupby('rel_type').weight.mean().reset_index().rename(columns = {'weight': 'rel_type_mean'})
    rel2std = gene2gene_hit.groupby('rel_type').weight.agg(np.std).reset_index().rename(columns = {'weight': 'rel_type_std'})

    gene2gene = gene2gene.merge(rel2std)
    gene2gene = gene2gene.merge(rel2mean)
    gene2gene.loc[:,'z_rel'] = (gene2gene['weight'] - gene2gene['rel_type_mean'])/gene2gene['rel_type_std']
    gene2gene = gene2gene.rename(columns={'z_rel': 'importance'})

    g2g = gene2gene.groupby(['h_idx', 't_idx']).importance.max().reset_index()
    g2g_with_rel_type = pd.merge(g2g, gene2gene, left_on=['h_idx', 't_idx', 'importance'], right_on=['h_idx', 't_idx', 'importance'], how='left')
    g2g = g2g_with_rel_type[['h_idx', 't_idx', 'importance', 'h_type', 't_type', 'rel_type']]

    g2g.loc[:,'h_id'] = g2g['h_idx'].apply(lambda x: idx2id['Gene'][x])
    g2g.loc[:,'t_id'] = g2g['t_idx'].apply(lambda x: idx2id['Gene'][x])
    g2g = g2g[g2g.h_idx != g2g.t_idx]

    ## G2P

    rel2mean = gene2program_hit.groupby('rel_type').weight.mean().reset_index().rename(columns = {'weight': 'rel_type_mean'})
    rel2std = gene2program_hit.groupby('rel_type').weight.agg(np.std).reset_index().rename(columns = {'weight': 'rel_type_std'})

    gene2go = gene2go.merge(rel2std)
    gene2go = gene2go.merge(rel2mean)
    gene2go.loc[:,'z_rel'] = (gene2go['weight'] - gene2go['rel_type_mean'])/gene2go['rel_type_std']
    gene2go = gene2go.rename(columns={'z_rel': 'importance'})

    g2p = gene2go.groupby(['h_idx', 't_idx']).importance.max().reset_index()
    g2p_with_rel_type = pd.merge(g2p, gene2go, left_on=['h_idx', 't_idx', 'importance'], right_on=['h_idx', 't_idx', 'importance'], how='left')
    g2p = g2p_with_rel_type[['h_idx', 't_idx', 'importance', 'h_type', 't_type', 'rel_type']]

    g2p.loc[:,'h_id'] = g2p['h_idx'].apply(lambda x: go2name[idx2id['BiologicalProcess'][x]].capitalize() if idx2id['BiologicalProcess'][x] in go2name else idx2id['BiologicalProcess'][x])
    g2p.loc[:,'t_id'] = g2p['t_idx'].apply(lambda x: idx2id['Gene'][x])


    ## G2V

    gene2snp = df_network[(df_network.h_type == 'SNP') 
                       & (df_network.t_type == 'Gene')]

    gene2snp_hit = gene2snp[gene2snp.h_idx.isin(hit_snps_idx) & gene2snp.t_idx.isin(gene_hit_idx)]

    rel2mean = gene2snp_hit.groupby('rel_type').weight.mean().reset_index().rename(columns = {'weight': 'rel_type_mean'})
    rel2std = gene2snp_hit.groupby('rel_type').weight.agg(np.std).reset_index().rename(columns = {'weight': 'rel_type_std'})

    gene2snp = gene2snp.merge(rel2std)
    gene2snp = gene2snp.merge(rel2mean)
    gene2snp.loc[:,'z_rel'] = (gene2snp['weight'] - gene2snp['rel_type_mean'])/gene2snp['rel_type_std']
    gene2snp = gene2snp.rename(columns={'z_rel': 'importance'})

    g2v = gene2snp.groupby(['h_idx', 't_idx']).importance.max().reset_index()
    g2v_with_rel_type = pd.merge(g2v, gene2snp, left_on=['h_idx', 't_idx', 'importance'], right_on=['h_idx', 't_idx', 'importance'], how='left')
    g2v = g2v_with_rel_type[['h_idx', 't_idx', 'importance', 'h_type', 't_type', 'rel_type']]

    g2v.loc[:,'h_id'] = g2v['h_idx'].apply(lambda x: idx2id['SNP'][x])
    g2v.loc[:,'t_id'] = g2v['t_idx'].apply(lambda x: idx2id['Gene'][x])
    
    print('Number of hit snps: ', len(hit_snps))
    process_func = partial(get_local_interpretation, v2g=v2g, g2g=g2g, g2p=g2p, g2v=g2v, id2idx=id2idx, K_neighbors=K_neighbors)

    with Pool(num_cpus) as p:
        res = list(tqdm(p.imap(process_func, hit_snps), total=len(hit_snps)))
    try:
        df_variant_interpretation = pd.concat([i for i in res if i is not None])
    except:
        df_variant_interpretation = pd.DataFrame()

    return df_variant_interpretation, disease_critical_network