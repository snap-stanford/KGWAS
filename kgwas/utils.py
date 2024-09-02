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
import matplotlib.pyplot as plt
from tdc import Evaluator
from multiprocessing import Pool


from .params import main_data_path, cohort_data_path, kinship_path, withdraw_path

def evaluate_minibatch_multi(loader, model, args, binary_output):
    model.eval()
    pred_all = []
    truth = []
    results = {}
    for step, batch in enumerate(tqdm(loader)):        
        batch = batch.to(args.device)    
        out = model(batch)
        if not binary_output:
            pred = out.reshape(-1)
        else:
            pred = torch.sigmoid(out.reshape(-1))
        y_batch = batch[('SNP', 'SNP-Pheno', 'ICD10')]['edge_label'].float()
        pred_all.extend(pred.detach().cpu().numpy())
        truth.extend(y_batch.detach().cpu().numpy())
        del batch, out, pred, y_batch
    results['pred'] = np.hstack(pred_all)
    results['truth'] = np.hstack(truth)
    return results

def evaluate_minibatch(loader, model, args, idx2id, lmdb_connection, pheno_idx, pheno_y, get_lmdb, p, binary_output):
    #p = Pool(16)
    #def get_lmdb(x):
    #    with lmdb_connection.begin(write=False) as txn:
    #        genotype = np.array(np.frombuffer(txn.get(x.encode()), dtype = 'int8'))[pheno_idx]
    #    return genotype
    
    model.eval()
    pred_all = []
    truth = []
    results = {}
    for step, batch in enumerate(tqdm(loader)):        
        batch = batch.to(args.device, 'edge_index')
        bs_batch = batch['SNP'].batch_size
        
        if args.add_genotype:
            rs_id = [idx2id['SNP'][i.item()] for i in batch['SNP']['n_id'][:bs_batch]]
            #with lmdb_connection.begin(write=False) as txn:
            #    genotype = np.vstack([np.array(np.frombuffer(txn.get(idx.encode()), dtype = 'int8'))[pheno_idx] for idx in rs_id])
            genotype = np.vstack(p.map(get_lmdb, rs_id))
            if pheno_y is not None:
                genotype = np.hstack((genotype, np.tile(pheno_y, (bs_batch, 1))))
            genotype = torch.tensor(genotype).float().to(args.device)
        else:
            genotype = 0
        
        out = model(batch.x_dict, batch.edge_index_dict, bs_batch, genotype)
        
        if not binary_output:
            if out.shape[1] > 1:
                ## multi-task mode, takes the last value
                pred = out[:,-1].reshape(-1)
            else:
                pred = out.reshape(-1)
        else:
            pred = torch.sigmoid(out.reshape(-1))
            
        y_batch = batch['SNP'].y[:bs_batch]
        
        pred_all.extend(pred.detach().cpu().numpy())
        truth.extend(y_batch.detach().cpu().numpy())
        del y_batch, pred, batch, out
        
    results['pred'] = np.hstack(pred_all)
    results['truth'] = np.hstack(truth)
    return results


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




def evaluate(loader, model, device, uncertainty, uncertainty_metric, model_name = None, binary_softmax = False, snp_pred = False):
    model.eval()
    pred = []
    truth = []
    unc = []
    low = []
    upp = []
    f_all = []
    g_all = []
    results = {}
    for step, (X, y) in enumerate(tqdm(loader)):
        if snp_pred:
            X = X.to(device)
        else:
            X = X.float().to(device)
        y = y.float().to(device)
        if uncertainty:
            if uncertainty_metric == 'normal':
                y_hat, logvar = model(X)
                unc.extend(logvar.detach().cpu().numpy())
            elif uncertainty_metric == 'quantile':
                y_hat, l, u = model(X)
                low.extend(l.detach().cpu().numpy())
                upp.extend(u.detach().cpu().numpy())
            elif uncertainty_metric == 'abstention':
                y_hat, f, g = model(X)
                if binary_softmax:
                    f = f.softmax(dim = 1)[:, 1]
                f_all.extend(f.detach().cpu().numpy())
                g_all.extend(g.detach().cpu().numpy())
        else:
            if model_name == 'DeepNormal':
                y_hat = model(X).mean
            else:
                y_hat = model(X)
        if binary_softmax:
            y_hat = y_hat.softmax(dim = 1)[:, 1]
                
        pred.extend(y_hat.detach().cpu().numpy())
        truth.extend(y.detach().cpu().numpy())
        
    results['pred'] = np.hstack(pred)
    results['truth'] = np.hstack(truth)
    if uncertainty:
        if uncertainty_metric == 'normal':
            results['logvar'] = np.hstack(unc)
        elif uncertainty_metric == 'quantile':
            results['lower'] = np.hstack(low)
            results['upper'] = np.hstack(upp)
        elif uncertainty_metric == 'abstention':
            results['f'] = np.hstack(f_all)
            results['g'] = np.hstack(g_all)
    return results

def compute_metrics(results, binary, coverage = None, uncertainty_reg = 1, loss_fct = None):
    metrics = {}
    if 'logvar' in results:
        try:
            metrics['unc_correlation'] = pearsonr(np.exp(-results['logvar']), (results['truth'] - results['pred'])**2)[0]
        except:
            print_sys('Uncertainty correlation goes to infinity...')
            metrics['unc_correlation'] = 0
    
    if 'g' in results:
        pred = torch.tensor(results['pred'])
        y =  torch.tensor(results['truth'])
        g = torch.tensor(results['g'])
        f = torch.tensor(results['f'])
        f_loss, select_loss, pred_loss = loss_fct(pred, f, g, y, reg = uncertainty_reg, coverage = coverage)
        g_mean = np.mean(g.numpy())
        metrics['g_mean'] = float(g_mean)
        metrics['g_open_num'] = len(np.where(results['g'] > 0.5)[0])
        metrics['abstention_loss'] = (uncertainty_reg * f_loss + select_loss) + pred_loss
        
        evaluator = Evaluator(name = 'PR-AUC')
        
        if metrics['g_open_num'] != 0:
            metrics['prauc_abs_gate'] = evaluator(results['truth'][np.where(results['g'] > 0.5)[0]], results['pred'][np.where(results['g'] > 0.5)[0]])
        else:
            metrics['prauc_abs_gate'] = 0
            
        evaluator_acc = Evaluator(name = 'Accuracy')
        for q in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
            q_val = np.quantile(results['g'], q = q)
            masked_pos = np.where(results['g'] > q_val)[0] 
            if len(masked_pos) != 0:
                metrics['prauc_abs_q' + str(q)] =  evaluator(results['truth'][masked_pos], results['pred'][masked_pos])
                metrics['acc_abs_q' + str(q)] =  evaluator_acc(results['truth'][masked_pos], results['pred'][masked_pos], threshold = 0.5)
            else:
                metrics['prauc_abs_q' + str(q)] =  2
                metrics['acc_abs_q' + str(q)] =  2
            
    if binary:
        metrics['bce'] = F.binary_cross_entropy(torch.tensor(results['truth']), torch.tensor(results['pred'])).item()
        
        evaluator = Evaluator(name = 'ROC-AUC')
        try:
            metrics['auroc'] =  evaluator(results['truth'], results['pred'])
        except:
            metrics['auroc'] =  0          
        
        evaluator = Evaluator(name = 'PR-AUC')
        metrics['prauc'] =  evaluator(results['truth'], results['pred'])
        evaluator = Evaluator(name = 'PR@K')
        metrics['pr@90'] =  evaluator(results['truth'], results['pred'], threshold = 0.9)
        metrics['pr@95'] =  evaluator(results['truth'], results['pred'], threshold = 0.95)
        metrics['pr@80'] =  evaluator(results['truth'], results['pred'], threshold = 0.8)
        evaluator = Evaluator(name = 'RP@K')
        metrics['rp@90'] =  evaluator(results['truth'], results['pred'], threshold = 0.9)
        metrics['rp@95'] =  evaluator(results['truth'], results['pred'], threshold = 0.95)
        metrics['rp@80'] =  evaluator(results['truth'], results['pred'], threshold = 0.8)
        
        evaluator = Evaluator(name = 'PR-AUC')
        evaluator_acc = Evaluator(name = 'Accuracy')
        for q in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
            q_val = np.quantile(results['pred'], q = q)
            masked_pos = np.where(results['pred'] > q_val)[0] 
            if len(masked_pos) != 0:
                metrics['prauc_q' + str(q)] =  evaluator(results['truth'][masked_pos], results['pred'][masked_pos])
                metrics['acc_q' + str(q)] =  evaluator_acc(results['truth'][masked_pos], results['pred'][masked_pos], threshold = 0.5)
            else:
                metrics['prauc_q' + str(q)] =  2
                metrics['acc_q' + str(q)] =  2
        
    else:
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



def visualize_local_subgraph(query_snp, id2idx):
    q = id2idx['SNP'][query_snp]
    eval_input_nodes = ('SNP', np.array([q]))
    eval_loader = NeighborLoader(data, num_neighbors=[args.num_neighbor_samples] * args.gnn_num_layers,
                                    input_nodes=eval_input_nodes, **eval_kwargs)
    batch = next(iter(eval_loader))
    model, x_dict, edge_index_dict = best_model, batch.x_dict, batch.edge_index_dict
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
        
        
    edge_weight_node_type = {}

    for node_type in batch.node_types:
        edge2weight_l1 = {}
        edge2weight_l2 = {}

        edge_type_node = [i for i,j in batch.edge_index_dict.items() if i[2] == node_type]
        edge_type_node_len = [j.shape[1] for i,j in batch.edge_index_dict.items() if i[2] == node_type]

        for idx, edge_type in enumerate(edge_type_node):
            edge2weight_l1[edge_type] = attention_all_layers[0][node_type][idx]
            assert edge_type_node_len[idx] == edge2weight_l1[edge_type][0].shape[1]

            edge2weight_l2[edge_type] = attention_all_layers[1][node_type][idx]
            assert edge_type_node_len[idx] == edge2weight_l2[edge_type][0].shape[1]

            ## inplace, so edge2weight_l2 does not need to change
            edge2weight_l1[edge_type][0][0] = torch.LongTensor([idx2n_id[edge_type[0]][ent] for ent in edge2weight_l1[edge_type][0][0].detach().cpu().numpy()])
            edge2weight_l1[edge_type][0][1] = torch.LongTensor([idx2n_id[edge_type[2]][ent] for ent in edge2weight_l1[edge_type][0][1].detach().cpu().numpy()])
            #edge2weight_l2[edge_type][0][0] = torch.LongTensor([idx2n_id[edge_type[0]][ent] for ent in edge2weight_l2[edge_type][0][0].detach().cpu().numpy()])
            #edge2weight_l2[edge_type][0][1] = torch.LongTensor([idx2n_id[edge_type[2]][ent] for ent in edge2weight_l2[edge_type][0][1].detach().cpu().numpy()])

        edge_weight_node_type[node_type] = {'l1': edge2weight_l1, 'l2': edge2weight_l2}
    
    l2_all = pd.DataFrame()

    for rel, val_pair in edge_weight_node_type['SNP']['l2'].items():
        head_type = rel[0]
        tail_type = rel[2]

        head_id = [idx2id[head_type][i] for i in val_pair[0][0].detach().cpu().numpy()]
        tail_id = [idx2id[tail_type][i] for i in val_pair[0][1].detach().cpu().numpy()]
        weight = val_pair[1].detach().cpu().numpy().reshape(-1)

        l2_rel = pd.DataFrame((head_id, weight, tail_id, [rel[1]] * len(head_id), [rel[0]]* len(head_id), [rel[2]] * len(head_id), ['l2'] * len(head_id))).T
        l2_rel = l2_rel.drop_duplicates()

        l2_all = l2_all.append(l2_rel)

    l1_all = pd.DataFrame()

    for rel, val_pair in edge_weight_node_type['Gene']['l1'].items():
        head_type = rel[0]
        tail_type = rel[2]

        head_id = [idx2id[head_type][i] for i in val_pair[0][0].detach().cpu().numpy()]
        tail_id = [idx2id[tail_type][i] for i in val_pair[0][1].detach().cpu().numpy()]
        weight = val_pair[1].detach().cpu().numpy().reshape(-1)

        l1_rel = pd.DataFrame((head_id, weight, tail_id, [rel[1]] * len(head_id), [rel[0]]* len(head_id), [rel[2]] * len(head_id), ['l1'] * len(head_id))).T
        l1_rel = l1_rel.drop_duplicates()

        l1_all = l1_all.append(l1_rel)
        
    ### remove the rev nodes for the second hop 
    l1_all = l1_all[l1_all[0] != query_snp]
    ### remove genes that connect to themselves
    l1_all = l1_all[l1_all[0] != l1_all[2]]
    ### pydot does not support : in the node attribute
    l1_all[0] = l1_all[0].apply(lambda x: x.split(':')[0] + '-' + x.split(':')[1] if ':' in x else x)

    l_all = pd.concat([l1_all, l2_all])
    
    for rel in l2_all[l2_all[2] == query_snp][3].unique():
        G = nx.DiGraph()
        filter_df = l2_all[(l2_all[2] == query_snp) & (l2_all[3] == rel)]
        G.add_edges_from(filter_df[[0, 2]].values)
        ax = plt.gca()
        ax.set_title(rel + ' - ' + query_snp)

        options = {
            "node_color": "#A0CBE2",
            "edge_color": filter_df[1].values,
            "width": 1,
            "edge_cmap": plt.cm.Blues,
            "with_labels": False,
            "node_size": 50
        }
        pos = graphviz_layout(G, prog="dot")
        nx.draw(G, pos, **options)
        text = nx.draw_networkx_labels(G,pos)

        for _,t in text.items():
            t.set_rotation(45)

        plt.show()
        
    query_gene = 'GOLGA7'
    for rel in l1_all[l1_all[2] == query_gene][3].unique():
        print(rel)
        G = nx.DiGraph()
        filter_df = l1_all[(l1_all[2] == query_gene) & (l1_all[3] == rel)]
        G.add_edges_from(filter_df[[0, 2]].values)
        ax = plt.gca()
        ax.set_title(rel + ' - ' + query_gene)

        options = {
            "node_color": "#A0CBE2",
            "edge_color": filter_df[1].values,
            "width": 1,
            "edge_cmap": plt.cm.Blues,
            "with_labels": False,
            "node_size": 50
        }
        pos = graphviz_layout(G, prog="dot")
        nx.draw(G, pos, **options)
        text = nx.draw_networkx_labels(G,pos)

        for _,t in text.items():
            t.set_rotation(45)

        plt.show()
    
    G = nx.DiGraph()
    G.add_edges_from(l2_all[[0, 2]].values)
    G.add_edges_from(l1_all[[0, 2]].values)
    with_label = [True if i in l2_all[2].unique().tolist() + l2_all[0].unique().tolist() else False for i in G.nodes()]
    labels = {}    
    for node in G.nodes():
        if node in l2_all[2].unique().tolist() + l2_all[0].unique().tolist():
            #set the node name as the key and the label as its value 
            labels[node] = node

    figure(figsize=(20, 6), dpi=80)

    options = {
        "node_color": "#A0CBE2",
        "edge_color": [np.mean(l_all[(l_all[0] == i[0]) & (l_all[2] == i[1])][1].values) for i in G.edges()],
        "width": 1,
        "edge_cmap": plt.cm.Blues,
        "with_labels": False,
        "node_size": 3
    }
    pos = graphviz_layout(G, prog="dot")
    nx.draw(G, pos, **options)
    nx.draw_networkx_labels(G,pos,labels,font_size=16)
    nx.draw_networkx_nodes(G, pos, labels, node_color='#A0CBE2')
    
    
    
    
def abstention_loss_fct(pred, f, g, y, reg = 1, coverage = 1):
    f_loss = torch.mean((pred - y)**2 * g)
    pred_loss = torch.mean((pred - y)**2)
    g_mean = torch.mean(g)
    select_loss = torch.max(torch.zeros_like(g_mean).to(device), coverage - g_mean)
    #select_loss = g_mean
    #return 0.5*(f_loss + reg * select_loss) + 0.5*pred_loss
    return f_loss, select_loss, pred_loss


def abstention_loss_fct_binary(pred, f, g, y, reg = 1, coverage = 1):
    f_loss = torch.mean(F.binary_cross_entropy(f, y, reduction= 'none') * g)
    pred_loss = F.binary_cross_entropy(pred, y)
    g_mean = torch.mean(g)
    select_loss = torch.pow(F.relu(coverage - g_mean), 2)
    #select_loss = g_mean
    #return 0.5*(f_loss + reg * select_loss) + 0.5*pred_loss
    return f_loss, select_loss, pred_loss
    
    
def abstention_loss_fct_binary_softmax(pred, f, g, y, reg = 1, coverage = 1):
    ce_1 = nn.CrossEntropyLoss(reduction= 'none')
    ce = nn.CrossEntropyLoss()
    f_loss = torch.mean(ce_1(pred, y) * g)
    pred_loss = ce(pred, y)
    g_mean = torch.mean(g)
    select_loss = torch.pow(F.relu(coverage - g_mean), 2)
    #select_loss = g_mean
    #return 0.5*(f_loss + reg * select_loss) + 0.5*pred_loss
    return f_loss, select_loss, pred_loss


def normal_loss_fct(pred, logvar, y, epoch, reg = 1, print_out = False):
    pred_loss = torch.sum((pred - y)**2)
    unc_loss = reg * torch.sum(0.5 * torch.exp(-logvar) * (pred - y)**2 + 0.5 * logvar)
    #if print_out:
    #    print(str(pred_loss.item()/pred.shape[0]) + '_' + str(unc_loss.item()/pred.shape[0]))
    #loss_all = (pred_loss + unc_loss)/pred.shape[0]        
    loss_all = unc_loss/pred.shape[0]        
    return loss_all
    
def normal_loss_fct_binary(pred, logvar, y, epoch, reg = 1, print_out = False):
    pred_loss = F.binary_cross_entropy(pred, y)
    unc_loss = reg * torch.sum(0.5 * torch.exp(-logvar) * pred_loss + 0.5 * logvar)
    #if print_out:
    #    print(str(pred_loss.item()/pred.shape[0]) + '_' + str(unc_loss.item()/pred.shape[0]))
    #loss_all = (pred_loss + unc_loss)/pred.shape[0]        
    loss_all = unc_loss/pred.shape[0]        
    return loss_all
    
def quantile_loss_fct(pred, lower, upper, y, alpha):
    low_bound = alpha
    upp_bound = 1 - alpha
    median_loss = torch.mean(torch.max((0.5 - 1) * (y - pred), 0.5 * (y - pred)))
    low_loss = torch.mean(torch.max((low_bound - 1) * (y - lower), low_bound * (y - lower)))
    upp_loss = torch.mean(torch.max((upp_bound - 1) * (y - upper), upp_bound * (y - upper)))
    return median_loss, low_loss, upp_loss

def error_pred_loss_fct(pred, logvar, y, epoch, reg = 1):
    pred_loss = torch.sum((pred - y)**2)
    if epoch > 5:
        unc_loss = reg * torch.sum((logvar - (pred - y)**2)**2)
        print(str(pred_loss.item()/pred.shape[0]) + '_' + str(unc_loss.item()/pred.shape[0]))        
        loss_all = (pred_loss + unc_loss)/pred.shape[0]        
    else:
        loss_all = pred_loss/pred.shape[0]
    return loss_all

def deepnormal_loss(normal_dist, y):
    neg_log_likelihood = -normal_dist.log_prob(y)
    return torch.mean(neg_log_likelihood)


def nig_nll(y, gamma, v, alpha, beta):
    two_blambda = 2 * beta * (1 + v)
    nll = 0.5 * torch.log(np.pi / v) \
            - alpha * torch.log(two_blambda) \
            + (alpha + 0.5) * torch.log(v * (y - gamma) ** 2 + two_blambda) \
            + torch.lgamma(alpha) \
            - torch.lgamma(alpha + 0.5)
        
    return nll


def nig_reg(y, gamma, v, alpha, beta):
    error = F.l1_loss(y, gamma, reduction="none")
    evi = 2 * v + alpha
    return error * evi


def evidential_regresssion_loss(y, pred, coeff=1.0, weight_loss = None):
    gamma, v, alpha, beta = pred
    loss_nll = nig_nll(y, gamma, v, alpha, beta)
    loss_reg = nig_reg(y, gamma, v, alpha, beta)
    
    if weight_loss is None:
        loss_ = loss_nll.mean() + coeff * (loss_reg.mean() - 1e-4)
        return loss_
    else:
        loss_ = torch.sum(weight_loss * loss_nll) + coeff * (torch.sum(loss_reg * weight_loss) - 1e-4)
        
        return loss_
    
def get_pred_evidential_aleatoric(out):
    gamma, v, alpha, beta = out
    var = beta / (alpha - 1)
    return gamma, var

def get_pred_evidential_epistemic(out):
    gamma, v, alpha, beta = out
    var = beta / (v * (alpha - 1))
    return gamma, var



def compute_calibration(true_labels, pred_labels, confidences, num_bins=10):
    """Collects predictions into bins used to draw a reliability diagram.
    Arguments:
        true_labels: the true labels for the test examples
        pred_labels: the predicted labels for the test examples
        confidences: the predicted confidences for the test examples
        num_bins: number of bins
    The true_labels, pred_labels, confidences arguments must be NumPy arrays;
    pred_labels and true_labels may contain numeric or string labels.
    For a multi-class model, the predicted label and confidence should be those
    of the highest scoring class.
    Returns a dictionary containing the following NumPy arrays:
        accuracies: the average accuracy for each bin
        confidences: the average confidence for each bin
        counts: the number of examples in each bin
        bins: the confidence thresholds for each bin
        avg_accuracy: the accuracy over the entire test set
        avg_confidence: the average confidence over the entire test set
        expected_calibration_error: a weighted average of all calibration gaps
        max_calibration_error: the largest calibration gap across all bins
    """
    assert(len(confidences) == len(pred_labels))
    assert(len(confidences) == len(true_labels))
    assert(num_bins > 0)

    bin_size = 1.0 / num_bins
    bins = np.linspace(0.0, 1.0, num_bins + 1)
    indices = np.digitize(confidences, bins, right=True)

    bin_accuracies = np.zeros(num_bins, dtype=np.float)
    bin_confidences = np.zeros(num_bins, dtype=np.float)
    bin_counts = np.zeros(num_bins, dtype=np.int)

    for b in range(num_bins):
        selected = np.where(indices == b + 1)[0]
        if len(selected) > 0:
            bin_accuracies[b] = np.mean(true_labels[selected] == pred_labels[selected])
            bin_confidences[b] = np.mean(confidences[selected])
            bin_counts[b] = len(selected)

    avg_acc = np.sum(bin_accuracies * bin_counts) / np.sum(bin_counts)
    avg_conf = np.sum(bin_confidences * bin_counts) / np.sum(bin_counts)

    gaps = np.abs(bin_accuracies - bin_confidences)
    ece = np.sum(gaps * bin_counts) / np.sum(bin_counts)
    mce = np.max(gaps)

    return { "accuracies": bin_accuracies, 
             "confidences": bin_confidences, 
             "counts": bin_counts, 
             "bins": bins,
             "avg_accuracy": avg_acc,
             "avg_confidence": avg_conf,
             "expected_calibration_error": ece,
             "max_calibration_error": mce }


def _reliability_diagram_subplot(ax, bin_data, 
                                 draw_ece=True, 
                                 draw_bin_importance=False,
                                 title="Reliability Diagram", 
                                 xlabel="Confidence", 
                                 ylabel="Expected Accuracy"):
    """Draws a reliability diagram into a subplot."""
    accuracies = bin_data["accuracies"]
    confidences = bin_data["confidences"]
    counts = bin_data["counts"]
    bins = bin_data["bins"]

    bin_size = 1.0 / len(counts)
    positions = bins[:-1] + bin_size/2.0

    widths = bin_size
    alphas = 0.3
    min_count = np.min(counts)
    max_count = np.max(counts)
    normalized_counts = (counts - min_count) / (max_count - min_count)

    if draw_bin_importance == "alpha":
        alphas = 0.2 + 0.8*normalized_counts
    elif draw_bin_importance == "width":
        widths = 0.1*bin_size + 0.9*bin_size*normalized_counts

    colors = np.zeros((len(counts), 4))
    colors[:, 0] = 240 / 255.
    colors[:, 1] = 60 / 255.
    colors[:, 2] = 60 / 255.
    colors[:, 3] = alphas

    gap_plt = ax.bar(positions, np.abs(accuracies - confidences), 
                     bottom=np.minimum(accuracies, confidences), width=widths,
                     edgecolor=colors, color=colors, linewidth=1, label="Gap")

    acc_plt = ax.bar(positions, 0, bottom=accuracies, width=widths,
                     edgecolor="black", color="black", alpha=1.0, linewidth=3,
                     label="Accuracy")

    ax.set_aspect("equal")
    ax.plot([0,1], [0,1], linestyle = "--", color="gray")
    
    if draw_ece:
        ece = (bin_data["expected_calibration_error"] * 100)
        ax.text(0.98, 0.02, "ECE=%.2f" % ece, color="black", 
                ha="right", va="bottom", transform=ax.transAxes)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    #ax.set_xticks(bins)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    ax.legend(handles=[gap_plt, acc_plt])


def _confidence_histogram_subplot(ax, bin_data, 
                                  draw_averages=True,
                                  title="Examples per bin", 
                                  xlabel="Confidence",
                                  ylabel="Count"):
    """Draws a confidence histogram into a subplot."""
    counts = bin_data["counts"]
    bins = bin_data["bins"]

    bin_size = 1.0 / len(counts)
    positions = bins[:-1] + bin_size/2.0

    ax.bar(positions, counts, width=bin_size * 0.9)
   
    ax.set_xlim(0, 1)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if draw_averages:
        acc_plt = ax.axvline(x=bin_data["avg_accuracy"], ls="solid", lw=3, 
                             c="black", label="Accuracy")
        conf_plt = ax.axvline(x=bin_data["avg_confidence"], ls="dotted", lw=3, 
                              c="#444", label="Avg. confidence")
        ax.legend(handles=[acc_plt, conf_plt])


def _reliability_diagram_combined(bin_data, 
                                  draw_ece, draw_bin_importance, draw_averages, 
                                  title, figsize, dpi, return_fig):
    """Draws a reliability diagram and confidence histogram using the output
    from compute_calibration()."""
    figsize = (figsize[0], figsize[0] * 1.4)

    fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=figsize, dpi=dpi, 
                           gridspec_kw={"height_ratios": [4, 1]})

    plt.tight_layout()
    plt.subplots_adjust(hspace=-0.1)

    _reliability_diagram_subplot(ax[0], bin_data, draw_ece, draw_bin_importance, 
                                 title=title, xlabel="")

    # Draw the confidence histogram upside down.
    orig_counts = bin_data["counts"]
    bin_data["counts"] = -bin_data["counts"]
    _confidence_histogram_subplot(ax[1], bin_data, draw_averages, title="")
    bin_data["counts"] = orig_counts

    # Also negate the ticks for the upside-down histogram.
    new_ticks = np.abs(ax[1].get_yticks()).astype(np.int)
    ax[1].set_yticklabels(new_ticks)    

    plt.show()

    if return_fig: return fig


def reliability_diagram(true_labels, pred_labels, confidences, num_bins=10,
                        draw_ece=True, draw_bin_importance=False, 
                        draw_averages=True, title="Reliability Diagram", 
                        figsize=(6, 6), dpi=72, return_fig=False):
    """Draws a reliability diagram and confidence histogram in a single plot.
    
    First, the model's predictions are divided up into bins based on their
    confidence scores.
    The reliability diagram shows the gap between average accuracy and average 
    confidence in each bin. These are the red bars.
    The black line is the accuracy, the other end of the bar is the confidence.
    Ideally, there is no gap and the black line is on the dotted diagonal.
    In that case, the model is properly calibrated and we can interpret the
    confidence scores as probabilities.
    The confidence histogram visualizes how many examples are in each bin. 
    This is useful for judging how much each bin contributes to the calibration
    error.
    The confidence histogram also shows the overall accuracy and confidence. 
    The closer these two lines are together, the better the calibration.
    
    The ECE or Expected Calibration Error is a summary statistic that gives the
    difference in expectation between confidence and accuracy. In other words,
    it's a weighted average of the gaps across all bins. A lower ECE is better.
    Arguments:
        true_labels: the true labels for the test examples
        pred_labels: the predicted labels for the test examples
        confidences: the predicted confidences for the test examples
        num_bins: number of bins
        draw_ece: whether to include the Expected Calibration Error
        draw_bin_importance: whether to represent how much each bin contributes
            to the total accuracy: False, "alpha", "widths"
        draw_averages: whether to draw the overall accuracy and confidence in
            the confidence histogram
        title: optional title for the plot
        figsize: setting for matplotlib; height is ignored
        dpi: setting for matplotlib
        return_fig: if True, returns the matplotlib Figure object
    """
    bin_data = compute_calibration(true_labels, pred_labels, confidences, num_bins)
    return _reliability_diagram_combined(bin_data, draw_ece, draw_bin_importance,
                                         draw_averages, title, figsize=figsize, 
                                         dpi=dpi, return_fig=return_fig)


def reliability_diagrams(results, num_bins=10,
                         draw_ece=True, draw_bin_importance=False, 
                         num_cols=4, dpi=72, return_fig=False):
    """Draws reliability diagrams for one or more models.
    
    Arguments:
        results: dictionary where the key is the model name and the value is
            a dictionary containing the true labels, predicated labels, and
            confidences for this model
        num_bins: number of bins
        draw_ece: whether to include the Expected Calibration Error
        draw_bin_importance: whether to represent how much each bin contributes
            to the total accuracy: False, "alpha", "widths"
        num_cols: how wide to make the plot
        dpi: setting for matplotlib
        return_fig: if True, returns the matplotlib Figure object
    """
    ncols = num_cols
    nrows = (len(results) + ncols - 1) // ncols
    figsize = (ncols * 4, nrows * 4)

    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, sharey=True, 
                           figsize=figsize, dpi=dpi, constrained_layout=True)

    for i, (plot_name, data) in enumerate(results.items()):
        y_true = data["true_labels"]
        y_pred = data["pred_labels"]
        y_conf = data["confidences"]
        
        bin_data = compute_calibration(y_true, y_pred, y_conf, num_bins)
        
        row = i // ncols
        col = i % ncols
        _reliability_diagram_subplot(ax[row, col], bin_data, draw_ece, 
                                     draw_bin_importance, 
                                     title="\n".join(plot_name.split()),
                                     xlabel="Confidence" if row == nrows - 1 else "",
                                     ylabel="Expected Accuracy" if col == 0 else "")

    for i in range(i + 1, nrows * ncols):
        row = i // ncols
        col = i % ncols        
        ax[row, col].axis("off")
        
    plt.show()

    if return_fig: return fig


class LagrangianOptimization:

    min_alpha = None
    max_alpha = None
    device = None
    original_optimizer = None
    batch_size_multiplier = None
    update_counter = 0

    def __init__(self, original_optimizer, device, init_alpha=0.55, min_alpha=-2, max_alpha=200, alpha_optimizer_lr=1e-2, batch_size_multiplier=None):
        self.min_alpha = min_alpha
        self.max_alpha = max_alpha
        self.device = device
        self.batch_size_multiplier = batch_size_multiplier
        self.update_counter = 0

        self.alpha = torch.tensor(init_alpha, device=device, requires_grad=True)
        self.optimizer_alpha = torch.optim.RMSprop([self.alpha], lr=alpha_optimizer_lr, centered=True)
        self.original_optimizer = original_optimizer

    def update(self, f, g):
        """
        L(x, lambda) = f(x) + lambda g(x)
        :param f_function:
        :param g_function:
        :return:
        """

        if self.batch_size_multiplier is not None and self.batch_size_multiplier > 1:
            if self.update_counter % self.batch_size_multiplier == 0:
                self.original_optimizer.zero_grad()
                self.optimizer_alpha.zero_grad()

            self.update_counter += 1
        else:
            self.original_optimizer.zero_grad()
            self.optimizer_alpha.zero_grad()

        loss = f + torch.nn.functional.softplus(self.alpha) * g
        loss.backward()

        if self.batch_size_multiplier is not None and self.batch_size_multiplier > 1:
            if self.update_counter % self.batch_size_multiplier == 0:
                self.original_optimizer.step()
                self.alpha.grad *= -1
                self.optimizer_alpha.step()
        else:
            self.original_optimizer.step()
            self.alpha.grad *= -1
            self.optimizer_alpha.step()

        if self.alpha.item() < -2:
            self.alpha.data = torch.full_like(self.alpha.data, -2)
        elif self.alpha.item() > 30:
            self.alpha.data = torch.full_like(self.alpha.data, 30)

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