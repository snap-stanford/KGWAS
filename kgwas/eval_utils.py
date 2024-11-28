import numpy as np
import pandas as pd
import torch
from scipy import interpolate

from .utils import load_dict
import pandas as pd
import numpy as np
from copy import copy

def find_closest_x(df_pred, lower_bound=0, upper_bound=200, tolerance=0.01):
    upper = 1e-2
    lower = 1e-3
    
    while lower_bound <= upper_bound:
        mid = (lower_bound + upper_bound) / 2
        #result = len(np.where(df_pred.P_weighted.values * mid < 2e-4)[0]) / len(np.where(df_pred.P.values < 2e-4)[0])
        res1 = len(np.where((df_pred.P_weighted.values * mid < upper) & (df_pred.P_weighted.values * mid > lower))[0])
        res2 = len(np.where((df_pred.P.values < upper) & (df_pred.P.values > lower))[0])
        result = res1/res2
        if abs(result - 1) < tolerance:
            return mid
        elif result > 1:
            lower_bound = mid + tolerance
        else:
            upper_bound = mid - tolerance

    return mid

def get_clumps_gold_label(data_path, gold_label_gwas, t_p = 5e-8, no_hla = False, column = 'P', snp2ld_snps = None):
    snp2ld_snps_with_hla = load_dict(data_path + 'ld_score/ukb_white_ld_10MB.pkl')
    snp2ld_snps_no_hla = load_dict(data_path + 'ld_score/ukb_white_ld_10MB_no_hla.pkl')

    if not snp2ld_snps:
        if no_hla:
            snp2ld_snps = snp2ld_snps_no_hla
        else:
            snp2ld_snps = snp2ld_snps_with_hla
    clumps = []
    snps_in_clumps = []
    snp_hits = gold_label_gwas[gold_label_gwas[column] < t_p].sort_values(column).SNP.values
    for snp in snp_hits:
        if snp in snps_in_clumps:
            ## already in existing clumps => not create a new clump
            pass
        else:
            if snp in snp2ld_snps:
                # ld block
                clumps.append([snp] + snp2ld_snps[snp])
                snps_in_clumps += snp2ld_snps[snp]
                snps_in_clumps += [snp]
            else:
                # no other SNPs tagged
                clumps.append([snp])
                snps_in_clumps += [snp]
    return clumps

def get_meta_clumps(clumps, data_path):
    snp2cm = dict(pd.read_csv(data_path + 'misc_data/ukb_white_with_cm.bim', sep = '\t', header = None)[[1, 2]].values)
    snp2chr = dict(pd.read_csv(data_path + 'misc_data/ukb_white_with_cm.bim', sep = '\t', header = None)[[1, 0]].values)

    idx2clump = {'Clump ' + str(idx): i for idx, i in enumerate(clumps)}
    idx2clump_chromosome = {'Clump ' + str(idx): snp2chr[i[0]] for idx, i in enumerate(clumps)}
    idx2clump_cm = {'Clump ' + str(idx): snp2cm[i[0]] for idx, i in enumerate(clumps)}
    
    idx2clump_cm_min = {'Clump ' + str(idx): min([snp2cm[x] for x in i]) for idx, i in enumerate(clumps)}
    idx2clump_cm_max = {'Clump ' + str(idx): max([snp2cm[x] for x in i]) for idx, i in enumerate(clumps)}
    
    df_clumps = pd.DataFrame([idx2clump_chromosome, idx2clump_cm, idx2clump, idx2clump_cm_min, idx2clump_cm_max]).T.reset_index().rename(columns = {'index': 'Clump idx', 0: 'Chromosome', 1: 'cM',  2: 'Clump rsids', 3: 'cM_min',4: 'cM_max'})
    
    all_mega_clump_across_chr = []
    for chrom in df_clumps.Chromosome.unique():
        df_clump_chr = df_clumps[df_clumps.Chromosome == chrom]
        all_mega_clump = []
        cur_mega_clump = []
        base_cM = 0
        for i,cM_hit,cM_min,cM_max in df_clump_chr.sort_values('cM')[['Clump idx', 'cM', 'cM_min', 'cM_max']].values:
            if (cM_min - base_cM) < 0.1:
                cur_mega_clump.append(i)
                base_cM = cM_max
            else:
                ### this clump is >0.1 cM farther away from the previous clump
                all_mega_clump.append(cur_mega_clump)
                base_cM = cM_max
                cur_mega_clump = [i]
        all_mega_clump.append(cur_mega_clump)
        if len(all_mega_clump[0]) == 0:
            all_mega_clump_across_chr += all_mega_clump[1:]
        else:
            all_mega_clump_across_chr += all_mega_clump
    idx2mega_clump = {'Mega-Clump '+str(idx): i for idx, i in enumerate(all_mega_clump_across_chr)}
    
    def flatten(l):
        return [item for sublist in l for item in sublist]
    
    idx2mega_clump_rsid = {'Mega-Clump '+str(idx): flatten([idx2clump[j] for j in i]) for idx, i in enumerate(all_mega_clump_across_chr)}
    idx2mega_clump_chrom = {'Mega-Clump '+str(idx): idx2clump_chromosome[i[0]] for idx, i in enumerate(all_mega_clump_across_chr)}
    
    return idx2mega_clump, idx2mega_clump_rsid, idx2mega_clump_chrom
    
    
def get_mega_clump_query(data_path, clumps, snp_hits, no_hla = False, snp2ld_snps = None):
    snp2ld_snps_with_hla = load_dict(data_path + 'ld_score/ukb_white_ld_10MB.pkl')
    snp2ld_snps_no_hla = load_dict(data_path + 'ld_score/ukb_white_ld_10MB_no_hla.pkl')

    if not snp2ld_snps:
        if no_hla:
            snp2ld_snps = snp2ld_snps_no_hla
        else:
            snp2ld_snps = snp2ld_snps_with_hla
        
    clumps_pred = []
    snps_in_clumps_pred = []
    K = max(len(clumps) * 3, 100)
    for snp in snp_hits:
        ## top ranked snps
        if len(clumps_pred) >= K:
            ## just going to get the top K clumps where K is set to be very large number -> we don't generate all clumps since as K goes extremely large, they are never prioritized and evaluated.
            break
        else:
            if snp in snps_in_clumps_pred:
                ## already in previous found clumps, move forward
                pass
            else:
                if snp in snp2ld_snps:
                    # this snp has ld tagged snps
                    clumps_pred.append([snp] + snp2ld_snps[snp])
                    snps_in_clumps_pred += snp2ld_snps[snp]
                    snps_in_clumps_pred += [snp]
                else:
                    # this snp does not have ld tagged snps, at least in UKB
                    clumps_pred.append([snp])
                    snps_in_clumps_pred += [snp]
    idx2mega_clump_pred, idx2mega_clump_rsid_pred, idx2mega_clump_chrom_pred = get_meta_clumps(clumps_pred, data_path)
    return idx2mega_clump_pred, idx2mega_clump_rsid_pred, idx2mega_clump_chrom_pred

def get_curve(mega_clump_pred, mega_clump_gold):
    recall_k = {}
    precision_k = {}
    found_clump_idx = []
    clump_idx_record = {}
    pred_clump_has_hit_count = 0
    for k, query_clump in enumerate(mega_clump_pred):
        ## go through the predicted top ranked clumps one by one
        k += 1
        does_this_clump_overlap_with_any_true_clumps = False
        ## this is used to calculate precision, to see if this clump overlaps with any of the gold clumps
        for clump_idx, clump in enumerate(mega_clump_gold):
            ## overlaps with this gold clump
            if len(np.intersect1d(query_clump, clump)) > 0:
                if clump_idx not in found_clump_idx:
                    ## if the clump is never found before, flag it
                    found_clump_idx.append(clump_idx)
                does_this_clump_overlap_with_any_true_clumps = True
        clump_idx_record[k] = copy(found_clump_idx)
        if does_this_clump_overlap_with_any_true_clumps:
            pred_clump_has_hit_count += 1

        recall_k[k] = len(found_clump_idx)/len(mega_clump_gold)
        precision_k[k] = pred_clump_has_hit_count/k

    #sns.scatterplot([recall_k[k+1] for k in range(len(mega_clump_pred))], [precision_k[k+1] for k in range(len(mega_clump_pred))], s = 1)
    return recall_k, precision_k, clump_idx_record

def get_prec_recall(pred_hits, gold_hits):
    recall = len(np.intersect1d(pred_hits, gold_hits))/len(gold_hits)
    if len(pred_hits) != 0:
        precision = len(np.intersect1d(pred_hits, gold_hits))/len(pred_hits)
    else:
        precision = 0
    return {'recall': recall,
           'precision': precision}

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

def get_cluster_from_gwas(df, cluster_distance_threshold = 500000, \
                          threshold_extend = False, cluster_compare_threshold = None, \
                         verbose = True):
    
    cluster_chr_pos = {}
    cluster_chr_rs = {}

    for chr_num in df['#CHROM'].unique():
        df_hits_chr = df[df['#CHROM'] == chr_num]
        df_hits_chr = df_hits_chr.sort_values('POS')
        pos = df_hits_chr.POS.values
        rs = df_hits_chr.ID.values

        cluster_set = []
        cluster_set_rs = []

        cur_pos = pos[0]
        cur_rs = rs[0]
        cur_set = [cur_pos]
        cur_set_rs = [rs[0]]

        for idx, next_pos in enumerate(pos[1:]):

            if next_pos - cur_pos < cluster_distance_threshold:
                cur_set.append(next_pos)
                cur_set_rs.append(rs[idx + 1])
                if threshold_extend:
                    cur_pos = next_pos
            else:
                cluster_set.append(cur_set)
                cluster_set_rs.append(cur_set_rs)
                cur_pos = next_pos
                cur_set = [cur_pos]
                cur_set_rs = [rs[idx + 1]]

        cluster_set.append(cur_set)
        cluster_set_rs.append(cur_set_rs)

        cluster_chr_pos[chr_num] = cluster_set
        cluster_chr_rs[chr_num] = cluster_set_rs
        
    cluster_chr_pos_flatten = {}
    cluster_chr_cluster_idx_flatten = {}
    cluster_chr_cluster_pos2idx_flatten = {}

    for chr_num, cluster_list in cluster_chr_pos.items():
        pos_flatten = []
        idx_flatten = []
        for idx, cluster in enumerate(cluster_list):
            pos_flatten = pos_flatten + cluster
            idx_flatten = idx_flatten + [idx] * len(cluster)
        cluster_chr_pos_flatten[chr_num] = pos_flatten
        cluster_chr_cluster_idx_flatten[chr_num] = idx_flatten
        cluster_chr_cluster_pos2idx_flatten[chr_num] = dict(zip(pos_flatten, idx_flatten))
        
    if verbose:
        print('Number of clusters: ' + str(sum([len(j) for j in cluster_chr_pos.values()])))
    
    cluster_chr_range = {}
    for i,j in cluster_chr_pos.items():
        cluster_chr_range[i] = [(min(x) - cluster_compare_threshold, max(x) + cluster_compare_threshold) for x in j]
    
    return cluster_chr_pos, cluster_chr_rs, cluster_chr_pos_flatten, \
            cluster_chr_cluster_idx_flatten, cluster_chr_cluster_pos2idx_flatten, cluster_chr_range


def get_cluster_hits_from_pred(pred_hits, threshold, lr_uni, cluster_chr_pos_flatten, cluster_chr_cluster_pos2idx_flatten):
    df_hits = lr_uni[lr_uni.ID.isin(pred_hits)]
    df_hits['closest_cluster'] = df_hits.apply(lambda x: find_nearest(cluster_chr_pos_flatten[x['#CHROM']], x.POS), axis = 1)
    df_hits['distance2cluster'] = df_hits.apply(lambda x: abs(x.closest_cluster - x.POS), axis = 1)
    df_hits['include_as_cluster'] = df_hits.apply(lambda x: x.distance2cluster < threshold, axis = 1)
    df_hits['cluster_id'] = df_hits.apply(lambda x: str(x['#CHROM']) + '_' + str(cluster_chr_cluster_pos2idx_flatten[x['#CHROM']][x['closest_cluster']]), axis = 1)
    cluster2count = dict(df_hits[df_hits.include_as_cluster].cluster_id.value_counts())
    num_non_hits = len(df_hits[~df_hits.include_as_cluster])
    novel_rs_id = df_hits[~df_hits.include_as_cluster].ID.values
    print('Number of predicted hits: ' + str(len(pred_hits)))
    print('Number of predicted hits not in the existing clusters: ' + str(len(novel_rs_id)))
    print('Number of cluster hits: ' + str(len(cluster2count)))
    return cluster2count, num_non_hits, df_hits, novel_rs_id

def plot_cluster_range(chr_num, gnn_cluster_chr_range, cluster_chr_range, \
                       gold_cluster_chr_range, findor_cluster_chr_range, x_start = None, x_end = None, \
                       base_gwas_name = 'FastGWA', gold_ref_name = 'GWAS Catalog'):

    fig = plt.figure(figsize=(14, 3)) # Set the figure size
    ax = fig.add_subplot(111)
    
    if chr_num not in cluster_chr_range:
        cluster_chr_range[chr_num] = {}
    if chr_num not in gnn_cluster_chr_range:
        gnn_cluster_chr_range[chr_num] = {}
    if chr_num not in gold_cluster_chr_range:
        gold_cluster_chr_range[chr_num] = {}
        
    if chr_num not in findor_cluster_chr_range:
        findor_cluster_chr_range[chr_num] = {}
    
    for i in findor_cluster_chr_range[chr_num]:
        plt.plot(i, ['FINDOR', 'FINDOR'], '*-')  
    
    for i in gnn_cluster_chr_range[chr_num]:
        plt.plot(i, ['GNN', 'GNN'], 's-')

    for i in cluster_chr_range[chr_num]:
        plt.plot(i, [base_gwas_name, base_gwas_name], '^-')

    for i in gold_cluster_chr_range[chr_num]:
        plt.plot(i, [gold_ref_name, gold_ref_name], 'o-')  

    plt.xlabel('Position Index at Chromosome ' + str(chr_num))
    
    if x_start is not None:
        ax.set_xlim([x_start,x_end])
    plt.show()

def get_pr_curve(cluster_distance_threshold, gold_label_gwas_hits, method_hit_gwas, low_data_gwas_hits, \
                 cluster_compare_threshold = None, method_name = 'gnn'):
    if cluster_compare_threshold is None:
        cluster_compare_threshold = int(cluster_distance_threshold/2)
    gold_cluster_chr_pos, gold_cluster_chr_rs, \
    gold_cluster_chr_pos_flatten, gold_cluster_chr_cluster_idx_flatten, \
    gold_cluster_chr_cluster_pos2idx_flatten, gold_cluster_chr_range = get_cluster_from_gwas(gold_label_gwas_hits, \
                                                                     cluster_distance_threshold, \
                                                                    threshold_extend = threshold_extend, \
                                                                    cluster_compare_threshold = cluster_compare_threshold, \
                                                                    verbose = False)

    cluster_chr_pos, cluster_chr_rs, \
    cluster_chr_pos_flatten, cluster_chr_cluster_idx_flatten, \
    cluster_chr_cluster_pos2idx_flatten, cluster_chr_range = get_cluster_from_gwas(low_data_gwas_hits, \
                                                                cluster_distance_threshold, \
                                                                threshold_extend = threshold_extend, \
                                                                cluster_compare_threshold = cluster_compare_threshold, \
                                                                verbose = False)
    
    gnn_cluster_chr_pos, gnn_cluster_chr_rs, \
    gnn_cluster_chr_pos_flatten, gnn_cluster_chr_cluster_idx_flatten, \
    gnn_cluster_chr_cluster_pos2idx_flatten, gnn_cluster_chr_range = get_cluster_from_gwas(method_hit_gwas, \
                                                                    cluster_distance_threshold, \
                                                                    threshold_extend = threshold_extend, \
                                                                    cluster_compare_threshold = cluster_compare_threshold, \
                                                                    verbose = False)        
    
    total = sum([len(j) for i,j in gold_cluster_chr_range.items()])
    
    #plink_set_overlap = sum([len(j) for j in find_overlap_clusters(cluster_chr_range, gold_cluster_chr_range).values()])
    plink_set_total = sum([len(j) for i,j in cluster_chr_range.items()])
    
    plink_set_overlap_ref = 0
    plink_set_overlap_query = 0
    for j in find_overlap_clusters(cluster_chr_range, gold_cluster_chr_range).values():
        plink_set_overlap_ref += len(np.unique([set(i[1]) for i in j]))
        plink_set_overlap_query += len(np.unique([set(i[0]) for i in j]))
        
    #gnn_set_overlap = sum([len(j) for j in find_overlap_clusters(gnn_cluster_chr_range, gold_cluster_chr_range).values()])
    gnn_set_total = sum([len(j) for i,j in gnn_cluster_chr_range.items()])
    
    gnn_set_overlap_ref = 0
    gnn_set_overlap_query = 0
    for j in find_overlap_clusters(gnn_cluster_chr_range, gold_cluster_chr_range).values():
        gnn_set_overlap_ref += len(np.unique([set(i[1]) for i in j]))
        gnn_set_overlap_query += len(np.unique([set(i[0]) for i in j]))
    
    
    '''
    low_data_gold_hits = low_data_gwas[low_data_gwas.ID.isin(gold_label_gwas_hits.ID.values)]
    low_data_gold_hits['cluster_id'] = low_data_gold_hits.apply(lambda x: str(x['#CHROM']) + '_' + \
                                                            str(gold_cluster_chr_cluster_pos2idx_flatten[x['#CHROM']][x.POS]), axis = 1)
    cluster2min_p = dict(low_data_gold_hits.groupby('cluster_id').P.min())
    flat_clusters = [i for i,j in cluster2min_p.items() if j > 1e-3]
    gold_label_gwas_hits['closest_cluster'] = gold_label_gwas_hits.apply(lambda x: find_nearest(gold_cluster_chr_pos_flatten[x['#CHROM']], x.POS), axis = 1)
    gold_label_gwas_hits['distance2cluster'] = gold_label_gwas_hits.apply(lambda x: abs(x.closest_cluster - x.POS), axis = 1)
    gold_label_gwas_hits['cluster_id'] = gold_label_gwas_hits.apply(lambda x: str(x['#CHROM']) + '_' + str(gold_cluster_chr_cluster_pos2idx_flatten[x['#CHROM']][x['closest_cluster']]), axis = 1)
    pos_pred = np.unique(low_data_gwas_hits.ID.values.tolist() + pred_hits.tolist())
    flat_cluster_range = {}
    for i in flat_clusters:
        chr_num = int(i.split('_')[0])
        cluster_idx = int(i.split('_')[1])
        if chr_num in flat_cluster_range:
            flat_cluster_range[chr_num].append(gold_cluster_chr_range[chr_num][cluster_idx])
        else:
            flat_cluster_range[chr_num] = [gold_cluster_chr_range[chr_num][cluster_idx]]

    flat_cluster_recalled = sum([len(j) for j in find_overlap_clusters(gnn_cluster_chr_range, flat_cluster_range).values()])
    flat_cluster_recalled_plink = sum([len(j) for j in find_overlap_clusters(cluster_chr_range, flat_cluster_range).values()])

    '''
    
    if gnn_set_total == 0:
        gnn_set_precision = -1
    else:
        gnn_set_precision = gnn_set_overlap_query/gnn_set_total
    
    if plink_set_total == 0:
        plink_precision = -1
    else:
        plink_precision = plink_set_overlap_query/plink_set_total

    
    return {'plink_precision':plink_precision, 
            'plink_recall': plink_set_overlap_ref/total,
            method_name + '_precision': gnn_set_precision,
            method_name + '_recall': gnn_set_overlap_ref/total,
            'plink_set_overlap_ref': plink_set_overlap_ref,
            'plink_set_overlap_query': plink_set_overlap_query,
            'plink_set_total': plink_set_total,
            method_name + '_set_overlap_ref': gnn_set_overlap_ref,
            method_name + '_set_overlap_query': gnn_set_overlap_query,
            method_name + '_set_total': gnn_set_total,
            'total_set': total
            #'gnn_flat_cluster_recall': flat_cluster_recalled/len(flat_clusters),
            #'plink_flat_cluster_recall': flat_cluster_recalled_plink/len(flat_clusters)
           }

from tqdm import tqdm
def find_overlap_clusters(query_cluster2range, gold_cluster2range):
    set_found_cluster_all = {}
    for chr_num, eval_cluster in query_cluster2range.items():
        if chr_num in gold_cluster2range:
            gold_cluster = gold_cluster2range[chr_num]
            set_found_cluster = []
            for a in eval_cluster:
                for b in gold_cluster:
                    if (a[0] <= b[1]) and (b[0] <= a[1]):
                        set_found_cluster.append((a, b))
                        break
            set_found_cluster_all[chr_num] = set_found_cluster 

    return set_found_cluster_all


def find_non_overlap_clusters(query_cluster2range, gold_cluster2range):
    set_not_found_cluster_all = {}
    for chr_num, eval_cluster in query_cluster2range.items():
        gold_cluster = gold_cluster2range[chr_num]
        
        set_not_found_cluster = []
        for a in eval_cluster:
            set_found_cluster = []
            for b in gold_cluster:
                if (a[0] <= b[1]) and (b[0] <= a[1]):
                    set_found_cluster.append((a, b))
                    break
                    
            if len(set_found_cluster) == 0:
                set_not_found_cluster.append(a)
                
        set_not_found_cluster_all[chr_num] = set_not_found_cluster 

    return set_not_found_cluster_all


### eval support functions

def quantileNormalize(df_input):
    df = df_input.copy()
    #compute rank
    dic = {}
    for col in df:
        dic.update({col : sorted(df[col])})
    sorted_df = pd.DataFrame(dic)
    rank = sorted_df.mean(axis = 1).tolist()
    #sort
    for col in df:
        t = np.searchsorted(np.sort(df[col]), df[col])
        df[col] = [rank[i] for i in t]
    return df

def get_cluster_count(method_hit_gwas, cluster_distance_threshold, cluster_compare_threshold, threshold_extend, gold_cluster_chr_range):
    gnn_cluster_chr_pos, gnn_cluster_chr_rs, \
    gnn_cluster_chr_pos_flatten, gnn_cluster_chr_cluster_idx_flatten, \
    gnn_cluster_chr_cluster_pos2idx_flatten, gnn_cluster_chr_range = get_cluster_from_gwas(method_hit_gwas, \
                                                                    cluster_distance_threshold, \
                                                                    threshold_extend = threshold_extend, \
                                                                    cluster_compare_threshold = cluster_compare_threshold, \
                                                                    verbose = False)        

    total = sum([len(j) for i,j in gold_cluster_chr_range.items()])
    gnn_set_total = sum([len(j) for i,j in gnn_cluster_chr_range.items()])

    gnn_set_overlap_ref = 0
    gnn_set_overlap_query = 0
    for j in find_overlap_clusters(gnn_cluster_chr_range, gold_cluster_chr_range).values():
        gnn_set_overlap_ref += len(np.unique([set(i[1]) for i in j]))
        gnn_set_overlap_query += len(np.unique([set(i[0]) for i in j]))
        
        
    return {'set_overlap_ref': gnn_set_overlap_ref,
            'set_overlap_query': gnn_set_overlap_query,
            'set_total': gnn_set_total,
            'total_set': total
           }

## search every 100 until it is larger than k, then search every 10, then search every 1
def get_top_k_clusters(query_rank, top_hits_k_range, cluster_distance_threshold, cluster_compare_threshold, threshold_extend, gold_cluster_chr_range):
    snp_k = 0
    k_to_cluster = {}
    k_to_closest_x = {}
    for k in top_hits_k_range:
        while True:
            out = get_cluster_count(query_rank[:snp_k], cluster_distance_threshold, 
                          cluster_compare_threshold, threshold_extend, gold_cluster_chr_range)
            if out['set_total'] < k:
                snp_k += 100
            else:
                snp_k -= 100
                while True:
                    out = get_cluster_count(query_rank[:snp_k], cluster_distance_threshold, 
                          cluster_compare_threshold, threshold_extend, gold_cluster_chr_range)
                    if out['set_total'] < k:
                        snp_k += 10
                    else:
                        closest_x = snp_k
                        closest_distance = abs(out['set_total'] - k)
                        for x in range(snp_k - 10, snp_k):
                            out = get_cluster_count(query_rank[:x], cluster_distance_threshold, 
                                  cluster_compare_threshold, threshold_extend, gold_cluster_chr_range)
                            if abs(out['set_total'] - k) <= closest_distance:
                                closest_x = x
                                closest_distance = abs(out['set_total'] - k)
                        break
                break

        k_to_cluster[k] = get_cluster_count(query_rank[:closest_x], cluster_distance_threshold, 
                      cluster_compare_threshold, threshold_extend, gold_cluster_chr_range)
        k_to_closest_x[k] = closest_x
        
    return k_to_cluster, k_to_closest_x


def storey_pi_estimator(gwas_data, bin_index):
    """
    Estimate pi0/pi1 using Storey and Tibshirani (PNAS 2003) estimator.
    Argss
    =====
    bin_index: array of indices for a particular bin
    """
    pvalue = gwas_data.loc[bin_index,'P'] # extract pvalues from specific bin based index
        
    #assert(pvalue.min() >= 0 and pvalue.max() <= 1), "Error: p-values should be between 0 and 1"
    total_tests = float(len(pvalue))
    pi0 = []
    lam = np.arange(0.05, 0.95, 0.05)
    counts = np.array([(pvalue > i).sum() for i in np.arange(0.05, 0.95, 0.05)])
    for l in range(len(lam)):
        pi0.append(counts[l] / (total_tests * (1 - lam[l])))

    # fit  cubic spline
    if not np.all(np.isfinite(pi0)):
        print("Not all pi0 is finite!!! filtering to finite indices...")
        finite_indices = np.isfinite(pi0)
        lam = lam[finite_indices]
        pi0 = pi0[finite_indices]
    
    cubic_spline = interpolate.CubicSpline(lam, pi0)
    pi0_est = cubic_spline(lam[-1])
    if(pi0_est >1): #take care of out of bounds estimate
        pi0_est = 1
    return pi0_est

def storey_ribshirani_integrate(gwas_data, column = 'pred', num_bins = 100):
    num_bins = float(num_bins)
    quantiles = np.arange(0, 1 + 1 / (num_bins+1), 1 / num_bins)
    predicted_tagged_variance_quantiles = gwas_data[column].quantile(quantiles)
    #expand top quantiles to ensure everything is within range
    predicted_tagged_variance_quantiles[0] = predicted_tagged_variance_quantiles[0]-1
    predicted_tagged_variance_quantiles[1] = predicted_tagged_variance_quantiles[1]+1
    predicted_tagged_variance_quantiles = predicted_tagged_variance_quantiles.drop_duplicates()
    num_bins = len(predicted_tagged_variance_quantiles)-1
    bins = pd.cut(gwas_data[column], predicted_tagged_variance_quantiles, labels=np.arange(num_bins)) #create the lables
    gwas_data['bin_number'] = bins

    gwas_data['pi0'] = None
    
    if (gwas_data['P'].min() < 0) or (gwas_data['P'].max() > 1):
        print("detected p-values < 0 or > 1, please double check. we clipped it to 0-1 for now...")
        gwas_data['P'] = gwas_data['P'].clip(lower=0, upper=1)
        
    #print("Estimating pi0 within each bin")
    for i in range(num_bins):
        bin_index = gwas_data['bin_number']== i # determine index of snps in bin number i
        if len(gwas_data[bin_index])>0:
            pi0 = storey_pi_estimator(gwas_data, bin_index)
            ## preventing exploding weights
            if pi0 < 1e-5:
                pi0 = 1e-5
            if pi0 > 1-1e-5:
                pi0 = 1-1e-5
            gwas_data.loc[bin_index, 'pi0'] = pi0
    if any(gwas_data['pi0'] == 1): # if a bin is estimated to be all null, give the smallest non-null weight
        one_index = gwas_data['pi0'] == 1
        largest_pi0 = gwas_data.loc[~one_index]['pi0'].max()
        gwas_data.loc[one_index,'pi0'] = largest_pi0
        
    if any(gwas_data['pi0'] == 0): # if a bin is estimated to be all alternative, give the largest non-null weight
        one_index = gwas_data['pi0'] == 0
        largest_pi0 = gwas_data.loc[~one_index]['pi0'].min()
        gwas_data.loc[one_index,'pi0'] = largest_pi0
        
    #print("Re-weighting SNPs")
    weights = (1-gwas_data['pi0'])/(gwas_data['pi0'])
    
    ## avoiding exploding p-values
    #weights = np.maximum(1, weights.values)
    mean_weight = weights.mean()
    weights = weights/mean_weight #normalize weights to have mean 1
    
    ## avoiding exploding p-values
    #weights = np.maximum(1, weights.values)
    
    gwas_data['weights'] = weights
    gwas_data['P_weighted'] = gwas_data['P']/weights #reweight SNPs

    index = gwas_data['P_weighted'] > 1
    #gwas_data.loc[index, 'P_weighted'] = 1
    gwas_data.loc[index, 'P_weighted'] = gwas_data['P'][index] ## using original p-value when above 1
    gwas_data.loc[gwas_data['P_weighted'].isnull(), 'P_weighted'] = 1    
    return gwas_data['P_weighted'].values