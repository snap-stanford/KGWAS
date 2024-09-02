from .utils import load_dict
import pandas as pd
import numpy as np
from copy import copy

snp2ld_snps_with_hla = load_dict('/dfs/project/datasets/20220524-ukbiobank/data/ukb_white_ld_10MB.pkl')
snp2ld_snps_no_hla = load_dict('/dfs/project/datasets/20220524-ukbiobank/data/ukb_white_ld_10MB_no_hla.pkl')

snp2cm = dict(pd.read_csv('/dfs/project/datasets/20220524-ukbiobank/data/genetics/ukb_white_with_cm.bim', sep = '\t', header = None)[[1, 2]].values)
snp2chr = dict(pd.read_csv('/dfs/project/datasets/20220524-ukbiobank/data/genetics/ukb_white_with_cm.bim', sep = '\t', header = None)[[1, 0]].values)


def get_clumps_gold_label(gold_label_gwas, t_p = 5e-8, no_hla = False, column = 'P', snp2ld_snps = None):
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

def get_meta_clumps(clumps):
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
    
    
def get_mega_clump_query(clumps, snp_hits, no_hla = False, snp2ld_snps = None):
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
    idx2mega_clump_pred, idx2mega_clump_rsid_pred, idx2mega_clump_chrom_pred = get_meta_clumps(clumps_pred)
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