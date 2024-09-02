import argparse
from copy import deepcopy
from tqdm import tqdm
import os
import subprocess
import sys
sys.path.append('../')
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score

from ggwas.data import ukbb_cohort
from ggwas.utils import print_sys, evaluate, compute_metrics, save_dict, \
                        load_dict, get_args, load_pretrained, save_model, \
                        get_gwas_results
from ggwas.params import main_data_path, cohort_data_path, kinship_path, withdraw_path, gwas_result_path

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='plink', choices = ['plink', 'fastgwa_full', 'fastgwa_match', 'gold_label', 'fastgwa_gold'])
parser.add_argument('--threshold', type=float, default=5e-8)

parser.add_argument('--data_split_seed', type=int, default=42)
parser.add_argument('--wandb', action="store_true", default=False)
parser.add_argument('--normalize_y', type=str, default='None', choices = ['quantile_normalization', 'std', 'log', 'None'])
parser.add_argument('--covar_file', type=str, default='real_pca15', choices = ['full', 'old', 'real_value', 'real_pca15'])

parser.add_argument('--test_set_fraction', type=float, default=0.7)
parser.add_argument('--pheno', type=str, default='50') 
parser.add_argument('--icd10', action="store_true", default=False)
parser.add_argument('--external_cohort', action="store_true", default=False)
parser.add_argument('--randomize', action="store_true", default=False)
parser.add_argument('--use_sample_size', action="store_true", default=False)
parser.add_argument('--sample_size', type=int, default=-1)
parser.add_argument('--randomize_seed', type=int, default=42)
parser.add_argument('--non_redundant_traits', action="store_true", default=False)

parser.add_argument('--simulate', action="store_true", default=False)
parser.add_argument('--num_causal_hits', type=int, default=1000)
parser.add_argument('--heritability', type=float, default=0.1)
parser.add_argument('--simulate_seed', type=int, default=42)
parser.add_argument('--simulate_graph', action="store_true", default=False)
parser.add_argument('--simulate_graph_func', action="store_true", default=False)
parser.add_argument('--small_cohort', type=int, default=-1)
parser.add_argument('--null_simulation', action="store_true", default=False)
parser.add_argument('--network_plant', action="store_true", default=False)


args = parser.parse_args()
print(args)

if args.icd10:
    binary = True
else:
    if args.non_redundant_traits and (args.pheno in ['body_BALDING1', 'cancer_BREAST', 'disease_ALLERGY_ECZEMA_DIAGNOSED', 'disease_HYPOTHYROIDISM_SELF_REP', 'other_MORNINGPERSON', 'pigment_SUNBURN']):
        print('Using binary traits...')
        binary = True
    else:
        print('Using continuous traits...')
        binary = False    
    
if args.simulate:
    if args.simulate_graph:
        name = str(args.num_causal_hits) + '_' + str(args.simulate_seed) + '_' + str(args.heritability) + '_graph_v2'
        pheno_path = '/dfs/project/datasets/20220524-ukbiobank/data/genetics/simulate_data/result/causal_snp_' + str(args.num_causal_hits) + '_' + str(args.simulate_seed) + '_' + str(args.heritability) + '_graph_v2.phen'
    elif args.network_plant:
        if args.small_cohort != -1:
            name = str(args.num_causal_hits) + '_' + str(args.simulate_seed) + '_' + str(args.heritability)+ '_' + str(args.small_cohort) + '_graph_funct_v2_ggi'
            pheno_path = '/dfs/project/datasets/20220524-ukbiobank/data/genetics/simulate_data/result/causal_snp_' + str(args.num_causal_hits) + '_' + str(args.simulate_seed) + '_' + str(args.heritability) + '_' + str(args.small_cohort) + '_graph_funct_v2_ggi.phen'
            if not os.path.exists(pheno_path):
                print('creating pheno_path at: ' + pheno_path)
                pd.read_csv('/dfs/project/datasets/20220524-ukbiobank/data/genetics/simulate_data/result/causal_snp_' + str(args.num_causal_hits) + '_' + str(args.simulate_seed) + '_' + str(args.heritability) + '_graph_funct_v2_ggi.phen', header = None).sample(n = args.small_cohort, random_state = 42).to_csv(pheno_path, index = False, header = None)
                
        else:
            name = str(args.num_causal_hits) + '_' + str(args.simulate_seed) + '_' + str(args.heritability)+ '_graph_funct_v2_ggi'
            pheno_path = '/dfs/project/datasets/20220524-ukbiobank/data/genetics/simulate_data/result/causal_snp_' + str(args.num_causal_hits) + '_' + str(args.simulate_seed) + '_' + str(args.heritability) + '_graph_funct_v2_ggi.phen'
    elif args.simulate_graph_func:
        if args.small_cohort != -1:
            pheno_path = '/dfs/project/datasets/20220524-ukbiobank/data/genetics/simulate_data/result/causal_snp_' + str(args.num_causal_hits) + '_' + str(args.simulate_seed) + '_' + str(args.heritability) + '_' + str(args.small_cohort) + '_graph_funct_v2.phen'
            if not os.path.exists(pheno_path):
                print('creating pheno_path at: ' + pheno_path)
                pd.read_csv('/dfs/project/datasets/20220524-ukbiobank/data/genetics/simulate_data/result/causal_snp_' + str(args.num_causal_hits) + '_' + str(args.simulate_seed) + '_' + str(args.heritability) + '_graph_funct_v2.phen', header = None).sample(n = args.small_cohort, random_state = 42).to_csv(pheno_path, index = False, header = None)

            name = str(args.num_causal_hits) + '_' + str(args.simulate_seed) + '_' + str(args.heritability) + '_' + str(args.small_cohort)+ '_graph_funct_v2'                
        else:
            name = str(args.num_causal_hits) + '_' + str(args.simulate_seed) + '_' + str(args.heritability)+ '_graph_funct_v2'
            pheno_path = '/dfs/project/datasets/20220524-ukbiobank/data/genetics/simulate_data/result/causal_snp_' + str(args.num_causal_hits) + '_' + str(args.simulate_seed) + '_' + str(args.heritability) + '_graph_funct_v2.phen'
    else:
        if args.small_cohort != -1:
            if args.null_simulation:
                pheno_path = '/dfs/project/datasets/20220524-ukbiobank/data/genetics/simulate_data/result/causal_snp_' + str(args.simulate_seed) + '_0_' + str(args.small_cohort) + '_null.phen'
                name = str(args.simulate_seed) + '_0_' + str(args.small_cohort) + '_null'
            else:
                pheno_path = '/dfs/project/datasets/20220524-ukbiobank/data/genetics/simulate_data/result/causal_snp_' + str(args.num_causal_hits) + '_' + str(args.simulate_seed) + '_' + str(args.heritability) + '_' + str(args.small_cohort) + '.phen'
                if not os.path.exists(pheno_path):
                    print('creating pheno_path at: ' + pheno_path)
                    pd.read_csv('/dfs/project/datasets/20220524-ukbiobank/data/genetics/simulate_data/result/causal_snp_' + str(args.num_causal_hits) + '_' + str(args.simulate_seed) + '_' + str(args.heritability) + '.phen', header = None).sample(n = args.small_cohort, random_state = 42).to_csv(pheno_path, index = False, header = None)
                    
                name = str(args.num_causal_hits) + '_' + str(args.simulate_seed) + '_' + str(args.heritability) + '_' + str(args.small_cohort)
        else:
            if args.null_simulation:
                pheno_path = '/dfs/project/datasets/20220524-ukbiobank/data/genetics/simulate_data/result/causal_snp_' + str(args.simulate_seed) + '_0_null.phen'
                name = str(args.simulate_seed) + '_0_null'
            else:
                pheno_path = '/dfs/project/datasets/20220524-ukbiobank/data/genetics/simulate_data/result/causal_snp_' + str(args.num_causal_hits) + '_' + str(args.simulate_seed) + '_' + str(args.heritability) + '.phen'
                name = str(args.num_causal_hits) + '_' + str(args.simulate_seed) + '_' + str(args.heritability)
        
    
    if (args.null_simulation) or ((args.small_cohort != -1) and (args.small_cohort <= 3000)):
        gwas_result_path_file = os.path.join(gwas_result_path, name + '.PHENO1.glm.linear')
        if os.path.exists(gwas_result_path_file):
            print_sys('local file detected... not running again...')
        else:
            print_sys('Running GWAS on PLINK... Takes > 1 hour...')
            cmd = "bash ./plink_python_interface_gwas.sh " + pheno_path + ' ' + name
            p = subprocess.Popen(cmd, shell = True, stdout=subprocess.PIPE, bufsize=1)
            for line in iter(p.stdout.readline, b''):
                print_sys(line)
            p.stdout.close()
            p.wait()
            
    else:
        gwas_result_path_file = os.path.join(gwas_result_path, name + '.fastGWA')
        print(gwas_result_path_file)
    
        if os.path.exists(gwas_result_path_file):
            print_sys('local file detected... not running again...')
        else:
            print_sys('Running GWAS on fastGWA... Takes > 1 hour...')
            cmd = "bash ./fastgwa_python_interface.sh " + pheno_path + ' ' + name
            p = subprocess.Popen(cmd, shell = True, stdout=subprocess.PIPE, bufsize=1)
            for line in iter(p.stdout.readline, b''):
                print_sys(line)
            p.stdout.close()
            p.wait()
            
else:    
    if args.model not in ['gold_label', 'fastgwa_gold']:
        if args.use_sample_size:
            name = str(args.pheno) + '_' + args.model + '_' + str(args.sample_size) + '_' + str(args.data_split_seed)
        else:
            name = str(args.pheno) + '_' + args.model + '_' + str(args.test_set_fraction) + '_' + str(args.data_split_seed)
        if args.threshold != 5e-8:
            name += ('_' + str(args.threshold))
    elif args.model == 'fastgwa_gold':
        name = str(args.pheno) + '_with_rel_fastgwa'
        args.test_set_fraction = 1
    else:
        name = str(args.pheno) + '_no_rel'
        args.test_set_fraction = 1

    if args.external_cohort:
        name += '_sep_cohort'

    if args.randomize:
        name += '_randomize' + str(args.randomize_seed)

    if args.wandb:
        import wandb
        wandb.init(project='GGWAS', name=name)
        wandb.config.update(args)


    if args.model in ['fastgwa_full', 'fastgwa_match', 'fastgwa_gold']:
        keep_relatives = True
    else:
        keep_relatives = False

    if args.model =='fastgwa_match':
        fastgwa_match = True
    else:
        fastgwa_match = False

    ukbb_data = ukbb_cohort(main_data_path, cohort_data_path, withdraw_path, keep_relatives=keep_relatives)
    
    if (not args.icd10) and (not args.non_redundant_traits):
        args.pheno = int(args.pheno)
    
    if args.non_redundant_traits:
        pheno = ukbb_data.get_external_traits(args.pheno, to_plink = True, random_seed = args.data_split_seed, sep_cohort = args.external_cohort, randomize = args.randomize, use_sample_size = args.use_sample_size, sample_size = args.sample_size, randomize_seed = args.randomize_seed)
    else:
        pheno = ukbb_data.get_phenotype(args.pheno, to_plink = True, normalize = args.normalize_y, frac = args.test_set_fraction, random_seed = args.data_split_seed, fastgwa_match = fastgwa_match, icd10 = args.icd10, sep_cohort = args.external_cohort, randomize = args.randomize, use_sample_size = args.use_sample_size, sample_size = args.sample_size, randomize_seed = args.randomize_seed)
 
    if args.model == 'plink':
        if args.use_sample_size:
            pheno_path = os.path.join(cohort_data_path, str(args.pheno) + '_plink_no_relatives_' + str(args.sample_size) + '_' + str(args.data_split_seed))
        else:
            pheno_path = os.path.join(cohort_data_path, str(args.pheno) + '_plink_no_relatives_' + str(args.test_set_fraction) + '_' + str(args.data_split_seed))
    elif args.model == 'fastgwa_full':
        if args.use_sample_size:
            pheno_path = os.path.join(cohort_data_path, str(args.pheno) + '_plink_with_relatives_' + str(args.sample_size) + '_' + str(args.data_split_seed))
        else:
            pheno_path = os.path.join(cohort_data_path, str(args.pheno) + '_plink_with_relatives_' + str(args.test_set_fraction) + '_' + str(args.data_split_seed))
    elif args.model == 'fastgwa_match':
        pheno_path = os.path.join(cohort_data_path, str(args.pheno) + '_plink_with_relatives_' + str(args.test_set_fraction) + '_' + str(args.data_split_seed) + '_match')
    elif args.model == 'gold_label':
        pheno_path = os.path.join(cohort_data_path, str(args.pheno) + '_plink_no_relatives')
    elif args.model == 'fastgwa_gold':
        pheno_path = os.path.join(cohort_data_path, str(args.pheno) + '_plink_with_relatives')

    if args.external_cohort:
        pheno_path += '_sep_cohort'

    if args.randomize:
        pheno_path += '_randomize' + str(args.randomize_seed)

    pheno_path += '.txt'

## feed to PLINK/FastGWA to run GWAS on K% of data

    if args.model in ['plink', 'gold_label']:
        if binary:
            gwas_result_path_file = os.path.join(gwas_result_path, name + '.PHENO1.glm.logistic.hybrid')
        else:
            gwas_result_path_file = os.path.join(gwas_result_path, name + '.PHENO1.glm.linear')
        if os.path.exists(gwas_result_path_file):
            print_sys('local file detected... not running again...')
        else:
            print_sys('Running GWAS on PLINK... Takes > 1 hour...')
            cmd = "bash ./plink_python_interface_gwas.sh " + pheno_path + ' ' + name
            p = subprocess.Popen(cmd, shell = True, stdout=subprocess.PIPE, bufsize=1)
            for line in iter(p.stdout.readline, b''):
                print_sys(line)
            p.stdout.close()
            p.wait()

    elif args.model in ['fastgwa_full', 'fastgwa_match', 'fastgwa_gold']:
        gwas_result_path_file = os.path.join(gwas_result_path, name + '.fastGWA')
        if os.path.exists(gwas_result_path_file):
            print_sys('local file detected... not running again...')
        else:
            print_sys('Running GWAS on fastGWA... Takes > 1 hour...')
            cmd = "bash ./fastgwa_python_interface.sh " + pheno_path + ' ' + name
            p = subprocess.Popen(cmd, shell = True, stdout=subprocess.PIPE, bufsize=1)
            for line in iter(p.stdout.readline, b''):
                print_sys(line)
            p.stdout.close()
            p.wait()
        
        
    df_gwas = pd.read_csv(gwas_result_path_file, sep = '\t')
    if args.model in ['plink', 'gold_label']:
        df_gwas['SNP'] = df_gwas['ID']
    df_gwas.to_csv(gwas_result_path_file, index = False, sep = '\t')