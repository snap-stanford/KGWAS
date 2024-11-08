from torch_geometric.data import HeteroData
import torch_geometric.transforms as T

from sklearn.model_selection import train_test_split
import pandas as pd
import torch
import numpy as np
import pickle
import os

from .utils import ldsc_regression_weights, load_dict
from .params import scdrs_traits

class KGWAS_Data:
    def __init__(self, data_path = './data'):
        self.data_path = data_path
        if not os.path.exists(data_path):
            os.makedirs(data_path)

    def load_kg(self, snp_init_emb = 'enformer', 
                    go_init_emb = 'random',
                    gene_init_emb = 'esm', 
                    sample_edges = False, 
                    sample_ratio = 1):

        data_path = self.data_path
        
        ## Load KG

        print('--loading KG---')
        idx2id = load_dict(data_path + 'cell_kg/network/node_idx2id.pkl')
        edge_index_all = load_dict(data_path + 'cell_kg/network/edge_index.pkl')
        id2idx = load_dict(data_path + 'cell_kg/network/node_id2idx.pkl')  
        self.id2idx = id2idx
        self.idx2id = idx2id
        
        data = HeteroData()

        ## Load initialized embeddings
        
        if snp_init_emb == 'random':
            print('--using random SNP embedding--')

            data['SNP'].x = torch.rand((len(idx2id['SNP']), 128), requires_grad = False)
            snp_init_dim_size = 128
        elif snp_init_emb == 'kg':
            print('--using KG SNP embedding--')

            id2idx_kg = load_dict(data_path + 'cell_kg/node_emb/transe_emb/transe_emb_id2idx_kg.pkl')
            kg_emb = load_dict(data_path + 'cell_kg/node_emb/transe_emb/transe_emb_inverse_triplets.pkl')
            node_map = idx2id['SNP']
            data['SNP'].x = torch.vstack([torch.tensor(kg_emb[id2idx_kg[node_map[i]]]) if node_map[i] in id2idx_kg \
                                              else torch.rand(50, requires_grad = False) for i in range(len(node_map))])
            snp_init_dim_size = 50

        elif snp_init_emb == 'cadd':
            print('--using CADD SNP embedding--')

            df_variant = pd.read_csv(data_path + 'cell_kg/node_emb/variant_emb/cadd_feat.csv')
            df_variant = df_variant.set_index('Unnamed: 0')
            variant_feat = df_variant.values
            node_map = idx2id['SNP']
            rs2idx_feat = dict(zip(df_variant.index.values, range(len(df_variant.index.values)))) 
            data['SNP'].x = torch.vstack([torch.tensor(variant_feat[rs2idx_feat[node_map[i]]]) if node_map[i] in rs2idx_feat \
                                                  else torch.rand(64, requires_grad = False) for i in range(len(node_map))]).float()
            snp_init_dim_size = 64


        elif snp_init_emb == 'baselineLD': 
            print('--using baselineLD SNP embedding--')
            node_map = idx2id['SNP']
            rs2idx_feat = load_dict(data_path + 'cell_kg/node_emb/variant_emb/baselineld_feat.pkl')
            data['SNP'].x = torch.vstack([torch.tensor(rs2idx_feat[node_map[i]]) if node_map[i] in rs2idx_feat \
                                                  else torch.rand(70, requires_grad = False) for i in range(len(node_map))]).float()
            snp_init_dim_size = 70

        elif snp_init_emb == 'SLDSC': 
            print('--using SLDSC SNP embedding--')
            node_map = idx2id['SNP']
            rs2idx_feat = load_dict(data_path + 'cell_kg/node_emb/variant_emb/sldsc_feat.pkl')
            data['SNP'].x = torch.vstack([torch.tensor(rs2idx_feat[node_map[i]]) if node_map[i] in rs2idx_feat \
                                                  else torch.rand(165, requires_grad = False) for i in range(len(node_map))]).float()
            snp_init_dim_size = 165 
        
        elif snp_init_emb == 'enformer':
            print('--using enformer SNP embedding--')
            node_map = idx2id['SNP']
            rs2idx_feat = load_dict(data_path + 'cell_kg/node_emb/variant_emb/enformer_feat.pkl')
            data['SNP'].x = torch.vstack([torch.tensor(rs2idx_feat[node_map[i]]) if node_map[i] in rs2idx_feat \
                                                  else torch.rand(20, requires_grad = False) for i in range(len(node_map))]).float()
            snp_init_dim_size = 20 
        
        
        if go_init_emb == 'random': 
            print('--using random go embedding--')

            for rel in ['CellularComponent', 'BiologicalProcess', 'MolecularFunction']:
                data[rel].x = torch.rand((len(idx2id[rel]), 128), requires_grad = False)
            go_init_dim_size = 128
        elif go_init_emb == 'kg':
            print('--using KG go embedding--')

            id2idx_kg = load_dict(data_path + 'cell_kg/node_emb/transe_emb/transe_emb_id2idx_kg.pkl')
            kg_emb = load_dict(data_path + 'cell_kg/node_emb/transe_emb/transe_emb_inverse_triplets.pkl')

            for rel in ['CellularComponent', 'BiologicalProcess', 'MolecularFunction']:
                node_map = idx2id[rel]
                data[rel].x = torch.vstack([torch.tensor(kg_emb[id2idx_kg[node_map[i]]]) if node_map[i] in id2idx_kg \
                                              else torch.rand(50, requires_grad = False) for i in range(len(node_map))])
            go_init_dim_size = 50

        elif go_init_emb == 'biogpt':
            print('--using biogpt go embedding--')

            go2idx_feat = load_dict(data_path + 'cell_kg/node_emb/program_emb/biogpt_feat.pkl')
            for rel in ['CellularComponent', 'BiologicalProcess', 'MolecularFunction']:
                node_map = idx2id[rel]
                data[rel].x = torch.vstack([torch.tensor(go2idx_feat[node_map[i]]) if node_map[i] in go2idx_feat \
                                                  else torch.rand(1600, requires_grad = False) for i in range(len(node_map))]).float()
            go_init_dim_size = 1600


        if gene_init_emb == 'random':   
            print('--using random gene embedding--')

            data['Gene'].x = torch.rand((len(idx2id['Gene']), 128), requires_grad = False)
            gene_init_dim_size = 128
        elif gene_init_emb == 'kg':
            print('--using KG gene embedding--')
            id2idx_kg = load_dict(data_path + 'cell_kg/node_emb/transe_emb/transe_emb_id2idx_kg.pkl')
            kg_emb = load_dict(data_path + 'cell_kg/node_emb/transe_emb/transe_emb_inverse_triplets.pkl')
            node_map = idx2id['Gene']
            data['Gene'].x = torch.vstack([torch.tensor(kg_emb[id2idx_kg[node_map[i]]]) if node_map[i] in id2idx_kg \
                                          else torch.rand(50, requires_grad = False) for i in range(len(node_map))])
            gene_init_dim_size = 50

        elif gene_init_emb == 'esm':
            print('--using ESM gene embedding--')

            gene2idx_feat = load_dict(data_path + 'cell_kg/node_emb/gene_emb/esm_feat.pkl')
            node_map = idx2id['Gene']
            data['Gene'].x = torch.vstack([torch.tensor(gene2idx_feat[node_map[i]]) if node_map[i] in gene2idx_feat \
                                              else torch.rand(5120, requires_grad = False) for i in range(len(node_map))]).float()
            gene_init_dim_size = 5120
        elif gene_init_emb == 'pops':
            print('--using PoPs expression+PPI+pathways gene embedding--')

            gene2idx_feat = load_dict(data_path + 'cell_kg/node_emb/gene_emb/pops_feat.pkl')
            node_map = idx2id['Gene']
            data['Gene'].x = torch.vstack([torch.tensor(gene2idx_feat[node_map[i]]) if node_map[i] in gene2idx_feat \
                                              else torch.rand(57742, requires_grad = False) for i in range(len(node_map))]).float()
            gene_init_dim_size = 57742
        elif gene_init_emb == 'pops_expression':
            print('--using PoPs expression only gene embedding--')

            gene2idx_feat = load_dict(data_path + 'cell_kg/node_emb/gene_emb/pops_expression_feat.pkl')
            node_map = idx2id['Gene']
            data['Gene'].x = torch.vstack([torch.tensor(gene2idx_feat[node_map[i]]) if node_map[i] in gene2idx_feat \
                                              else torch.rand(40546, requires_grad = False) for i in range(len(node_map))]).float()
            gene_init_dim_size = 40546    
        
        
        self.gene_init_dim_size = gene_init_dim_size
        self.go_init_dim_size = go_init_dim_size
        self.snp_init_dim_size = snp_init_dim_size
        
        for i,j in edge_index_all.items():
            
            if sample_edges:
                edge_index = torch.tensor(j)
                num_edges = edge_index.size(1)
                num_samples = int(num_edges * sample_ratio)
                indices = torch.randperm(num_edges)[:num_samples]
                sampled_edge_index = edge_index[:, indices]
                print(i, ' sampling ratio ', sample_ratio, ' from ', edge_index.shape[1], ' to ', sampled_edge_index.shape[1])
                data[i].edge_index = sampled_edge_index
            else:
                data[i].edge_index = torch.tensor(j)
        data = T.ToUndirected()(data)
        data = T.AddSelfLoops()(data)
        self.data = data

    def load_simulation_gwas(self, simulation_type, seed):
        data_path = self.data_path
        print('Using simulation data....')
        small_cohort = 5000
        num_causal_hits = 20000
        heritability = 0.3
        self.sample_size = small_cohort
        if simulation_type == 'causal_link':
            lr_uni = pd.read_csv(data_path + 'simulation_gwas/causal_link_simulation/' + str(num_causal_hits) + '_' + str(seed) + '_' + str(heritability) + '_graph_funct_v2_ggi.fastGWA', sep = '\t')
        elif simulation_type == 'causal':
            lr_uni = pd.read_csv(data_path + 'simulation_gwas/causal_simulation/' + str(num_causal_hits) + '_' + str(seed) + '_' + str(heritability) + '_' + str(small_cohort) + '_graph_funct_v2.fastGWA', sep = '\t')
        elif simulation_type == 'null':
            lr_uni = pd.read_csv(data_path + 'simulation_gwas/null_simulation/' + str(num_causal_hits) + '_' + str(seed) + '_' + str(heritability) + '_' + str(small_cohort) + '.fastGWA', sep = '\t')
           
        if ('SNP' in lr_uni.columns.values) and ('ID' in lr_uni.columns.values):
            self.lr_uni = lr_uni.rename(columns = {'CHR': '#CHROM'})
        else:
            self.lr_uni = lr_uni.rename(columns = {'CHR': '#CHROM', 'SNP': 'ID'})
        self.seed = seed
        self.pheno = 'simulation'
    
    def load_external_gwas(self, path, seed = 42):
        lr_uni = pd.read_csv(path)
        if 'CHR' not in lr_uni.columns.values:
            raise ValueError('CHR chromosome not in the file!')
        if 'SNP' not in lr_uni.columns.values:
            raise ValueError('SNP column not in the file!')
        if 'P' not in lr_uni.columns.values:
            raise ValueError('P column not in the file!')  
        if 'N' not in lr_uni.columns.values:
            raise ValueError('N column number of sample size not in the file!')  
        lr_uni = lr_uni.rename(columns = {'CHR': '#CHROM', 'SNP': 'ID'})
        
        self.lr_uni = lr_uni
        self.sample_size = lr_uni.N.values[0]
        self.pheno = 'EXTERNAL'
        self.seed = seed
        
        
    def load_full_gwas(self, pheno, seed):
        data_path = self.data_path
        if pheno in scdrs_traits:
            print('Using scdrs traits...')
            self.pheno = pheno
            lr_uni = pd.read_csv(data_path + 'scDRS_Data/sumstats_ukb_snps.csv')
            lr_uni = lr_uni[['CHR', 'SNP', 'POS', 'A1', 'A2', 'N', 'AF1', pheno]]
            lr_uni = lr_uni[lr_uni[pheno].notnull()].reset_index(drop = True)
            lr_uni = lr_uni.rename(columns = {'CHR': '#CHROM', 'SNP': 'ID', pheno: 'chi'})
            print('number of SNPs:', len(lr_uni))
            self.lr_uni = lr_uni
            self.seed = seed
            
            trait2size = pickle.load(open(data_path + 'scDRS_data/trait2size.pkl', 'rb'))
            self.sample_size = trait2size[pheno]
            
        else:
            ## load GWAS files
            self.pheno = pheno
            lr_uni = pd.read_csv(data_path + 'full_gwas/' + str(self.pheno) + '_with_rel_fastgwa.fastGWA', sep = '\t')
            lr_uni = lr_uni.rename(columns = {'CHR': '#CHROM', 'SNP': 'ID'})

            self.lr_uni = lr_uni
            self.seed = seed
            self.sample_size = 387113
    
    def load_gwas_subsample(self, pheno, sample_size, seed):
        data_path = self.data_path
        if pheno in ['body_BALDING1', 'cancer_BREAST', 'disease_ALLERGY_ECZEMA_DIAGNOSED', 'disease_HYPOTHYROIDISM_SELF_REP', 'other_MORNINGPERSON', 'pigment_SUNBURN']:
            binary = True
        else:
            binary = False
        ## load GWAS files
        self.sample_size = sample_size
        self.pheno = pheno
        if (sample_size > 3000):
            lr_uni = pd.read_csv(data_path + 'subsample_gwas/' + str(self.pheno) + \
                     '_fastgwa_full_'+ str(sample_size) + '_' + str(seed) + '.fastGWA', sep = '\t')
            lr_uni = lr_uni.rename(columns = {'CHR': '#CHROM', 'SNP': 'ID'})
        else:
            ## use PLINK if sample size <3000
            if binary:
                lr_uni = pd.read_csv(data_path + 'subsample_gwas/' + str(self.pheno) + \
                         '_plink_'+ str(sample_size) + '_' + str(seed) + '.PHENO1.glm.logistic.hybrid', sep = '\t')
            else:
                lr_uni = pd.read_csv(data_path + 'subsample_gwas/' + + str(self.pheno) + \
                         '_plink_'+ str(sample_size) + '_' + str(seed) + '.PHENO1.glm.linear', sep = '\t')
        self.lr_uni = lr_uni
        self.seed = seed

    def process_gwas_file(self, label = 'chi'):
        data_path = self.data_path
        lr_uni = self.lr_uni
        ## LD scores

        ld_scores = pd.read_csv(data_path + 'ld_score/filter_genotyped_ldscores.csv')
        w_ld_scores = pd.read_csv(data_path + 'ld_score/ldscores_from_data.csv')

        m = 15000000
        if 'N' not in lr_uni.columns.values:
            n = self.sample_size
        else:
            n = np.mean(lr_uni.N)
        h_g_2 = 0.5
        rs_id_2_ld_scores = dict(ld_scores.values)

        rs_id_2_ld_scores = dict(ld_scores.values)
        rs_id_2_w_ld = dict(w_ld_scores.values)

        ## use min ld score for snps with no ld score
        min_ld = min(rs_id_2_ld_scores.values())
        lr_uni['ld_score'] = lr_uni.ID.apply(lambda x: rs_id_2_ld_scores[x] if x in rs_id_2_ld_scores else min_ld)
        rs_id_2_ld_scores = dict(lr_uni[['ID', 'ld_score']].values)

        min_ld = min(rs_id_2_w_ld.values())
        ## the data LD is without the query SNP itself. so here add 1 
        lr_uni['w_ld_score'] = 1 + lr_uni.ID.apply(lambda x: rs_id_2_w_ld[x] if x in rs_id_2_w_ld else min_ld)
        rs_id_2_w_ld = dict(lr_uni[['ID', 'w_ld_score']].values)

        print('Using ldsc weight...')
        ld = np.array([rs_id_2_ld_scores[rs_id] for rs_id in lr_uni.ID.values])
        w_ld = np.array([rs_id_2_w_ld[rs_id] for rs_id in lr_uni.ID.values])

        ldsc_weight = ldsc_regression_weights(ld, w_ld, n, m, h_g_2)
        ldsc_weight = ldsc_weight/np.mean(ldsc_weight)
        print('ldsc_weight mean: ', np.mean(ldsc_weight))
        self.rs_id_to_ldsc_weight = dict(zip(lr_uni.ID.values, ldsc_weight))

        ## chi-square label
        if label == 'chi':
            if 'chi' in lr_uni.columns.values:
                print('chi pre-computed...')
                lr_uni['y'] = lr_uni['chi'].values
            else:    
                if self.pheno in (['body_BALDING1', 'cancer_BREAST', 'disease_ALLERGY_ECZEMA_DIAGNOSED', 'disease_HYPOTHYROIDISM_SELF_REP', 'other_MORNINGPERSON', 'pigment_SUNBURN']) and (self.sample_size <= 3000):
                    lr_uni['y'] = lr_uni['Z_STAT'].values**2
                    lr_uni['y'] = lr_uni.y.fillna(0)   
                else:
                    lr_uni['y'] = (lr_uni['BETA']/lr_uni['SE']).values**2
                    lr_uni['y'] = lr_uni.y.fillna(0)   
        elif label == 'residual-w-ld':
            lr_uni['y'] = (lr_uni['BETA']/lr_uni['SE']).values**2
            lr_uni['y'] = lr_uni.y.fillna(0)   
            lr_uni['ld_weight'] = lr_uni.ID.apply(lambda x: self.rs_id_to_ldsc_weight[x])
            import statsmodels.api as sm

            X = lr_uni.w_ld_score.values
            y = lr_uni.y.values
            weights = lr_uni.ld_weight.values
            X = sm.add_constant(X)
            model = sm.WLS(y, X, weights=weights)
            results = model.fit()
            y_pred = results.params[0] + results.params[1] * lr_uni.w_ld_score.values
            lr_uni['y'] = y - y_pred 
        elif label == 'residual-ld':
            lr_uni['y'] = (lr_uni['BETA']/lr_uni['SE']).values**2
            lr_uni['y'] = lr_uni.y.fillna(0)   
            lr_uni['ld_weight'] = lr_uni.ID.apply(lambda x: self.rs_id_to_ldsc_weight[x])
            import statsmodels.api as sm

            X = lr_uni.ld_score.values
            y = lr_uni.y.values
            weights = lr_uni.ld_weight.values
            X = sm.add_constant(X)
            model = sm.WLS(y, X, weights=weights)
            results = model.fit()
            y_pred = results.params[0] + results.params[1] * lr_uni.w_ld_score.values
            lr_uni['y'] = y - y_pred         
        elif label == 'residual-ld-ols':
            lr_uni['y'] = (lr_uni['BETA']/lr_uni['SE']).values**2
            lr_uni['y'] = lr_uni.y.fillna(0)   
            import statsmodels.api as sm

            X = lr_uni.ld_score.values
            y = lr_uni.y.values
            X = sm.add_constant(X)
            model = sm.OLS(y, X)
            results = model.fit()
            y_pred = results.params[0] + results.params[1] * lr_uni.w_ld_score.values
            lr_uni['y'] = y - y_pred 
        elif label == 'residual-ld-ols-abs':
            lr_uni['y'] = (lr_uni['BETA']/lr_uni['SE']).values**2
            lr_uni['y'] = lr_uni.y.fillna(0)   
            import statsmodels.api as sm

            X = lr_uni.ld_score.values
            y = lr_uni.y.values
            X = sm.add_constant(X)
            model = sm.OLS(y, X)
            results = model.fit()
            y_pred = results.params[0] + results.params[1] * lr_uni.w_ld_score.values
            lr_uni['y'] = np.abs(y - y_pred)
        elif label == 'residual-w-ld-ols':
            lr_uni['y'] = (lr_uni['BETA']/lr_uni['SE']).values**2
            lr_uni['y'] = lr_uni.y.fillna(0)   
            import statsmodels.api as sm

            X = lr_uni.w_ld_score.values
            y = lr_uni.y.values
            X = sm.add_constant(X)
            model = sm.OLS(y, X)
            results = model.fit()
            y_pred = results.params[0] + results.params[1] * lr_uni.w_ld_score.values
            lr_uni['y'] = y - y_pred     
            
        id2y = dict(lr_uni[['ID', 'y']].values)
        all_ids = lr_uni.ID.values
        self.all_ids = np.array([self.id2idx['SNP'][i] for i in all_ids])
        self.y = lr_uni.y.values
        #idx2y = dict(zip(self.all_ids, y))

        self.lr_uni = lr_uni

    def prepare_split(self, test_set_fraction_data = 0.05):

        ## split SNPs to train/test/valid
        train_val_ids, test_ids, y_train_val, y_test = train_test_split(self.all_ids, self.y, test_size=test_set_fraction_data, random_state=self.seed)
        train_ids, val_ids, y_train, y_val = train_test_split(train_val_ids, y_train_val, test_size=0.05, random_state=self.seed)

        self.train_input_nodes = ('SNP', train_ids)
        self.val_input_nodes = ('SNP', val_ids)
        self.test_input_nodes = ('SNP', test_ids)

        y_snp = torch.zeros(self.data['SNP'].x.shape[0]) - 1
        y_snp[train_ids] = torch.tensor(y_train).float()
        y_snp[val_ids] = torch.tensor(y_val).float()
        y_snp[test_ids] = torch.tensor(y_test).float()

        self.data['SNP'].y = y_snp
        for i in self.data.node_types:
            self.data[i].n_id = torch.arange(self.data[i].x.shape[0])

        self.data.train_mask = train_ids
        self.data.val_mask = val_ids
        self.data.test_mask = test_ids
        self.data.all_mask = self.all_ids
        #data = data.to(args.device)

