import os, sys
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
import torch
from .params import main_data_path, cohort_data_path, kinship_path, withdraw_path, fam_path
from .utils import get_fields, get_row_last_values, remove_kinships, save_dict, load_dict, print_sys


class ukbb_cohort:
    
    def __init__(self, main_data_path, cohort_data_path, withdraw_path, keep_relatives = False):
        self.keep_relatives = keep_relatives
        self.cohort_data_path = cohort_data_path
        self.main_data_path = main_data_path
        
        if keep_relatives:
            cohort_path = os.path.join(cohort_data_path, 'cohort_with_relatives.pkl')
        else:
            cohort_path = os.path.join(cohort_data_path, 'cohort_no_relatives.pkl')
        
        if not os.path.exists(cohort_path):
            print_sys('construct from scratch...')
            '''
            exclusions:
            the original uk biobank nature paper supplementary S 3.4:
            22006: Genetic ethnic grouping -> to retain only white british ancestry

            https://www.frontiersin.org/articles/10.3389/fgene.2022.866042/full
            22018: genetic relatedness exclusions
            22019: sex chromosome aneuploidy 
            31 <-> 22001: mismatch between self-reported sex and genetically determined sex 
            22010: recommended genomic analysis exclusions, signs of insufficient data quality

            (optional) further remove relatives based on KING relative scores, choose the first one in relative group
            remove the list of eids who do not want to be in the study anymore            
            '''
            
            all_field_ids = [22006, 22018, 22019, 22001, 22010, 31]
            df_main = get_fields(all_field_ids, main_data_path)
            cur_size = len(df_main)
            print_sys('Total sample size: ' + str(cur_size))
            df_main = df_main[df_main['22006-0.0'] == 1]
            print_sys('Keeping only white british ancestry (ID: 22006), cutting from ' + str(cur_size) + ' to ' + str(len(df_main)))
            cur_size = len(df_main)

            df_main = df_main[df_main['22018-0.0'].isnull()]
            print_sys('Removing genetics related samples (ID: 22018), cutting from ' + str(cur_size) + ' to ' + str(len(df_main)))
            cur_size = len(df_main)

            df_main = df_main[df_main['22019-0.0'].isnull()]
            print_sys('Removing sex chromosome aneuploidy (ID: 22019), cutting from ' + str(cur_size) + ' to ' + str(len(df_main)))
            cur_size = len(df_main)

            df_main = df_main[df_main['31-0.0'] == df_main['22001-0.0']]
            print_sys('Removing samples with mismatched self-reported sex and genetic determined sex (ID: 31 <-> 22001), cutting from ' + str(cur_size) + ' to ' + str(len(df_main)))
            cur_size = len(df_main)

            df_main = df_main[df_main['22010-0.0'].isnull()]
            print_sys('Removing samples with genomic data quality (ID: 22010), cutting from ' + str(cur_size) + ' to ' + str(len(df_main)))
            cur_size = len(df_main)
            
            save_dict(os.path.join(cohort_data_path, 'cohort_with_relatives.pkl'), df_main.eid.values)
            
            kinship_mask = remove_kinships(df_main.eid)
            df_main = df_main[kinship_mask]
            save_dict(os.path.join(cohort_data_path, 'cohort_no_relatives.pkl'), df_main.eid.values)
        else:
            print_sys('Found local copy...')
            
        self.cohort = load_dict(cohort_path)
        print_sys('There are ' + str(len(self.cohort)) + ' samples!')
        
        if keep_relatives:
            self.no_rel_eid = load_dict(os.path.join(cohort_data_path, 'cohort_no_relatives.pkl'))
            
        if os.path.exists(withdraw_path):
            ## todo: when there is a withdraw file, implement this...
            pass
        
    def get_covariates(self, to_plink = False, plink_num_pca = 15, return_full = False, plink_filter = False):
        '''
        covariates:

        31: sex
        21003: Age when attended assessment centre
        22009: pca
        54: assessment center
        batch from params.fam_path file
        '''
        covar_path = os.path.join(self.cohort_data_path, 'covariates_all.pkl')
        if os.path.exists(covar_path):
            print_sys('Found local copy...')
            self.covar = load_dict(covar_path)
        else:
            print_sys('construct co-variates from scratch...')
            df_covar = get_fields([31, 54, 21003, 22009], self.main_data_path)
            column_name_map = {'22009-0.' + str(i): 'pca ' + str(i) for i in range(1, 41)}
            column_name_map['31-0.0'] = 'sex'
            column_name_map['21003-0.0'] = 'age'
            column_name_map['54-0.0'] = 'assessment_center'
            self.covar = df_covar.rename(columns = column_name_map)
            
            enc = OneHotEncoder(handle_unknown='ignore')
            enc.fit(self.covar['assessment_center'].unique().reshape(-1,1))
            center_array = enc.transform(self.covar['assessment_center'].values.reshape(-1,1)).toarray()
            center_one_hot = pd.DataFrame(center_array).astype('int').rename(columns = dict(zip(range(22), ['center_' + str(i) for i in range(22)])))
            
            self.covar = self.covar.drop(['21003-1.0', '21003-2.0', '21003-3.0', 'assessment_center', '54-1.0', '54-2.0', '54-3.0'], axis = 1)
            self.covar = self.covar.join(center_one_hot)
            
            df_fam = pd.read_csv(fam_path)
            enc = OneHotEncoder(handle_unknown='ignore')
            enc.fit(df_fam.trait.unique().reshape(-1,1))
            batch_one_hot = enc.transform(df_fam['trait'].values.reshape(-1,1)).toarray()
            batch_num = batch_one_hot.shape[1]
            id2batch = dict(zip(df_fam.fid.values, batch_one_hot.astype(int)))
            df_batch = pd.DataFrame(np.stack(self.covar['eid'].apply(lambda x: id2batch[x] if x in id2batch else np.zeros(batch_one_hot.shape[1]).astype(int)).values)).rename(columns = dict(zip(range(batch_num), ['batch_' + str(i) for i in range(batch_num)])))
            self.covar = self.covar.join(df_batch)
            
            save_dict(covar_path, self.covar)
            print_sys('Done! Saving...')
            
        if not to_plink:
            if return_full:
                return self.covar.reset_index(drop = True)
            else:
                return self.covar[self.covar.eid.isin(self.cohort)].reset_index(drop = True)
        else:
            plink_path = os.path.join(self.cohort_data_path, 'covar_pca' + str(plink_num_pca) + '_all_real_value')
            if plink_filter:
                plink_path += '_null_removed'
            plink_path += '.txt'
            
            if not os.path.exists(plink_path):
                pca_columns = [i for i in self.covar.columns.values if (i[:3]=='pca') and int(i.split()[-1]) <= plink_num_pca]
                #center_one_hot_columns = ['center_' + str(i) for i in range(22)]
                #batch_columns = ['batch_' + str(i) for i in range(batch_num)]
                #self.covar[['eid', 'eid', 'age', 'sex'] + pca_columns + center_one_hot_columns + batch_columns].to_csv(plink_path, header=None, index=None, sep=' ')
                center = np.argmax(self.covar.loc[:, self.covar.columns.str.contains('center')].values, axis = 1)
                batch = np.argmax(self.covar.loc[:, self.covar.columns.str.contains('batch')].values, axis = 1)
                self.covar = self.covar.iloc[:, :43]
                self.covar['assessment_center'] = center
                self.covar['batch'] = batch
                if plink_filter:
                    self.covar = self.covar[self.covar.eid.isin(self.cohort)].reset_index(drop = True)
                self.covar[['eid', 'eid', 'age', 'sex', 'assessment_center', 'batch'] + pca_columns].to_csv(plink_path, header=None, index=None, sep=' ')
            self.covar_plink = pd.read_csv(plink_path, header = None, sep = ' ')
            return self.covar_plink
    
    def get_external_traits(self, trait_name, to_plink = False, to_str = True, random_seed = 42, sep_cohort = False, randomize = False, use_sample_size = False, sample_size = -1, randomize_seed = 42):
        '''
        example:
        standing heights: 50
        '''
        if trait_name in ['body_BALDING1', 'cancer_BREAST', 'disease_ALLERGY_ECZEMA_DIAGNOSED', 'disease_HYPOTHYROIDISM_SELF_REP', 'other_MORNINGPERSON', 'pigment_SUNBURN']:
            trait_type = 'binary'
        else:
            trait_type = 'continuous'

        pheno_path = os.path.join(self.cohort_data_path, str(trait_name) + '_pheno.pkl')
        if os.path.exists(pheno_path):
            print_sys('Found local copy...')
            self.pheno = load_dict(pheno_path)
        else:
            print_sys('construct phenotype from scratch...')
            
            self.pheno = pd.read_csv(os.path.join(data_path, 'full_gwas', trait_name+'_'+trait_type+'.csv'))
            self.pheno['eid'] = self.pheno.eid.astype('int')
            self.pheno = self.pheno[self.pheno['pheno'].notnull()]
            if trait_type == 'binary':
                self.pheno['pheno'] += 1
                self.pheno['pheno'] = self.pheno['pheno'].astype(int)
            save_dict(pheno_path, self.pheno)
            print_sys('Done! Saving...')
            
            
        
        # filtering to cohorts incl. with/without relatives            
        self.pheno = self.pheno[self.pheno.eid.isin(self.cohort)].reset_index(drop = True)
            
        if to_str:
            self.pheno['eid'] = self.pheno['eid'].astype('str')
        if not to_plink:
            return self.pheno
        else:
            plink_path = os.path.join(self.cohort_data_path, str(trait_name) + '_plink')
            if self.keep_relatives:
                plink_path = plink_path + '_with_relatives'
            else:
                plink_path = plink_path + '_no_relatives'
                
            if use_sample_size:
                plink_path = plink_path + '_' + str(sample_size) + '_' + str(random_seed)
            
            if sep_cohort:
                plink_path += '_sep_cohort'
             
            if randomize:
                plink_path += '_randomize' + str(randomize_seed)
            
            plink_path = plink_path + '.txt'
            
            if randomize:
                self.pheno['pheno'] = self.pheno['pheno'].sample(frac = 1, random_state = randomize_seed).values
            
            if use_sample_size:
                from sklearn.model_selection import train_test_split
                print('random_seed:', random_seed)
                pheno_shuffle = self.pheno.sample(frac = 1, random_state = random_seed)
                all_ids, y = pheno_shuffle.eid.values, pheno_shuffle['pheno'].values
                train_val_ids, test_ids, y_train_val, y_test = all_ids[:sample_size], all_ids[sample_size:], y[:sample_size], y[sample_size:]
                if sep_cohort:
                    self.pheno = self.pheno[self.pheno.eid.isin(test_ids)]
                else:
                    self.pheno = self.pheno[self.pheno.eid.isin(train_val_ids)]
                        
            if not os.path.exists(plink_path):
                print_sys('Saving...')
                self.pheno[['eid', 'eid', self.pheno.columns.values[-1]]].to_csv(plink_path, header=None, index=None, sep=' ')
            else:
                print_sys('Already existed! Loading...')
            
            self.pheno_plink = pd.read_csv(plink_path, header = None, sep = ' ')
            return self.pheno_plink
    
    
    
    
    def get_phenotype(self, field_id, aggregate = 'last_value', to_plink = False, to_str = True, normalize = 'None', frac = 1, random_seed = 42, fastgwa_match = False, icd10 = False, icd10_level = 2, sep_cohort = False, randomize = False, use_sample_size = False, sample_size = -1, randomize_seed = 42):
        '''
        example:
        standing heights: 50
        '''
        pheno_path = os.path.join(self.cohort_data_path, str(field_id) + '_pheno.pkl')
        if os.path.exists(pheno_path):
            print_sys('Found local copy...')
            self.pheno = load_dict(pheno_path)
        else:
            print_sys('construct phenotype from scratch...')
            if icd10:
                ## field_id is icd10 level
                icd10_df = self.get_icd10(to_plink = True, level = icd10_level, get_all = True)
                self.pheno = icd10_df[['FID', field_id]].rename(columns = {'FID': 'eid'})
                self.pheno['eid'] = self.pheno.eid.astype('int')
            else:
                ## from raw data field id
                self.pheno = get_fields([field_id], self.main_data_path)
            save_dict(pheno_path, self.pheno)
            print_sys('Done! Saving...')
        
        if len(self.pheno.columns.values) > 2:
            print_sys('There are multiple index for this phenotype... aggregate...')
            if aggregate == 'last_value':
                print_sys('Getting the latest measure...')
                tmp = pd.DataFrame()
                tmp['eid'] = self.pheno.loc[:, 'eid']
                tmp[str(field_id)] = get_row_last_values(self.pheno.iloc[:, 1:])
                self.pheno = tmp
                print_sys('There are ' + str(len(self.pheno[self.pheno[str(field_id)].isnull()])) + ' samples with NaN values. Removing them ...')
                self.pheno = self.pheno[self.pheno[str(field_id)].notnull()]
                
        if fastgwa_match:
            # get the number of without relatives:
            if not self.keep_relatives:
                raise ValueError('If you turned fastgwa_match = True, then keep_relatives = True!')
            self.rel_ratio = len(self.pheno[self.pheno.eid.isin(self.no_rel_eid)])/len(self.pheno[self.pheno.eid.isin(self.cohort)])
        
        # filtering to cohorts incl. with/without relatives            
        self.pheno = self.pheno[self.pheno.eid.isin(self.cohort)].reset_index(drop = True)

        if normalize != 'None':
            y = self.pheno[str(field_id)].values
            if normalize == 'log':
                y = np.log(y)
            elif normalize == 'std':
                y = (y - np.mean(y))/np.std(y)
            elif normalize == 'quantile_normalization':
                from sklearn.preprocessing import quantile_transform
                y = quantile_transform(y.reshape(-1,1), output_distribution = 'normal', random_state = 42).reshape(-1)
            self.pheno[str(field_id)] = y 
            
        if to_str:
            self.pheno['eid'] = self.pheno['eid'].astype('str')
        if not to_plink:
            return self.pheno
        else:
            plink_path = os.path.join(self.cohort_data_path, str(field_id) + '_plink')
            if self.keep_relatives:
                plink_path = plink_path + '_with_relatives'
            else:
                plink_path = plink_path + '_no_relatives'
                
            if normalize != 'None':
                plink_path = plink_path + '_' + str(normalize)
            if use_sample_size:
                plink_path = plink_path + '_' + str(sample_size) + '_' + str(random_seed)
            else:
                if frac != 1:
                    plink_path = plink_path + '_' + str(frac) + '_' + str(random_seed)
                
            if fastgwa_match:
                plink_path += '_match'
                
            if sep_cohort:
                plink_path += '_sep_cohort'
             
            if randomize:
                plink_path += '_randomize' + str(randomize_seed)
                
            
            plink_path = plink_path + '.txt'
            
            if randomize:
                self.pheno[str(field_id)] = self.pheno[str(field_id)].sample(frac = 1, random_state = randomize_seed).values
            
            
            if use_sample_size:
                from sklearn.model_selection import train_test_split
                
                if icd10:
                    df_cases = self.pheno[self.pheno[str(field_id)] == 2]
                    df_cases_shuffle = df_cases.sample(frac = 1, random_state = random_seed)
                    all_ids, y = df_cases_shuffle.eid.values, df_cases_shuffle[str(field_id)].values
                    train_val_ids, test_ids, y_train_val, y_test = all_ids[:sample_size], all_ids[sample_size:], y[:sample_size], y[sample_size:]
                    train_val_ids = np.concatenate((train_val_ids, self.pheno[self.pheno[str(field_id)] == 1].eid.values))
                    self.pheno = self.pheno[self.pheno.eid.isin(train_val_ids)]
                    if sep_cohort:
                        raise NotImplementedError
                else:
                    print('random_seed', random_seed)
                    pheno_shuffle = self.pheno.sample(frac = 1, random_state = random_seed)
                    all_ids, y = pheno_shuffle.eid.values, pheno_shuffle[str(field_id)].values
                    train_val_ids, test_ids, y_train_val, y_test = all_ids[:sample_size], all_ids[sample_size:], y[:sample_size], y[sample_size:]
                    if fastgwa_match:
                        raise ValueError('Not used anymore...')
                    if sep_cohort:
                        self.pheno = self.pheno[self.pheno.eid.isin(test_ids)]
                    else:
                        self.pheno = self.pheno[self.pheno.eid.isin(train_val_ids)]
            else:
                if frac!=1:
                    from sklearn.model_selection import train_test_split
                    all_ids, y = self.pheno.eid.values, self.pheno[str(field_id)].values                
                    train_val_ids, test_ids, y_train_val, y_test = train_test_split(all_ids, y, test_size=frac, random_state=random_seed)
                    if fastgwa_match:
                        train_val_ids, test_ids, y_train_val, y_test = train_test_split(train_val_ids, y_train_val, test_size=1-self.rel_ratio, random_state=42)
                    if sep_cohort:
                        self.pheno = self.pheno[self.pheno.eid.isin(test_ids)]
                    else:
                        self.pheno = self.pheno[self.pheno.eid.isin(train_val_ids)]
                        
            if not os.path.exists(plink_path):
                self.pheno[['eid', 'eid', self.pheno.columns.values[-1]]].to_csv(plink_path, header=None, index=None, sep=' ')
            else:
                print_sys('Already existed! Loading...')
            
            self.pheno_plink = pd.read_csv(plink_path, header = None, sep = ' ')
            return self.pheno_plink
            
        
    def get_icd10(self, to_plink = False, level = 2, get_all = False):
        '''
        icd10: 41270
        '''
        pheno_path = os.path.join(self.cohort_data_path, 'icd10.pkl')
        level_str = 'level' + str(level)
        if os.path.exists(pheno_path):
            print_sys('Found local copy...')
            self.icd10 = load_dict(pheno_path)
        else:
            print_sys('construct from scratch...')
            icd10_raw_concat = get_fields([41270], self.main_data_path)
            icd10_columns = icd10_raw_concat.columns.values[1:]
            icd10_tuple = icd10_raw_concat.apply(lambda x: (x.eid, x[icd10_columns][x[icd10_columns].notnull()].values), axis = 1)
            icd10 = pd.DataFrame(list(icd10_tuple.values)).rename(columns = {0: 'eid', 1: 'level3'})
            icd10['level2'] = icd10['level3'].apply(lambda x: np.unique([i[:3] for i in x]))
            save_dict(pheno_path, icd10)
            print_sys('Done! Saving...')
            self.icd10 = icd10
        if get_all:
            self.pheno = self.icd10.reset_index(drop = True)
        else:
            self.pheno = self.icd10[self.icd10.eid.isin(self.cohort)].reset_index(drop = True)
        if not to_plink:
            return self.pheno
        else:            
            if self.keep_relatives or get_all:
                plink_path = os.path.join(self.cohort_data_path, 'icd10_plink_with_relatives_' + level_str + '.txt')
            else:
                plink_path = os.path.join(self.cohort_data_path, 'icd10_plink_no_relatives_' + level_str + '.txt')
            
            if os.path.exists(plink_path):
                print_sys("Found local copy...")
                self.icd10_plink = pd.read_csv(plink_path, sep=' ')
            else:
                print_sys('transforming to plink files... takes around 1 min...')
                unique_icd10 = np.unique([item for sublist in self.pheno[level_str].values for item in sublist])
                icd10_2_idx = dict(zip(unique_icd10, range(len(unique_icd10))))
                idx_2_icd10 = dict(zip(range(len(unique_icd10)), unique_icd10))

                self.pheno[level_str + '_idx'] = self.pheno[level_str].apply(lambda x: [icd10_2_idx[i] for i in x])

                tmp = np.zeros((len(self.pheno), len(unique_icd10)), dtype=np.int8)
                for idx, i in enumerate(self.pheno[level_str + '_idx'].values):
                    tmp[idx, i] = 1

                icd10_plink = pd.DataFrame(tmp).rename(columns = idx_2_icd10)
                icd102sample_size = dict(icd10_plink.sum(axis = 0))
                icd_100 = [i for i,j in icd102sample_size.items() if j > 100]
                icd10_plink = icd10_plink + 1
                icd10_plink['IID'] = self.pheno.eid.values
                icd10_plink['FID'] = self.pheno.eid.values
                icd10_plink = icd10_plink.loc[:, ['FID', 'IID'] + icd_100]
                print_sys('Only using ICD10 codes with at least 100 cases...')
                print_sys('There are ' + str(len(icd_100)) + ' ICD10 codes with at least 100 cases.')
                icd10_plink.to_csv(plink_path, index=None, sep=' ')
                self.icd10_plink = icd10_plink
                
            return self.icd10_plink