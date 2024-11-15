from kgwas import KGWAS, KGWAS_Data
data = KGWAS_Data(data_path = '/dfs/project/datasets/20220524-ukbiobank/data/kgwas_core_data/')
data.load_kg()

data.load_external_gwas('/dfs/project/datasets/20220524-ukbiobank/data/t2d_gwas_cleaned.csv')
#data.load_gwas_subsample(pheno = 'body_BALDING1', sample_size = 5000, seed = 1)
data.process_gwas_file()
data.prepare_split()

run = KGWAS(data,
            weight_bias_track = False,
            device = 'cuda:7',
            proj_name = 'KGWAS',
            exp_name = 'KGWAS_t2d',
            seed = 1)

run.initialize_model()
run.train(epoch = 1)