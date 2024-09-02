from kgwas import KGWAS, KGWAS_Data
data = KGWAS_Data()
data.load_kg(random_emb = True)

#data.load_external_gwas(PATH)
data.load_gwas_subsample(pheno = 'body_BALDING1', sample_size = 5000, seed = 1)
data.process_gwas_file()
data.prepare_split()

run = KGWAS(data,
            weight_bias_track = False,
            device = 'cuda:6',
            proj_name = 'KGWAS',
            exp_name = 'KGWAS_Test',
            seed = 1)

run.initialize_model()
run.train(epoch = 0)