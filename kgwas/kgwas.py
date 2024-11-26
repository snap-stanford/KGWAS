from copy import deepcopy
from tqdm import tqdm
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import pandas as pd
import numpy as np
import pickle
import subprocess

import torch
import torch.nn.functional as F
import torch.optim as optim

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
from torch_geometric.loader import NeighborLoader
from .utils import print_sys, compute_metrics, save_dict, \
                        load_dict, load_pretrained, save_model, \
                        evaluate_minibatch_clean, process_data, \
                        get_network_weight, generate_viz
from .eval_utils import storey_ribshirani_integrate, get_clumps_gold_label, get_meta_clumps, \
                        get_mega_clump_query, get_curve, find_closest_x
from .model import HeteroGNN

class KGWAS:
    def __init__(self,
                data,
                weight_bias_track = False,
                device = 'cuda',
                proj_name = 'KGWAS',
                exp_name = 'KGWAS',
                seed = 42):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        self.seed = seed
        torch.backends.cudnn.enabled = False
        use_cuda = torch.cuda.is_available()
        self.device = device if use_cuda else "cpu"

        self.data = data
        self.data_path = data.data_path
        if weight_bias_track:
            import wandb
            wandb.init(project=proj_name, name=exp_name)  
            self.wandb = wandb
        else:
            self.wandb = False
        self.exp_name = exp_name
        
        
    def initialize_model(self, gnn_num_layers = 2, gnn_hidden_dim = 128, gnn_backbone = 'GAT', gnn_aggr = 'sum', gat_num_head = 1, no_relu = False):

        self.config = {
            'gnn_num_layers': gnn_num_layers,
            'gnn_hidden_dim': gnn_hidden_dim,
            'gnn_backbone': gnn_backbone,
            'gnn_aggr': gnn_aggr,
            'gat_num_head': gat_num_head
        }

        self.gnn_num_layers = gnn_num_layers
        self.model = HeteroGNN(self.data.data, gnn_hidden_dim, 1, 
                               gnn_num_layers, gnn_backbone, gnn_aggr,
                               self.data.snp_init_dim_size,
                               self.data.gene_init_dim_size,
                               self.data.go_init_dim_size,
                               gat_num_head,
                               no_relu = no_relu,
                              ).to(self.device)
    
    
    def load_pretrained(self, path):
        with open(os.path.join(path, 'config.pkl'), 'rb') as f:
            config = pickle.load(f)
            
        self.initialize_model(**config)
        self.config = config
        
        self.model = load_pretrained(path, self.model)
        self.best_model = self.model
        self.kgwas_res = pd.read_csv(os.path.join(path, 'pred.csv'), sep = None, engine = 'python')
        self.save_name = path.split('/')[-1]

    def train(self, batch_size = 512, num_workers = 6, lr = 1e-4, 
                    weight_decay = 5e-4, epoch = 10, save_best_model = True, 
                    save_name = None, data_to_cuda = False):
        total_epoch = epoch
        if save_name is None:
            save_name = self.exp_name
        self.save_name = save_name
        print_sys('Creating data loader...')
        kwargs = {'batch_size': batch_size, 'num_workers': num_workers, 'drop_last': True}
        eval_kwargs = {'batch_size': 512, 'num_workers': num_workers, 'drop_last': False}
        
        if data_to_cuda:
            self.data.data = self.data.data.to(self.device)
        
        self.train_loader = NeighborLoader(self.data.data, num_neighbors=[-1] * self.gnn_num_layers, 
                                      sampler = None,
                                      input_nodes=self.data.train_input_nodes, **kwargs)
        self.val_loader = NeighborLoader(self.data.data, num_neighbors=[-1] * self.gnn_num_layers,
                                    input_nodes=self.data.val_input_nodes, **kwargs)
        self.test_loader = NeighborLoader(self.data.data, num_neighbors=[-1] * self.gnn_num_layers,
                                    input_nodes=self.data.test_input_nodes, **eval_kwargs)
        
        X_infer = self.data.lr_uni.ID.values
        #print_sys('# of to-infer SNPs: ' + str(len(X_infer)))
        infer_idx = np.array([self.data.id2idx['SNP'][i] for i in X_infer])
        infer_input_nodes = ('SNP', infer_idx)

        self.infer_loader = NeighborLoader(self.data.data, num_neighbors=[-1] * self.gnn_num_layers,
                                    input_nodes=infer_input_nodes, **eval_kwargs)
        
        ## model training
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay = weight_decay)

        loss_fct = F.mse_loss
        earlystop_validation_metric = 'pearsonr'
        binary_output = False
        earlystop_direction = 'ascend'
        min_val = -1000

        self.best_model = deepcopy(self.model).to(self.device)
        print_sys('Start Training...')
        for epoch in range(total_epoch):
            self.model.train()

            for step, batch in enumerate(tqdm(self.train_loader, desc=f"Training Progress Epoch {epoch+1}/{total_epoch}", total=len(self.train_loader))):
                optimizer.zero_grad()
                if data_to_cuda:
                    pass
                    #batch = batch.to(self.device, 'edge_index')
                else:
                    batch = batch.to(self.device)
                bs_batch = batch['SNP'].batch_size

                out = self.model(batch.x_dict, batch.edge_index_dict, bs_batch)
                pred = out.reshape(-1)

                y_batch = batch['SNP'].y[:bs_batch]
                rs_id = [self.data.idx2id['SNP'][i.item()] for i in batch['SNP']['n_id'][:bs_batch]]
                ld_weight = torch.tensor([self.data.rs_id_to_ldsc_weight[i] for i in rs_id]).to(self.device)

                loss = torch.mean(ld_weight * (pred - y_batch)**2)

                if self.wandb:
                    self.wandb.log({'training_loss': loss.item()})

                loss.backward()
                optimizer.step()

                if (step % 500 == 0) and (step >= 500):
                    log = "Epoch {} Step {} Train Loss: {:.4f}" 
                    print_sys(log.format(epoch + 1, step + 1, loss.item()))

            val_res = evaluate_minibatch_clean(self.val_loader, self.model, self.device)
            val_metrics = compute_metrics(val_res, binary_output, -1, -1, loss_fct)


            log = "Epoch {}: Validation MSE: {:.4f} " \
                      "Validation Pearson: {:.4f}. "
            print_sys(log.format(epoch + 1, val_metrics['mse'], 
                         val_metrics['pearsonr']))

            if self.wandb:
                for i,j in val_metrics.items():
                    self.wandb.log({'val_' + i: j})
                
            if val_metrics[earlystop_validation_metric] > min_val:
                min_val = val_metrics[earlystop_validation_metric]
                self.best_model = deepcopy(self.model)
                best_epoch = epoch


        if save_best_model:
            save_model_path = self.data_path + '/model/'
            print_sys('Saving models to ' + os.path.join(save_model_path, save_name))
            save_model(self.best_model, self.config, os.path.join(save_model_path, save_name))


        test_res = evaluate_minibatch_clean(self.test_loader, self.best_model, self.device)
        test_metric = compute_metrics(test_res, binary_output, -1, -1, loss_fct)
        if self.wandb:
            for i,j in test_metric.items():
                self.wandb.log({'test_' + i: j})    


        infer_res = evaluate_minibatch_clean(self.infer_loader, self.best_model, self.device)

        self.data.lr_uni['pred'] = infer_res['pred']
        lr_uni_to_save = deepcopy(self.data.lr_uni)

        self.data.lr_uni['abs_pred'] = np.abs(self.data.lr_uni['pred'])

        self.data.lr_uni['SR_P_val'] = storey_ribshirani_integrate(self.data.lr_uni, column = 'abs_pred', num_bins = 500)
        self.data.lr_uni['SR'] = -(np.log10(self.data.lr_uni['SR_P_val'].astype(float).values))
        lr_uni_to_save['P_weighted'] = self.data.lr_uni['SR_P_val']

        ## calibration
        scale_factor = find_closest_x(lr_uni_to_save)
        lr_uni_to_save['KGWAS_P'] = scale_factor * lr_uni_to_save['P_weighted']
        lr_uni_to_save['KGWAS_P'] = lr_uni_to_save['KGWAS_P'].clip(lower=0, upper=1)

        if not os.path.exists(self.data_path + '/model_pred/'):
            os.makedirs(self.data_path + '/model_pred/')
            os.makedirs(self.data_path + '/model_pred/new_experiments/')
        lr_uni_to_save.to_csv(self.data_path + '/model_pred/new_experiments/' + save_name + '_pred.csv', index = False, sep = '\t')
        print('KGWAS prediction and p-values saved to ' + self.data_path + '/model_pred/new_experiments/' + save_name + '_pred.csv')
        if save_best_model:
            lr_uni_to_save.to_csv(self.data_path + '/model/' + save_name + '/pred.csv', index = False, sep = '\t')
        self.kgwas_res = lr_uni_to_save

    def run_magma(self, path_to_magma, bfile):
        if 'N' in self.kgwas_res.columns:
            n_value = self.kgwas_res['N'].values[0]
        else:
            n_value = input("Please provide the sample size for the GWAS analysis.")
        
        url = "https://dataverse.harvard.edu/api/access/datafile/10731670"
        annot_file_path = os.path.join(self.data_path, 'gene_annotation.genes.annot')

        # Check if the example file is already downloaded
        if not os.path.exists(annot_file_path):
            print('Annotation file not found locally. Downloading...')
            self.data._download_with_progress(url, annot_file_path)
            print('Annotation file downloaded successfully.')
        else:
            print('Annotation file already exists locally.')

        gene_annot = annot_file_path

        magma_path = self.data_path + '/model_pred/new_experiments/' + self.save_name + '_magma_format.csv'
        self.kgwas_res[['ID', 'KGWAS_P']].rename(columns = {'ID': 'SNP', 'KGWAS_P': 'P'}).to_csv(magma_path, index = False, sep = '\t')

        # Construct the MAGMA command
        command = [
            path_to_magma,
            "--bfile", bfile,
            "--gene-annot", gene_annot,
            "--pval", magma_path, f"N={n_value}",
            "--out", self.data_path + '/model_pred/new_experiments/' + self.save_name + '_magma_out'
        ]
        
        try:
            # Run the command with real-time output
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            print("Running MAGMA...")

            # Stream stdout line by line
            for line in process.stdout:
                print(line, end="")  # Print each line as it's received

            # Wait for the process to complete and capture stderr
            stderr = process.communicate()[1]

            if process.returncode == 0:
                print("MAGMA command executed successfully.")
            else:
                print("MAGMA encountered an error.")
                print("Error message:", stderr)
        except FileNotFoundError:
            print("MAGMA executable not found. Ensure it is in the specified path.")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")


    def get_disease_critical_network(self, variant_threshold = 5e-8, 
                magma_path = None, magma_threshold = 0.05, program_threshold = 0.05,
                K_neighbors = 3, num_cpus = 1):
        df_network_weight = get_network_weight(self, self.data)
        df_variant_interpretation, disease_critical_network = generate_viz(self, df_network_weight, self.data_path, variant_threshold, magma_path, magma_threshold, program_threshold, K_neighbors, num_cpus)
        return df_network_weight, df_variant_interpretation, disease_critical_network