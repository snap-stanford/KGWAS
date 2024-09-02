bfile=/dfs/project/datasets/20220524-ukbiobank/data/genetics/ukb_all
post_qc_snp=/dfs/project/datasets/20220524-ukbiobank/data/genetics/ukb_all_out-basic-qc.snplist
post_qc_fam=/dfs/project/datasets/20220524-ukbiobank/data/genetics/ukb_all_out-basic-qc.fam
result=/dfs/project/datasets/20220524-ukbiobank/data/result/
covar=/dfs/project/datasets/20220524-ukbiobank/data/cohort/covar_pca15_all_real_value_null_removed.txt
pheno=$1
grm_file=/dfs/project/datasets/20220524-ukbiobank/data/gcta_result/gcta_rel

./gcta/gcta64 --bfile $bfile --grm-sparse $grm_file --fastGWA-mlm --qcovar $covar --pheno $pheno --thread-num 20 --out "${result}$2" --extract ${post_qc_snp}