bfile=/dfs/project/datasets/20220524-ukbiobank/data/genetics/ukb_all
post_qc_snp=/dfs/project/datasets/20220524-ukbiobank/data/genetics/ukb_all_out-basic-qc.snplist
post_qc_fam=/dfs/project/datasets/20220524-ukbiobank/data/genetics/ukb_all_out-basic-qc.fam
result=/dfs/project/datasets/20220524-ukbiobank/data/result/
covar=/dfs/project/datasets/20220524-ukbiobank/data/cohort/covar_pca15_all_real_value.txt
pheno=$1

./plink/plink2   --bfile ${bfile} \
        --keep ${post_qc_fam} \
        --extract ${post_qc_snp} \
        --glm hide-covar \
        --pheno ${pheno} \
        --covar ${covar} \
        --out "${result}$2" \
        --memory 50000 \
        --threads 20