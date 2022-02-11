INPUT_DIR="/home/ruchirgarg5/content/HAPCUT_preprocessing/data/variantcalling/1_1M"
OUTPUT_DIR="/home/ruchirgarg5/content/HAPCUT_preprocessing/data/variantcalling/1_1M"
BASELINE_VCF_FILE_PATH="chr20.GIAB_highconfidencecalls_1_1M.vcf.gz"
BASELINE_BED_FILE_PATH="chr20.GIAB_highconfidenceregions_1_1M.bed"
#OUTPUT_VCF_FILE_PATH="pre_processed_2.0.realigned_genotypes_preprocessed.vcf.gz"
OUTPUT_VCF_FILE_PATH="2.0.realigned_genotypes_1_1M.vcf"
#OUTPUT_VCF_FILE_PATH="merge_output.vcf.gz"
REF="GRCh38_no_alt_chr20.fa"
CONTIGS="chr20"
START_POS=1
END_POS=1000000
THREADS=4
#source activate happy-env
#conda install -c bioconda rtg-tools -y 
hap.py \
    ${INPUT_DIR}/${BASELINE_VCF_FILE_PATH} \
    ${OUTPUT_DIR}/${OUTPUT_VCF_FILE_PATH} \
    -f "${INPUT_DIR}/${BASELINE_BED_FILE_PATH}" \
    -r "${INPUT_DIR}/${REF}" \
    -o "${OUTPUT_DIR}/happy" \
    -l ${CONTIGS}:${START_POS}-${END_POS} \
    --engine=vcfeval \
    --threads="${THREADS}" \
    --pass-only
