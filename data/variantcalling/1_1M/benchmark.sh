INPUT_DIR="/home/ruchirgarg5/content/data"
OUTPUT_DIR="/home/ruchirgarg5/content/clair3_ont_demo"
BASELINE_VCF_FILE_PATH="HG003_GRCh38_chr20_v4.2.1_benchmark.vcf.gz"
BASELINE_BED_FILE_PATH="HG003_GRCh38_chr20_v4.2.1_benchmark_noinconsistent.bed"
#OUTPUT_VCF_FILE_PATH="pre_processed_2.0.realigned_genotypes_preprocessed.vcf.gz"
OUTPUT_VCF_FILE_PATH="longshot_output.vcf.gz"
#OUTPUT_VCF_FILE_PATH="merge_output.vcf.gz"
REF="GRCh38_no_alt_chr20.fa"
CONTIGS="chr20"
START_POS=100000
END_POS=300000
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
