Genome In A Bottle truth set for the GM24385 sample can be downloaded from the NCBI repository with:


	for ext in .bed .bed.gz .bed.gz.tbi .vcf.gz .vcf.gz.tbi; do
		wget https://ftp-trace.ncbi.nlm.nih.gov/ReferenceSamples/giab/release/AshkenazimTrio/HG002_NA24385_son/NISTv4.1/GRCh38/HG002_GRCh38_1_22_v4.1_draft_benchmark${ext}
	done


code can load truth set as .vcf or .vcf.gz
