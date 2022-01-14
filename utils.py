import numpy as np
import pandas as pd
import allel


def read_fragments(fragments_path):
	''' 
	Reads fragments file and returns read allele and quality data as 
		numpy matrices. Sites in matrices where there is no data are np.nan

	Args:	path to a fragments.txt file

	Returns 3 numpy arrays:
		array of fragment_ids corresponding to rows
		matrix of allele values {0,1,np.nan} where rows correspond to samples 
			and cols to sites
		matrix of Phred quality scores or np.nan where no read
	'''
	frag_ids, row_col_pairs, allele_vals, qual_scores = read_fragments_arrays(
		fragments_path
	)

	allele_mat = np.full(row_col_pairs.max(axis=0)+1, np.nan)
	allele_mat[row_col_pairs[:,0], row_col_pairs[:,1]] = allele_vals

	qual_mat = np.full(row_col_pairs.max(axis=0)+1, np.nan)
	qual_mat[row_col_pairs[:,0], row_col_pairs[:,1]] = qual_scores

	return frag_ids, allele_mat, qual_mat


def read_fragments_arrays(fragments_path):
	''' 
	Reads fragments file and returns data as numpy arrays of corresponding
		indices and data

	Args:	path to a fragments.txt file

	Returns 4 numpy arrays:
		fragment_ids corresponding to rows
		row col indices for the data in allele_vals and qual_scores
		allele values {0,1} at row,col locs
		Phred quality scores at row,col locs
	'''
	with open(fragments_path) as f:

		frag_ids = []		# fragment ids from col 2
		row_col_pairs = []	# row,col indices coresponding to allele values
		allele_vals = []	# 	and quality scores as matrices
		qual_scores = []

		row_ind = 0

		for line in f:
			line_data = line.strip().split()
			frag_ids.append(line_data[1])

			# get sample's row,col pairs and allele vals
			block_data = line_data[2:-1]

			for i in range(0, len(block_data), 2):
				block_start_ind = int(block_data[i])

				for start_offset in range(len(block_data[i + 1])):
					row_col_pairs.append(
						(row_ind, block_start_ind + start_offset)
					)
					allele_vals.append(block_data[i + 1][start_offset])

			# add quality scores
			qual_str = line_data[-1]
			for char in qual_str:
				qual_scores.append(ord(char) - 33)

			row_ind += 1

		# set indices to start at 0
		row_col_pairs = np.array(row_col_pairs)
		row_col_pairs -= row_col_pairs.min(axis=0, keepdims=True)

		return (
			np.array(frag_ids),
			row_col_pairs,
			np.array(allele_vals).astype(int),
			np.array(qual_scores)
		)


def get_bed_mask(bed_path, ls_callset_pos, chrom='chr20'):
	''' 
	Reads areas in GIAB from .bed and uses to mask longshot callset positions
	'''
	with open(bed_path) as f:
		starts = []
		ends = []

		for line in f:
			line_data = line.strip().split()

			if line_data[0] == chrom:
				starts.append(line_data[1])
				ends.append(line_data[2])

	starts = np.array(starts).astype(int)
	ends = np.array(ends).astype(int)

	in_bed_range = []

	for ls_pos in ls_callset_pos:
		in_bed_range.append(
			np.any((starts <= ls_pos) & (ends > ls_pos))
		)

	return np.array(in_bed_range)


def get_true_variants(longshot_vcf_path, ground_truth_vcf_path, giab_bed_path,
						return_vcfs=False):
	''' 
	Finds true/false variants in fragments file using GIAB ground truth vcf

	Args: 
		longshot_vcf_path: path to "2.0.realigned_genotypes.vcf" for longshot run
			that produced the fragments.txt file being used
		ground_truth_vcf_path: path to GIAB ground truth vcf
		giab_bed_path: giab ground truth corresponding .bed
		return_vcfs: if the longshot and ground_truth vcfs should be returned.
			Will be returned as callsets
	
	Returns:
		array length of number of cols of fragments matrix where each is
			labeled True/False wrt being real variants
		site_mask to use to filter columns of fragments matrix
		if return_vcfs, returns (true_variants, longshot_vcf, ground_truth_vcf)
	'''
	# load vcf from longshot run
	ls_callset = allel.read_vcf(longshot_vcf_path)

	# load ground truth vcf
	callset = allel.read_vcf(ground_truth_vcf_path)

	# find true variants
	chr20_mask = callset['variants/CHROM'] == ls_callset['variants/CHROM'][0]
	callset = mask_callset(callset, chr20_mask)

	in_truth = np.in1d(ls_callset['variants/POS'], callset['variants/POS'])

	# mask out regions not in .bed
	in_bed_mask = get_bed_mask(giab_bed_path, ls_callset['variants/POS'])

	# find where longshot predicts heterozygous
	ls_01 = np.all(np.equal(ls_callset['calldata/GT'], [0,1]), axis=2).T[0]
	ls_10 = np.all(np.equal(ls_callset['calldata/GT'], [1,0]), axis=2).T[0]
	ls_hetero = ls_01 | ls_10

	site_mask = in_bed_mask & ls_hetero

	if return_vcfs:
		return in_truth.astype(int)[site_mask], site_mask, ls_callset, callset
	else:
		return in_truth.astype(int)[site_mask], site_mask


def mask_callset(callset, mask):
	for key in list(callset):
		if key == 'samples':
			continue
		callset[key] = callset[key][mask]

	return callset


def matrix_sparsity_info(allele_mat, print_info=False):
	''' 
	Get info about sparsity of allele/quality/incorrect read matrix by
	rows and cols
	'''

	sample_reads = np.count_nonzero(~np.isnan(allele_mat), axis=1)
	site_reads = np.count_nonzero(~np.isnan(allele_mat), axis=0)

	if print_info:
		nonzero_sites = np.count_nonzero(~np.isnan(allele_mat))

		print("num elements not missing:\t{}".format(nonzero_sites))
		print("percent matrix not missing:\t{:.3f}".format(
			nonzero_sites / allele_mat.size
		))

		print("num fragments:\t{}".format(allele_mat.shape[0]))
		print("num sites:\t{}".format(allele_mat.shape[1]))

		print("\nfragments:")
		val = np.mean(sample_reads)
		print("\tmean reads:\t{:.1f}\t{:.3f}".format(val, val / allele_mat.shape[1]))
		val = np.median(sample_reads)
		print("\tmedian reads:\t{}\t{:.3f}".format(val, val / allele_mat.shape[1]))
		val = np.max(sample_reads)
		print("\tmax reads:\t{}\t{:.3f}".format(val, val / allele_mat.shape[1]))
		val = np.min(sample_reads)
		print("\tmin reads:\t{}\t{:.3f}".format(val, val / allele_mat.shape[1]))
		
		print("\nsites:")
		val = np.mean(site_reads)
		print("\tmean reads:\t{:.1f}\t{:.3f}".format(val, val / allele_mat.shape[0]))
		val = np.median(site_reads)
		print("\tmedian reads:\t{}\t{:.3f}".format(val, val / allele_mat.shape[0]))
		val = np.max(site_reads)
		print("\tmax reads:\t{}\t{:.3f}".format(val, val / allele_mat.shape[0]))
		val = np.min(site_reads)
		print("\tmin reads:\t{}\t{:.3f}".format(val, val / allele_mat.shape[0]))

	return sample_reads, site_reads


def save_preprocessed(path, fragments, qualities, variant_labels):
	np.savez(
		path, 
		fragments=fragments,
		qualities=qualities,
		variant_labels=variant_labels)


def load_preprocessed(path):
	''' Returns (fragments, qualities, variant_labels) from .npz '''
	data = np.load(path)
	return data['fragments'], data['qualities'], data['variant_labels']


def encode_genotype(gt):
	'''
	Encode genotype from giab vcf as int:
		0 if 0/0
		1 if 0/1
		2 if 1/1
		-1 if anything else
	'''
	if gt[0] == 0 and gt[1] == 0:
		return 0
	elif gt[0] == 1 and gt[1] == 1:
		return 2
	elif (gt[0] == 0 and gt[1] == 1) or (gt[0] == 1 and gt[1] == 0):
		return 1
	else:
		return -1


def load_data(fragments_path, longshot_vcf_path, ground_truth_vcf_path,
				giab_bed_path, save_path=None, return_full_frag_mats=False):
	''' 
	Load data from fragments.txt & 2.0.realigned_genotypes.vcf files
	generated by Longshot and the GIAB ground truth vcf and bed files

	Args:
		fragments_path: path to fragments.txt
		longshot_vcf_path: path to 2.0.realigned_genotypes.vcf
		ground_truth_vcf_path: path to something like 
			HG002_GRCh38_1_22_v4.1_draft_benchmark.vcf
		giab_bed_path: path to something like HG002_..._benchmark.bed
		save_path: location to save a file in a directory that already 
			exists
		return_full_frag_mats: if True, returns unmasked fragments and
			qualities matrices in addition to masked versions usually 
			returned

	Returns:
		(site_data, fragments, qualities) or if return_full_frag_mats:
		(site_data, fragments, qualities, fragments_unmasked, 
			qualities_unmasked)

		site_data: DataFrame with info for all sites HMM predicts are
			heterozygous
		fragments: fragments matrix as returned by read_fragments masked
			to sites HMM predicts are heterozygous
		qualities: qualities matrix as returned by read_fragments masked
			to sites HMM predicts are heterozygous
		fragments_unmasked: fragments matrix as returned by read_fragments
		qualities_unmasked: qualities matrix as returned by read_fragments		
	'''
	# load vcf from longshot run
	ls_callset = allel.read_vcf(longshot_vcf_path)

	# find where longshot hmm predicts heterozygous
	ls_01 = np.all(np.equal(ls_callset['calldata/GT'], [0,1]), axis=2).T[0]
	ls_10 = np.all(np.equal(ls_callset['calldata/GT'], [1,0]), axis=2).T[0]
	ls_hetero = ls_01 | ls_10
	hetero_site_inds = np.where(ls_hetero)[0]

	ls_callset = mask_callset(ls_callset, ls_hetero)

	# binary if site in .bed region (and therefore there will be ground truth)
	in_bed_mask = get_bed_mask(giab_bed_path, ls_callset['variants/POS']).astype(int)

	# load ground truth vcf
	giab_callset = allel.read_vcf(ground_truth_vcf_path)

	# mask GIAB vcf to correct chromesome and make DataFrame rep
	chr_mask = giab_callset['variants/CHROM'] == ls_callset['variants/CHROM'][0]
	giab_callset = mask_callset(giab_callset, chr_mask)

	giab_df = pd.DataFrame(
		np.vstack((
			giab_callset['variants/CHROM'],
			giab_callset['variants/POS'],
			giab_callset['variants/REF'],
			[','.join(giab_callset['variants/ALT'][i]) for i in range(
					giab_callset['variants/ALT'].shape[0])],
			[encode_genotype(gt[0]) for gt in giab_callset['calldata/GT']] 
		)).T,
		columns=['chrom', 'pos', 'ref', 'alt', 'genotype']
	)

	df = pd.DataFrame(
		np.array((
			hetero_site_inds, 
			in_bed_mask, 
			ls_callset['variants/CHROM'],
			ls_callset['variants/POS'])).T,
		columns=['site_ind', 'in_bed', 'chrom', 'pos']
	)

	df = pd.merge(df, giab_df, how='left', on=['chrom', 'pos'])

	# fill for genotype=0 sites in bed but not var sites
	df.loc[df.in_bed == 1, 'genotype'] = df.loc[df.in_bed == 1]['genotype'].fillna(0)

	# save
	if save_path:
		df.to_csv(save_path, na_rep='', sep='\t', index=False)

	# load read fragments and their qualities
	_, fragments, qualities = read_fragments(fragments_path)

	if return_full_frag_mats:
		return (df, fragments[:, df.site_ind.astype(int)], 
				qualities[:, df.site_ind.astype(int)], fragments, qualities)
	else:
		return (df, fragments[:, df.site_ind.astype(int)], 
				qualities[:, df.site_ind.astype(int)])


def load_full_data(fragments_path, longshot_vcf_path, ground_truth_vcf_path,
					giab_bed_path, save_path=None):
	''' 
	Load data from fragments.txt & 2.0.realigned_genotypes.vcf files
	generated by Longshot and the GIAB ground truth vcf and bed files.
	Does not mask data to Longshot predicted heterozygous sites as 
	load_data does.

	Args:
		fragments_path: path to fragments.txt
		longshot_vcf_path: path to 2.0.realigned_genotypes.vcf
		ground_truth_vcf_path: path to something like 
			HG002_GRCh38_1_22_v4.1_draft_benchmark.vcf
		giab_bed_path: path to something like HG002_..._benchmark.bed
		save_path: location to save a file in a directory that already 
			exists

	Returns:
		site_data, fragments, qualities

		site_data: DataFrame with info for all sites in fragments matrix
		fragments: fragments matrix as returned by read_fragments
		qualities: qualities matrix as returned by read_fragments	
	'''
	# load vcf from longshot run
	ls_callset = allel.read_vcf(longshot_vcf_path)

	# longshot hmm predictions
	ls_01 = np.all(np.equal(ls_callset['calldata/GT'], [0,1]), axis=2).T[0]
	ls_10 = np.all(np.equal(ls_callset['calldata/GT'], [1,0]), axis=2).T[0]
	ls_hetero = ls_01 | ls_10
	ls_11 = np.all(np.equal(ls_callset['calldata/GT'], [1,1]), axis=2).T[0]

	ls_gt = np.zeros(ls_callset['variants/POS'].shape[0]).astype(int)
	ls_gt[ls_hetero] = 1
	ls_gt[ls_11] = 2

	# binary if site in .bed region (and therefore there will be ground truth)
	in_bed_mask = get_bed_mask(giab_bed_path, ls_callset['variants/POS']).astype(int)

	# load ground truth vcf
	giab_callset = allel.read_vcf(ground_truth_vcf_path)

	# mask GIAB vcf to correct chromesome and make DataFrame rep
	chr_mask = giab_callset['variants/CHROM'] == ls_callset['variants/CHROM'][0]
	giab_callset = mask_callset(giab_callset, chr_mask)

	giab_df = pd.DataFrame(
		np.vstack((
			giab_callset['variants/CHROM'],
			giab_callset['variants/POS'],
			giab_callset['variants/REF'],
			[','.join(giab_callset['variants/ALT'][i]) for i in range(
					giab_callset['variants/ALT'].shape[0])],
			[encode_genotype(gt[0]) for gt in giab_callset['calldata/GT']]
		)).T,
		columns=['chrom', 'pos', 'ref', 'alt', 'genotype']
	)

	df = pd.DataFrame(
		np.array((
			range(ls_callset['variants/POS'].shape[0]), 
			in_bed_mask, 
			ls_callset['variants/CHROM'],
			ls_callset['variants/POS'],
			ls_gt
		)).T,
		columns=[
			'site_ind', 'in_bed', 'chrom', 'pos',
			'ls_hmm_pred_genotype'
		]
	)

	df = pd.merge(df, giab_df, how='left', on=['chrom', 'pos'])

	# fill for genotype=0 sites in bed but not var sites
	df.loc[df.in_bed == 1, 'genotype'] = df.loc[df.in_bed == 1]['genotype'].fillna(0)

	# save
	if save_path:
		df.to_csv(save_path, na_rep='', sep='\t', index=False)

	# load read fragments and their qualities
	_, fragments, qualities = read_fragments(fragments_path)

	return df, fragments, qualities
	

def load_longshot_data(fragments_path, longshot_vcf_path, 
						return_vcf=False):
	''' 
	Load data from fragments.txt & 2.0.realigned_genotypes.vcf files
	generated by Longshot.

	Args:
		fragments_path: path to fragments.txt
		longshot_vcf_path: path to 2.0.realigned_genotypes.vcf
		return_vcf: also returns longshot_vcf_path data dict as read
			by scikit-allel when True

	Returns:
		site_data, fragments, qualities, vcf_dict(opt)

		site_data: DataFrame with info for all sites HMM predicts are
			heterozygous
		fragments: fragments matrix as returned by read_fragments, possibly
			masked
		qualities: qualities matrix as returned by read_fragments, possibly
			masked
		vcf_dict: longshot_vcf_path data dict as read by scikit-allel
	'''
	# load vcf from longshot run
	ls_callset = allel.read_vcf(longshot_vcf_path)

	# find where longshot hmm predicts heterozygous
	ls_01 = np.all(np.equal(ls_callset['calldata/GT'], [0,1]), axis=2).T[0]
	ls_10 = np.all(np.equal(ls_callset['calldata/GT'], [1,0]), axis=2).T[0]
	ls_hetero = ls_01 | ls_10

	df = pd.DataFrame(
		np.array((
			range(ls_callset['variants/POS'].shape[0]),
			ls_callset['variants/CHROM'],
			ls_callset['variants/POS'],
			ls_hetero.astype(int)
		)).T,
		columns=['site_ind', 'chrom', 'pos', 'hmm_pred_hetero']
	)

	# load read fragments and their qualities
	_, fragments, qualities = read_fragments(fragments_path)

	if return_vcf:
		return df, fragments, qualities, ls_callset
	else:
		return df, fragments, qualities


if __name__ == '__main__':
	fragments_path='data/fragments/chr20_1-1M/fragments.txt'
	longshot_vcf_path='data/fragments/chr20_1-1M/2.0.realigned_genotypes.vcf'
	ground_truth_vcf_path='data/GIAB/HG002_GRCh38_1_22_v4.1_draft_benchmark.vcf'
	giab_bed_path='data/GIAB/HG002_GRCh38_1_22_v4.1_draft_benchmark.bed'
	site_data_save_path='data/preprocessed/1M_site_data.tsv'

	# df, fragments, qualities, ff, _ = load_data(
	# 	fragments_path, 
	# 	longshot_vcf_path, 
	# 	ground_truth_vcf_path,
	# 	giab_bed_path, 
	# 	save_path=site_data_save_path)

	# df, fragments, qualities = load_longshot_data(fragments_path, longshot_vcf_path)

	df, fragments, qualities = load_full_data(fragments_path, longshot_vcf_path, ground_truth_vcf_path,
				giab_bed_path)