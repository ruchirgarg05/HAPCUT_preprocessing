import math
import numpy as np
import pandas as pd

from utils import cluster_fragments, generate_matrix_for_visualization


def get_probability_fragments_from_same_fragment(reads, st_ens, index, error_rate):
    """
    Return a tuple of probabilities of 
     - read1 and read2 coming from the same Haplotype fragment
     - read1 and read2 coming from different Haplotype fragment
    
    param reads: the value of the fragments
    param st_ens: the start and end of each of the two reads 
    param error_rate: quality of the reads
    """
    assert len(reads) == len(st_ens) == 2
    # (11, 15)
    # (12, 14)
    # (12, 14)
    #import pdb;pdb.set_trace()
    st, en = max(st_ens[0][0], st_ens[1][0]), min(st_ens[0][1], st_ens[1][1])
    # Length of the overlapping part of the two reads.
    len_common = en - st
    r0_index_st, r1_index_st = st - st_ens[0][0], st - st_ens[1][0]
    r0_index_en, r1_index_en = r0_index_st + len_common, r1_index_st + len_common
    idx = None if index is None else index-st    
    read0, read1 = reads[0][r0_index_st: r0_index_en], reads[1][r1_index_st: r1_index_en]
    same, diff = 1., 1.
    
    for i, (v0, v1) in enumerate(zip(read0, read1)):
        if i == idx:
            continue
        if v0 == v1:
            same *= (1 - error_rate)
            diff *= error_rate
        else:
            # v0 and v1 are not same
            same *= error_rate
            diff *= (1- error_rate)
    sum_prob = same + diff
    same, diff = same/sum_prob, diff/sum_prob  
    return same, diff


def get_overlapping_fragments_for_variants_sites(reads, st_ens, index):
    """
    Return the fragments which contain the sites with variant at "index".
    """
    reads, st_ens = cluster_fragments(reads, st_ens)
    overalapping_frags = []
    overlapping_st_ens = []
    for read, st_en in zip(reads, st_ens):
        if st_en[0] > index:
            break
        if index>=st_en[0] and index<st_en[1]:
            overalapping_frags.append(read)
            overlapping_st_ens.append(st_en)
    return overalapping_frags, overlapping_st_ens


def get_likelihood_heterozygous_genotype(reads, st_en, index, qual):
    """
    Return the likelihood of the site at index being a heterozygous genotype, for the reads 
    - coming from same Haplotype,
    - coming from the different Haplotype.
    """
    #import ipdb;ipdb.set_trace()
    assert len(reads) == 2 and len(st_en)==2
    assert index>=max(st_en[0][0], st_en[1][0]) and index<=min(st_en[0][1], st_en[1][1])
    index0, index1 = index - st_en[0][0], index - st_en[1][0]
    try:
        assert index0 < len(reads[0]) and index1<len(reads[1])
    except:
        import ipdb;ipdb.set_trace()    
    if reads[0][index0] == reads[1][index1]:
        # Same value at the site
        likelihood_same = ((1-qual)*(1-qual))/2 + (qual*qual)/2
        likelihood_diff = qual * (1 - qual)
    else:
        # Different values at the given site
        likelihood_same = qual * (1 - qual)
        likelihood_diff = ((1-qual)*(1-qual))/2 + (qual*qual)/2
    # normalize the likelihood, as likelihood_same and likelihood_diff 
    # do not add upto 1.            
        
    sum_likelihood = likelihood_same + likelihood_diff   
    return likelihood_same/ sum_likelihood, likelihood_diff/ sum_likelihood


def calculate_likelihood_of_heterozygous_site(reads, st_en, index, qual, ref_H=None, false_variant_locs=None):
    """
    Given the reads what is the likelihood of allele lying at index being 
    a heterozygous genotype.
    param qual: the qual of the reads, currently it is a constant.maketrans()
    # TODO: Make qual an array which stores the qual of each of the index.
    """
    #import ipdb;ipdb.set_trace()
    reads, st_en = cluster_fragments(reads, st_en)
    overlapping_reads, overlapping_st_en = get_overlapping_fragments_for_variants_sites(reads, st_en, index)    
    #generate_matrix_for_visualization(ref_H, false_variant_locs, overlapping_reads, overlapping_st_en)
    num_frags = len(overlapping_reads)
    if num_frags == 0:
        # There are no overlapping reads covering the given site.
        # return None since we dont have anything to compare it to.
        return np.nan
    
    # Let the first bit for H1 be 0 and first bit of H2 = 1( without loss of generality) 
    # Likelihood for the reads lying in H1, H2 given i fragments are stored in the ith index of the
    # array likelihood_per_reads
    
    #ipdb.set_trace()
    frag_0 = (overlapping_reads[0], overlapping_st_en[0])
    likelihood_per_reads = [(1-qual, qual) if overlapping_reads[0][0] == 0 else (qual, 1-qual)]
    
    for i in range(1, num_frags):
        frag_1 = (overlapping_reads[i], overlapping_st_en[i])
        r, s_e = [frag_0[0], frag_1[0]],  [frag_0[1], frag_1[1]]
        
        prob_same, prob_diff = get_probability_fragments_from_same_fragment(r, s_e, index, qual)             
        same_lik, diff_lik = get_likelihood_heterozygous_genotype(r, s_e, index, qual)        
        l1_prev, l2_prev = likelihood_per_reads[-1]
        
        # l1 is the likelihood that the xi fragment is sampled from the H1
        # l2 is the likelihood that the xi fragment is sampled from the H2.         
        l1 = l1_prev*prob_same*same_lik + l2_prev*prob_diff*diff_lik
        l2 = l1_prev*prob_diff*diff_lik + l2_prev*prob_same*same_lik
        
        likelihood_per_reads.append((l1, l2))       
        
        frag_0 = frag_1
    # Finally my likelihood per_reads would contain,
    # the likelihood of the site at index to be heterozygous
    heteroz_likelihood = likelihood_per_reads[-1][0] + likelihood_per_reads[-1][1]
    return  heteroz_likelihood

def remove_site_from_samples(samples, st_en, index):
    """
    Remove the site from all the possible fragments from the given sample.
    Implementation detail: Do not delete the variant site, simply mark it as "_"
    """
    reads, st_en = cluster_fragments(samples, st_en)
    nreads = []
    for r, s_e in zip(reads, samples):
        if index > s_e[0]:
            # Index are sorted by start index
            break
        if index>=s_e[0] and index<=s_e[1]:
            idx = index - s_e[0]
            r[idx] = -1
        nreads.append(r)
    return nreads, st_en


def get_likelihood_without_haplotype_information(reads, st_en, ref_H):
    reads, st_en = cluster_fragments(reads, st_en)
    likelihoods = []
    for i in range(len(ref_H[0])):
        overlapping_reads = get_overlapping_fragments_for_variants_sites(reads, st_en, i)
        likelihoods.append(1/(1<<len(overlapping_reads[0])))
    return likelihoods


def remove_false_variants(reads, st_en, qual, ref_H=None):
    """
    Removes the sites which has less likelihood of it being heterozygous.
    """
    #import ipdb;ipdb.set_trace()
    false_variants = []
    while True:
        likelihood_false_variants = []
        likelihood_no_hap_info = get_likelihood_without_haplotype_information(reads, st_en, ref_H)
              
        for i in range(len(ref_H[0])):
            likelihood_false_variants.append((calculate_likelihood_of_heterozygous_site(reads, st_en, i, qual, ref_H), i))
        #ipdb.set_trace()
        likelihood_false_variants = sorted(likelihood_false_variants)
        false_variant_locs = []
        for likelihood, idx in likelihood_false_variants:
            if likelihood < likelihood_no_hap_info[idx]:
                false_variant_locs.append(idx)    
        if not len(false_variant_locs):
            break
        false_variants += false_variant_locs   
        for idx in false_variant_locs:
            # TODO: Remove all the false variants together.
            reads, st_en = remove_site_from_samples(reads, st_en, idx)
    return reads, st_en, false_variants