# imports
import math
import numpy as np
import os
import random


def create_reference_hap(ref_length, p=0.5, false_variance=0.0):
  """
  param false_variance: If set to true, there are some homozygous genotypes 
  at random locations, and also return its positions. THis is to simulate the false 
  variant reads in the sequenced fragments.
  """
  p = 0.5
  reference_hap1 = np.random.choice(a=[False, True], size=(ref_length), p=[p, 1-p])
  reference_hap2 = np.logical_not(reference_hap1)
  haps = [reference_hap1, reference_hap2]
  false_variant_locs = []
  if false_variance > 0.:
    #import pdb;pdb.set_trace()
    #import pdb;pdb.set_trace()
    
    # Make the ref hap 0/0 or 1/1 in both h1 or h2.
    num_false_variants = int(len(reference_hap1) * false_variance) 
    false_variant_locs = np.unique(np.random.randint(low = 0, 
                                                     high=ref_length,
                                                     size=num_false_variants))
    # Flip the bits in either reference hap1 or reference hap2.
    flip_h1_h2 = np.random.choice([0, 1 ], size=num_false_variants)
    for location, h1_h2 in zip(false_variant_locs, flip_h1_h2):
      haps[h1_h2][location] = not haps[h1_h2][location]
    print("\n")
    print(false_variant_locs)    
    print(" ".join([str(v) for v in np.array(haps[0]).astype("int")]))
    print(" ".join([str(v) for v in np.array(haps[1]).astype("int")]))
    print("\n")
    for loc in false_variant_locs:
      assert haps[0][loc] == haps[1][loc]     
  return haps, false_variant_locs

def generate_fragments(ref_H, read_length, std_read_length, coverage):
    read_length = int(read_length)
    ref_length = len(ref_H[0])
    num_reads_for_read_coverage = int(ref_length * coverage / read_length)
    # Pick the length of the read length from a normal distribution centered at
    # read_lenth and variance as std_read_length
    read_lengths = np.array(
        np.random.normal(loc = read_length, scale = std_read_length, 
                         size = num_reads_for_read_coverage), 
                         dtype="int32")
    
    # Uniformy choose start points from the reference haplotype, h1 or h2. 
    reads_start_idx = np.random.randint(low = 0, high=ref_length-read_length, 
                                        size=num_reads_for_read_coverage)
    
    reads_st_en = []
    #reads = np.array([reads_start_idx, reads_start_idx + read_length])
    for st_idx, read_len in zip(reads_start_idx, read_lengths):
      if st_idx + read_len > ref_length - 1:
        en = ref_length - 1
      else:
        en = st_idx + read_len
      reads_st_en.append( (st_idx, en ) )

    #reads_st_en = np.array(reads_st_en)
    #import pdb;pdb.set_trace()
    # We have the st and en of the reads, choose either h1 or h2 from H and 
    # sample it. Sample from S1 if True else sample from S2.
    h1_or_h2 = np.array(
        np.less(np.random.uniform(0, 1, num_reads_for_read_coverage), 0.5), 
        dtype="int")
    
    hap_samples = [np.array(ref_H[v][st:en]).astype("int") for v, (st, en) in zip(h1_or_h2, reads_st_en)]
    assert len(hap_samples) == num_reads_for_read_coverage
    return hap_samples, reads_st_en,


def generate_matrix_for_visualization(ref_H, false_variant_locs, 
                                      hap_samples, st_en):
    reference_length = len(ref_H[0])
    def print_false_variants_and_ref_H(ref_H, false_variant_locs):
        print(f"False variant locations are {sorted(false_variant_locs)}") 
        print("reference haplotype values for the variants are:")
        print("\n")    
        print(" ".join([str(v) for v in np.array(ref_H[0]).astype("int")]))
        print(" ".join([str(v) for v in np.array(ref_H[1]).astype("int")]))
        print("\n")
        
    print_false_variants_and_ref_H(ref_H, false_variant_locs)
    matrix = [["-" for _ in range(reference_length)] for _ in range(len(hap_samples))]
    for idx, ( (s,e),  sa ) in enumerate(zip(st_en, hap_samples)):
        for i, v in zip(range(s, e), sa):
            if v != -1:
                matrix[idx][i] = v
                
    for m in matrix:
        print(" ".join(str(v) for v in m) )

def simulate_haplotypes_errors(
        hap_samples, reads_st_en,
        false_variant_locs, 
        switch_error_rate=0.0, 
        miscall_error_rate=0.0, 
        missing_error_rate=0.0
        ):
    """
    param read_length: Avg read length.
    param std_read_length How the read lengths is distributed around the read_length, which is avg read length. 
    param coverage: Number of avegage reads per genome for the reference over snps
    param reference_length: THe length of the reference haptype 
    param false_variance:
    param switch_error_rate: 
    param missing_error_rate:
    param miscall_error_rate:
    """ 
    # Simulate switch errors:
    sw_hap_samples = []
    fragfilecontent = []
    # Get the switch error for each of the sample.
    # Get the missing error for each of the sample 
    qual = '~' if miscall_error_rate < 1e-9 else str(int(-10*math.log10(miscall_error_rate)))
    

    for sample, x in zip(hap_samples, reads_st_en):
      #import pdb;pdb.set_trace()      
      assert len(sample) == abs(x[1] - x[0])  
      switch_error = np.less( np.random.uniform(0, 1, size=len(sample)) , switch_error_rate)
      miscall_error = np.less(np.random.uniform(0, 1, size=len(sample)) , miscall_error_rate)      
      missing_error = np.less(np.random.uniform(0, 1, size=len(sample)) , missing_error_rate)
      
      # Simulate false variants
      # For the samples which have false variant locations in it, we flip the 
      # bits sampled with probability 0.5, from either of the samples.
      for idx in range(x[0], x[1]):
        if idx in false_variant_locs:
          if np.random.random() < 0.5:
            sample[idx-x[0]] = not sample[idx-x[0]]
            

      # Simulate switch errors
      switch_idxs = list(np.nonzero(switch_error))
      is_switched = False
      new_sample = []
      for sa, sw in zip(sample, switch_error):
        if sw:
          is_switched = not is_switched
          if is_switched:
            new_sample.append(not sa)
          else:
            new_sample.append(sa) 
        else:
            new_sample.append(sa) 
            
      updated_sample = new_sample
      assert len(updated_sample) == abs(x[1] - x[0])

      # Simulate miscall errors      
      new_sample = []
      for sa, miscall in zip(updated_sample, miscall_error):
        if miscall:
          new_sample.append(not sa)
        else:
          new_sample.append(sa)
      
      updated_sample = new_sample 
      assert len(updated_sample) == abs(x[1] - x[0])

      # Simulate missing errors  
      new_sample = []

      for sa, missing in zip(updated_sample, missing_error):
        if missing:
          new_sample.append(-1)
        else:
          new_sample.append(sa)
      assert len(new_sample) == abs(x[1] - x[0])
      # INdex of the variant, reference hap value, misscall rate
        
      fragfilecontent.append((x, new_sample, qual))      
      sw_hap_samples.append(new_sample)
        
    #import pdb;pdb.set_trace()
    # Update the hap samples with the new samples
    hap_samples = [list(np.array(s, dtype="int32")) for s in sw_hap_samples]

    return hap_samples, reads_st_en, fragfilecontent
  

def cluster_fragments(hap_samples, st_en):
  #import pdb;pdb.set_trace()  
  H_samples = [((st,en), sample) for (st, en), sample in zip(st_en, hap_samples)]
  H_samples = sorted(H_samples, key=lambda V: V[0])
  st_en = [(st, en) for (st, en), _ in H_samples]
  samples = [sample for _, sample in H_samples]
  return samples, st_en


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
        if index>=st_en[0] and index<=st_en[1]:
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
    index0, index1 = index - st_en[0][0] -1 , index - st_en[1][0] - 1
    if reads[0][index0] == reads[1][index1]:
        # Same value at the site
        likelihood_same = ((1-qual)*(1-qual))/2 + (qual*qual)/2
        likelihood_diff = qual * (1 - qual)
    else:
        # Different values at the given site
        likelihood_same = qual * (1 - qual)
        likelihood_diff = ((1-qual)*(1-qual))/2 + (qual*qual)/2
    return likelihood_same, likelihood_diff

 

def calculate_likelihood_of_heterozygous_site(ref_H, false_variant_locs, reads, st_en, index, qual):
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