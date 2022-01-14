# imports
import math
import numpy as np
import pandas as pd
import os
import random

pd.set_option('display.max_columns', 40)

def create_reference_hap(ref_length, p=0.5, false_variance=0.0, print_info=True):
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
    # Make the ref hap 0/0 or 1/1 in both h1 or h2.
    num_false_variants = int(len(reference_hap1) * false_variance) 
    false_variant_locs = np.unique(np.random.randint(low = 0, 
                                                     high=ref_length,
                                                     size=num_false_variants))
    # Flip the bits in either reference hap1 or reference hap2.
    flip_h1_h2 = np.random.choice([0, 1 ], size=num_false_variants)
    for location, h1_h2 in zip(false_variant_locs, flip_h1_h2):
      haps[h1_h2][location] = not haps[h1_h2][location]
      
    for loc in false_variant_locs:
      assert haps[0][loc] == haps[1][loc]
    if print_info:
      print_false_variants_and_ref_H(haps, false_variant_locs)  
        
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
    
    # Uniformly choose start points from the reference haplotype, h1 or h2. 
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

def print_false_variants_and_ref_H(ref_H, false_variant_locs):
        # print("\n")    
        # print(" ".join([str(v) for v in np.array(ref_H[0]).astype("int")]))
        # print(" ".join([str(v) for v in np.array(ref_H[1]).astype("int")]))
        # print("\n")
        print(f"False variant locations are {sorted(false_variant_locs)}") 
        print("reference haplotype values for the variants are:")
        print(pd.DataFrame(np.array([h.astype("int") for h in ref_H]), 
                       columns = [f"{i}" for i in range(len(ref_H[0]))],
                       index=["H1", "H2"]).to_string())        

        
def compress_fragments(fragments):
  """
  Get a matrix of 0, 1, np.nan for each read. 
  We need to return st, en and the fragment
  """
  #import ipdb;ipdb.set_trace()
  reads, st_en = [], []
  for frag in fragments:
    ones = np.where(frag==1.)[0]
    zeros = np.where(frag==0.)[0]
    if not len(ones) and not len(zeros):
      # no valid values in the fragment. 
      continue
    if len(ones):
      s = min(ones)
      e = max(ones)
      if len(zeros):
        s = min(s, min(zeros))
        e = max(e, max(ones))
        
    if len(zeros):
        s = min(zeros)
        e = max(zeros)
        if len(ones):
          s = min(s, min(ones))
          e = max(e, max(ones))
            
    # Each fragment either has 0, or 1 or
    reads.append(frag[s:e+1])
    st_en.append((s, e+1))
  return reads, st_en
       
def generate_matrix_for_visualization(ref_H, false_variant_locs, 
                                      hap_samples, st_en):
    reference_length = len(ref_H[0]) 
    print_false_variants_and_ref_H(ref_H, false_variant_locs)
    
    matrix = [["-" for _ in range(reference_length)] for _ in range(len(hap_samples))]
    for idx, ( (s,e),  sa ) in enumerate(zip(st_en, hap_samples)):
        for i, v in zip(range(s, e), sa):
            if v != -1:
                matrix[idx][i] = str(int(v))
    
    # for m in matrix:
    #     print(" ".join(str(v) for v in m) )
    with pd.option_context('display.max_rows', 100, 'display.max_columns', None):  # more options can be specified also
        print(pd.DataFrame(np.array(matrix),
                           columns = [f"{i}" for i in range(len(ref_H[0]))], 
                           index=[f"f_{i}" for i in range(len(matrix))]).to_string())   

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
  H_samples = sorted(H_samples, key=lambda V: (V[0][0], -1*V[0][1]))
  st_en = [(st, en) for (st, en), _ in H_samples]
  samples = [sample for _, sample in H_samples]
  return samples, st_en