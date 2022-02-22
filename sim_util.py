# imports
import math
import numpy as np
import pandas as pd
import os
import random
import warnings


pd.set_option('display.max_columns', 40)

def create_reference_hap(ref_length, p=0.5, false_variance=0.0, print_info=False):
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

def generate_samples_of_len(sz, freq):
    sample_size = 1000
    samples = []
    for v, pr in freq:
        samples += [v] * int(pr*sample_size)
    samples = np.array(samples)
    np.random.shuffle(samples)
    idx = np.random.randint(0, len(samples)-1, size = sz)
    return samples[idx]


def get_error_freq(fragments_path, longshot_vcf_path, plot=False):
    import allel
    from utils import read_fragments
    _, fragments, quals = read_fragments(fragments_path)
    callset = allel.read_vcf(longshot_vcf_path)
    filter = callset["variants/FILTER_PASS"]
    fragments_filter, quals_filter = fragments[:, filter], quals[:, filter]

    reads, st_en = cluster_fragments(*compress_fragments(fragments_filter, quals_filter))
    quals_array = [ np.array([x for y, x in zip(r[0],r[1]) if (y == 0 or y  == 1) ]) for r in reads]
    quals1 = np.concatenate(quals_array)
    values, counts = np.unique(quals1, return_counts=True)
    probs = counts / sum(counts)
    if plot:
      import matplotlib
      import matplotlib.pyplot as plt

      matplotlib.rcParams['figure.figsize'] = (16.0, 12.0)
      matplotlib.style.use('ggplot')

      plt.vlines(values, 0, probs, color='C0', )
    freq = np.asarray([values, probs]).T
    return freq    

def generate_miscall_error_rates_for(samples):
  """
  Zips the samples with their corresponding miscaal error rates
  """
  error_samples = []
  fragments_path='/home/ruchirgarg5/content/data/debug/fragments.txt'
  longshot_vcf_path='/home/ruchirgarg5/content/data/debug/2.0.realigned_genotypes.vcf'

  freq = get_error_freq(fragments_path, longshot_vcf_path)

  for sample in samples:
    error_frag = generate_samples_of_len(len(sample), freq)
    error_samples.append([sample,  error_frag])
  return error_samples  


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


def convert_qualities_to_error_rate(reads):
  reads_er = []
  for frag, qual in reads:
    reads_er.append((frag, np.power(10, -0.1*qual)))
  return reads_er  

def compress_fragments(fragments, qualities, return_index=False):
  """
  Get a matrix of 0, 1, np.nan for each read. 
  We need to return st, en and the fragment
  """
  #import ipdb;ipdb.set_trace()
  reads, st_en = [], []
  qualities = np.power(10, -0.1*qualities)
  index = []
  assert fragments.shape == qualities.shape
  fragments = np.nan_to_num(fragments, nan=-1.)
  
  for i, (frag, qual) in enumerate(zip(fragments, qualities)):
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
    reads.append((frag[s:e+1], qual[s:e+1]))
    st_en.append((s, e+1))
    index.append(i)
  if return_index:
    return reads, st_en, index
  return reads, st_en

def visualize_overlapping_reads_at(reads, st_en, index):
    import ipdb;ipdb.set_trace()
    from preprocess_utils import get_overlapping_fragments_for_variants_sites
    reads, st_en = cluster_fragments(reads, st_en)
    reads, st_en = get_overlapping_fragments_for_variants_sites(reads, st_en, index)
    min_idx = min(st_en, key=lambda v: v[0])[0]
    max_idx = max(st_en, key=lambda v: v[1])[1]
    matrix = [["-" for _ in range(max_idx - min_idx)] for _ in range(len(st_en))]
    for idx, ( (s,e),  sa ) in enumerate(zip(st_en, reads)):
        s, e = s - min_idx, e - min_idx
        for i, v in zip(range(s, e), sa[0]):
            if v == 0 or v == 1:
                matrix[idx][i] = str(int(v))

    idx_index = index - min_idx
    with pd.option_context('display.max_rows', 100, 'display.max_columns', None):  # more options can be specified also
        offset = index - min_idx
        st = max(offset - 5,  0) # 17
        en = min(offset + 5, max_idx - min_idx) # 27
        mat = np.array(matrix)[:, st:en]
        print(f"index corresponding to {index} is {idx_index}")
        df = pd.DataFrame( mat,
                           columns = [f"{i + st}" for i in range(en - st)], 
                           index=[f"f_{i}" for i in range(len(matrix))])

        print(df.to_string())   
       
def generate_matrix_for_visualization(ref_H, false_variant_locs, 
                                      hap_samples, st_en):
    import ipdb;ipdb.set_trace()
    reference_length = len(ref_H[0]) 
    print_false_variants_and_ref_H(ref_H, false_variant_locs)
    
    matrix = [["-" for _ in range(reference_length)] for _ in range(len(hap_samples))]
    for idx, ( (s,e),  sa ) in enumerate(zip(st_en, hap_samples)):
        for i, v in zip(range(s, e), sa[0]):
            if v != -1:
                matrix[idx][i] = str(int(v))
    
    # for m in matrix:
    #     print(" ".join(str(v) for v in m) )
    with pd.option_context('display.max_rows', 100, 'display.max_columns', None):  # more options can be specified also
        print(pd.DataFrame(np.array(matrix),
                           columns = [f"{i}" for i in range(len(ref_H[0]))], 
                           index=[f"f_{i}" for i in range(len(matrix))]).to_string())   



# Create models from data
def best_fit_distribution(data, bins=200, ax=None):
    """Model data by finding best fit distribution to data"""
    import scipy.stats as st
    import statsmodels.api as sm
    from scipy.stats._continuous_distns import _distn_names
    # Get histogram of original data
    y, x = np.histogram(data, bins=bins, density=True)
    x = (x + np.roll(x, -1))[:-1] / 2.0

    # Best holders
    best_distributions = []

    # Estimate distribution parameters from data
    for ii, distribution in enumerate([d for d in _distn_names if not d in ['levy_stable', 'studentized_range']]):

        print("{:>3} / {:<3}: {}".format( ii+1, len(_distn_names), distribution ))

        distribution = getattr(st, distribution)

        # Try to fit the distribution
        try:
            # Ignore warnings from data that can't be fit
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')
                
                # fit dist to data
                params = distribution.fit(data)

                # Separate parts of parameters
                arg = params[:-2]
                loc = params[-2]
                scale = params[-1]
                
                # Calculate fitted PDF and error with fit in distribution
                pdf = distribution.pdf(x, loc=loc, scale=scale, *arg)
                sse = np.sum(np.power(y - pdf, 2.0))
                
                # if axis pass in add to plot
                try:
                    if ax:
                        pd.Series(pdf, x).plot(ax=ax)
                    end
                except Exception:
                    pass

                # identify if this distribution is better
                best_distributions.append((distribution, params, sse))
        
        except Exception:
            pass

    
    return sorted(best_distributions, key=lambda x:x[2])

def make_pdf(dist, params, size=10000):
    """Generate distributions's Probability Distribution Function """

    # Separate parts of parameters
    arg = params[:-2]
    loc = params[-2]
    scale = params[-1]

    # Get sane start and end points of distribution
    start = dist.ppf(0.01, *arg, loc=loc, scale=scale) if arg else dist.ppf(0.01, loc=loc, scale=scale)
    end = dist.ppf(0.99, *arg, loc=loc, scale=scale) if arg else dist.ppf(0.99, loc=loc, scale=scale)

    # Build PDF and turn into pandas Series
    x = np.linspace(start, end, size)
    y = dist.pdf(x, loc=loc, scale=scale, *arg)
    pdf = pd.Series(y, x)

    return pdf

def best_fit(data):
  # Load data from statsmodels datasets
  data = pd.Series(data)

  # Plot for comparison
  plt.figure(figsize=(12,8))
  ax = data.plot(kind='hist', bins=50, density=True, alpha=0.5, color=list(matplotlib.rcParams['axes.prop_cycle'])[1]['color'])

  # Save plot limits
  dataYLim = ax.get_ylim()

  # Find best fit distribution
  best_distibutions = best_fit_distribution(data, 200, ax)
  best_dist = best_distibutions[0]

  # Update plots
  ax.set_ylim(dataYLim)

  # Make PDF with best params 
  pdf = make_pdf(best_dist[0], best_dist[1])

  # Display
  plt.figure(figsize=(12,8))
  ax = pdf.plot(lw=2, label='PDF', legend=True)
  data.plot(kind='hist', bins=50, density=True, alpha=0.5, label='Data', legend=True, ax=ax)

  param_names = (best_dist[0].shapes + ', loc, scale').split(', ') if best_dist[0].shapes else ['loc', 'scale']
  param_str = ', '.join(['{}={:0.2f}'.format(k,v) for k,v in zip(param_names, best_dist[1])])
  dist_str = '{}({})'.format(best_dist[0].name, param_str)

  ax.set_title(u'El Niño sea temp. with best fit distribution \n' + dist_str)
  ax.set_xlabel(u'Temp. (°C)')
  ax.set_ylabel('Frequency')

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
      assert len(sample[0]) == abs(x[1] - x[0])  
      switch_error = np.less( np.random.uniform(0, 1, size=len(sample[0])) , switch_error_rate)     
      missing_error = np.less(np.random.uniform(0, 1, size=len(sample[0])) , missing_error_rate)
      error_rate = sample[1]
      miscall_error = np.less(np.random.uniform(0, 1, size=len(sample[0])) , error_rate)
      # Simulate false variants
      # For the samples which have false variant locations in it, we flip the 
      # bits sampled with probability 0.5, from either of the samples.
      for idx in range(x[0], x[1]):
        if idx in false_variant_locs:
          if np.random.random() < 0.5:
            sample[0][idx-x[0]] = not sample[0][idx-x[0]]
            
      
      #import ipdb;ipdb.set_trace()
      # Simulate miscall errors      
      new_sample = []
      for sa, miscall in zip(sample[0], miscall_error):
        if miscall and (sa == 0 or sa ==1): 
          new_sample.append(1 - sa)
        else:
          new_sample.append(sa)
      
      updated_sample = new_sample 
      assert len(updated_sample) == abs(x[1] - x[0])

      # Simulate switch errors
      switch_idxs = list(np.nonzero(switch_error))
      is_switched = False
      new_sample = []
      for sa, sw in zip(updated_sample, switch_error):
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

      # Simulate missing errors  
      new_sample = []

      for sa, missing in zip(updated_sample, missing_error):
        if missing:
          new_sample.append(-1)
        else:
          new_sample.append(sa)
      assert len(new_sample) == abs(x[1] - x[0])
      # INdex of the variant, reference hap value, misscall rate
      #import ipdb;ipdb.set_trace()
      fragfilecontent.append((x, new_sample, qual))      
      sw_hap_samples.append([new_sample, error_rate])
        
    #import pdb;pdb.set_trace()
    # Update the hap samples with the new samples
    hap_samples = [[list(np.array(s, dtype="int32")) ,e ] for s, e in sw_hap_samples]

    return hap_samples, reads_st_en, fragfilecontent
  

def cluster_fragments(hap_samples, st_en, index=None):
  #import pdb;pdb.set_trace() 
  if index is None:
    H_samples = [((st,en), sample) for (st, en), sample in zip(st_en, hap_samples)]
  else:
    H_samples = [((st,en), sample, idx) for (st, en), sample, idx in zip(st_en, hap_samples, index)]
  
  H_samples = sorted(H_samples, key=lambda V: (V[0][0], -1*V[0][1]))
  if index is None:
    st_en, samples  = list(zip(*H_samples))
    return samples, st_en
  else:
    st_en, samples, indexes  = list(zip(*H_samples))
    return samples, st_en, indexes