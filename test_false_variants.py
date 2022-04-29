# Create a test suite to test all the possibilities
from utils import *
from preprocess_utils import *
from sim_util import *
from copy import deepcopy
import unittest

def test_false_vars_locs(ref_length=30, coverage=6, read_length=5, std_read_length=1):    
    H12, false_variant_locs = create_reference_hap(ref_length=ref_length,
                                                    false_variance = 0.1, 
                                                    print_info=True)
    ref_H = deepcopy(H12)
    samples, st_en = generate_fragments(ref_H,
                                        read_length=read_length,
                                        std_read_length=std_read_length,
                                        coverage=coverage)
    hap_samples, start_end = deepcopy(samples), deepcopy(st_en)
    reads = [(s, np.array([0.01]*len(s))) for s in hap_samples]
    RS, ST_EN = cluster_fragments(reads, start_end)
    _, __, false_vars = remove_false_variants(RS, ST_EN, len(ref_H[0]))
    assert not len(set(false_vars) - set(false_variant_locs))
    
def test_real_data_1(fragments_path='/home/ruchirgarg5/content/data/debug/fragments.txt',
                   longshot_vcf_path='/home/ruchirgarg5/content/data/debug/2.0.realigned_genotypes.vcf',
                   pre_processed_longshot_vcf_path='/home/ruchirgarg5/content/data/debug/2.1.realigned_genotypes_preprocessed.vcf',
                   ground_truth_vcf_path='/home/ruchirgarg5/content/data/HG003_GRCh38_chr20_v4.2.1_benchmark.vcf.gz',
                   giab_bed_path='/home/ruchirgarg5/content/data/HG003_GRCh38_chr20_v4.2.0_benchmark_noinconsistent.bed'):
    import allel
    a, fragments, quals = read_fragments(fragments_path)
    ls_callset = allel.read_vcf(longshot_vcf_path)
    ls_01 = np.all(np.equal(ls_callset['calldata/GT'], [0,1]), axis=2).T[0]
    ls_10 = np.all(np.equal(ls_callset['calldata/GT'], [1,0]), axis=2).T[0]
    ls_het = ls_01 | ls_10
    quals_het, fragments_het, ls_callset_hetero = quals[:, ls_het], fragments[:, ls_het], ls_callset["calldata/GT"][ls_het]
    reads, st_en = cluster_fragments(*compress_fragments(fragments_het, quals_het))

    _, __, false_vars = remove_false_variants(reads, st_en, len(fragments_het[0]))
    for idx, data in false_vars.items():
        genotype = data[0]
        confid, coverage = data[1][0], data[1][1]
        if coverage>10 and confid>100:
            ls_callset_hetero[idx] = np.array([[genotype, genotype]])

    ls_callset["calldata/GT"][ls_het] = ls_callset_hetero
    allel.write_vcf(pre_processed_longshot_vcf_path, ls_callset)
    
    get_hetero()
    
     # coverages = get_coverage_for_all_the_sites(reads, st_en, len(fragments_het[0]))
    # het_likelihood = get_likelihood_with_haplotype_information(reads, st_en, len(fragments_het[0]))
    # epsilon = 0.04
    # false_variants = dict(classifier(het_likelihood, coverages, epsilon))
    # false_variants_locs = list(false_variants.keys())
    
    
    # _, fragments, qualities = read_fragments(fragments_path)
    # #quals = np.power(10, -0.1*qualities)
    # fr1, qual1 = fragments[ls_het], qualities[ls_het]
    # reads, st_en = compress_fragments(fr1, qual1)
    # reads, st_en = cluster_fragments()
    
    # _, __, false_variants = remove_false_variants(reads, st_en, len(fr1[0]))
    
    
    