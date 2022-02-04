from sim_util import *
from preprocess_utils import *
from utils import *

from copy import deepcopy

def return_false_variant_prob(ref_length = 300, coverage=20, read_length=10,
                          std_read_length=2, false_variance_density=0.1):
    #ref_length, coverage, read_length, std_read_length, false_variance_density):
    """
    Return false variance with:
     -
    """
    H12, false_variant_locs = create_reference_hap(ref_length=ref_length,
                                                false_variance = false_variance_density)
    ref_H = deepcopy(H12)
    samples, st_en = generate_fragments(ref_H,
                                        read_length=read_length,
                                        std_read_length=std_read_length,
                                        coverage=coverage)


    hap_samples = deepcopy(samples)
    hap_samples = generate_miscall_error_rates_for(hap_samples)
    S1, ST_EN = cluster_fragments(hap_samples, st_en)
    er_hap_samples, sts_ens, _ = simulate_haplotypes_errors(hap_samples = S1,
                                                                            reads_st_en = ST_EN,
                                                                            false_variant_locs = false_variant_locs)
    _, __, false_vars = remove_false_variants(er_hap_samples, sts_ens, len(ref_H[0]))
    return false_variant_locs, false_vars


def predicted_label_with_threshold(prob_false_variance, threshold=None):
    return prob_false_variance.keys()


def get_accuracy_measures(ref_length = 300, coverage=20, read_length=10,
                          std_read_length=2, false_variance_density=0.1,
                          threshold=None):
    false_variant_locs, prob_false_variance = return_false_variant_prob(ref_length = ref_length,
                                                                        coverage=coverage,
                                                                        read_length=read_length,
                                                                        std_read_length=std_read_length,
                                                                         false_variance_density=false_variance_density)
    true_labels = false_variant_locs
    predicted_labels = predicted_label_with_threshold(prob_false_variance, threshold=threshold)
    true_positives = len(set(true_labels).intersection(predicted_labels))
    false_positives = len(set(predicted_labels) - set(true_labels))
    false_negatives = len(set(true_labels) - set(predicted_labels))
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)

    return precision, recall

def get_precision_recall_over_num_iterations(num_iterations):
    import pandas as pd
    precisions = []
    recalls = []
    for _ in range(num_iterations):
        precision, recall = get_accuracy_measures()
        precisions.append(precision)
        recalls.append(recall)
    pmean, rmean = np.mean(np.array(precisions)), np.mean(np.array(recalls))
    df = pd.DataFrame(index = range(len(precisions)))
    df["precision"] = precisions
    df["recall"] = recalls
    print(df)
    return pmean, rmean

def test_real_data():
    fragments_path='data/variantcalling/1_1M/fragments_1_2257.txt'
    longshot_vcf_path='data/variantcalling/1_1M/2.0.realigned_genotypes_1_1M.vcf'
    longshot_vcf_path_f='data/variantcalling/1_1M/2.1.realigned_genotypes_1_1M.vcf'
    true_variants = "data/variantcalling/1_1M/chr20.GIAB_highconfidencecalls_1_1M.vcf"
    high_confidence_bed = "data/variantcalling/1_1M/chr20.GIAB_highconfidenceregions_1_1M.bed"
    _, fragments, quals = read_fragments(fragments_path)
    callset = allel.read_vcf(longshot_vcf_path)
    filter = callset["variants/FILTER_PASS"]
    ls_01 = np.all(np.equal(callset['calldata/GT'], [0,1]), axis=2).T[0]
    ls_10 = np.all(np.equal(callset['calldata/GT'], [1,0]), axis=2).T[0]
    ls_hetero = ls_01 | ls_10
    filter = filter & ls_hetero
    fragments_filter, quals_filter = fragments[:, filter], quals[:, filter]
    reads, st_en = cluster_fragments(*compress_fragments(fragments_filter, quals_filter))
    fragments_filter, quals_filter = fragments[:, filter], quals[:, filter]
    reads, st_en = cluster_fragments(*compress_fragments(fragments_filter, quals_filter))
    reads, st_en, false_vars = remove_false_variants(reads, st_en, fragments_filter.shape[1], logging=True)
    false_vars = dict(sorted(list(false_vars.items()), key= lambda v:(-1*v[1][1][0], -1*v[1][1][1])))

    map_filter_index = []
    mp = {}
    cnt = 0
    for i,  v in enumerate(filter):
        if v:
            mp[cnt] = i 
            cnt += 1

    loc_gt = {}
    for k, v in false_vars.items():
        index = mp[k]
        gt = v[0]
        loc_gt[index] = gt

    with open(longshot_vcf_path_f, "r") as f:
        lines = f.readlines()

    dlines = []
    hlines = []
    for line in lines:
        if line.startswith("chr20"):
            dlines.append(line)
        else:
            hlines.append(line)

    for idx, gt in loc_gt.items():
        ld = dlines[idx].split("\t")
        ld1 = f"{gt}/{gt}" + ld[-1][3:]
        ld[-1] = ld1
        dlines[idx] = "\t".join(ld)
        
    with open(longshot_vcf_path_f, "w") as f:
        f.writelines(hlines)
        f.writelines(dlines)    
        
        