from sim_util import *
from preprocess_utils import *
from utils import *

from copy import deepcopy

def return_false_variant_prob(ref_length, coverage, read_length, std_read_length, false_variance_density):
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
    pmean, rmean = np.mean(np.array(precisions)), np.mean(np.array(recall))
    df = pd.DataFrame(index = range(len(precisions)))
    df["precision"] = precisions
    df["recall"] = recalls
    print(df)
    return pmean, rmean
    