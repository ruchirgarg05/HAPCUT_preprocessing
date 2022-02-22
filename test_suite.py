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

def get_genotypes_from_vcf(vcf_file, bed=None):
    callset = allel.read_vcf(vcf_file)
    filter = callset["variants/FILTER_PASS"]
    if bed:
        filter = filter & get_bed_mask(bed, callset["variants/POS"])
    ls_01 = np.all(np.equal(callset['calldata/GT'], [0,1]), axis=2).T[0]
    ls_10 = np.all(np.equal(callset['calldata/GT'], [1,0]), axis=2).T[0]
    ls_00 = np.all(np.equal(callset['calldata/GT'], [0,0]), axis=2).T[0]
    ls_11 = np.all(np.equal(callset['calldata/GT'], [1,1]), axis=2).T[0]
    ls_hetero = ls_01 | ls_10
    ls_homo = ls_00 | ls_11
    filter_het = filter & ls_hetero
    filter_homo = filter & ls_homo
    hetero, homo = callset["variants/POS"][filter_het], callset["variants/POS"][filter_homo]
    return set(hetero), set(homo)


def benchmark(longshot_vcf_path = 'data/variantcalling/1M_2M/2.0.realigned_genotypes_1M_2M.vcf' , 
             longshot_vcf_path_pre = 'data/variantcalling/1M_2M/2.1.realigned_genotypes_1M_2M.vcf', 
             true_variants='data/variantcalling/1M_2M/chr20.GIAB_highconfidencecalls_1M_2M.vcf', 
             high_confidence_bed='data/variantcalling/1M_2M/chr20.GIAB_highconfidenceregions_1M_2M.bed'):
    import pandas as pd
    input_het, input_homo = get_genotypes_from_vcf(longshot_vcf_path, high_confidence_bed)
    output_het, output_homo = get_genotypes_from_vcf(longshot_vcf_path_pre, high_confidence_bed)
    true_vals_het, true_vals_homo = get_genotypes_from_vcf(true_variants, high_confidence_bed)
    true_vals_pass = true_vals_het.union(true_vals_homo) 
    input_pass = input_het.union(input_homo)

    import ipdb;ipdb.set_trace()
    callset = allel.read_vcf(longshot_vcf_path)
    ls = list(callset["variants/POS"])
    # misclassified
    idxs = [ls.index(v) for v in true_vals_het.intersection(input_homo)]
    print(idxs)
    idxs_homo_fp = set(input_homo) - set(true_vals_pass)
    idxs_homo_fp_idxs = [ls.index(v) for v in idxs_homo_fp]
    print(idxs_homo_fp_idxs)
    idxs = [ls.index(v) for v in set(input_homo).intersection(true_vals_pass)]
    idx_common = [ls.index(v) for v in    true_vals_pass.intersection(input_homo)]
    print(idx_common)

    # All the sites that are heterozygous that are present in true_vcf
    inp_present_in_tv = true_vals_pass.intersection(input_het)

    true_labels = true_vals_het.intersection(input_pass)
    predicted_labels = input_het
    true_positives = len(set(true_labels).intersection(predicted_labels))
    false_positives = len(set(predicted_labels) - set(true_labels))
    false_negatives = len(set(input_homo) .intersection( set(true_labels)))
    input_precision = true_positives / (true_positives + false_positives)
    input_recall = true_positives / (true_positives + false_negatives)
    input_f1 = true_positives / (true_positives + 0.5*(false_positives + false_negatives))
    input_fra_na = len(input_het - input_het.intersection(true_vals_pass)) /  len(input_het.intersection(true_vals_pass))
    input_tot_comp = len(inp_present_in_tv)

    predicted_labels = output_het
    
    out_present_in_tv = true_vals_pass.intersection(output_het)
    true_positives = len(set(true_labels).intersection(predicted_labels))
    false_positives = len(set(predicted_labels) - set(true_labels))
    false_negatives = len(set(output_homo) .intersection( set(true_labels)))
    #false_negatives = len(set(true_labels) - set(predicted_labels))
    output_precision = true_positives / (true_positives + false_positives)
    output_recall = true_positives / (true_positives + false_negatives)
    output_f1 = true_positives / (true_positives + 0.5*(false_positives + false_negatives))
    output_fra_na = len(output_het - output_het.intersection(true_vals_pass)) /  len(output_het.intersection(true_vals_pass))
    output_tot_comp = len(out_present_in_tv)

    matrix = [[input_tot_comp, input_precision, input_recall, input_f1, input_fra_na],
               [output_tot_comp, output_precision, output_recall, output_f1, output_fra_na]]
    df = pd.DataFrame( matrix,
                           columns = ["tot_comparisons", "precision", "recall", "f1 score", "frac_na"], 
                           index=["input", "pre_processed"])
    print(df.to_string())
    return df

def get_index_for_fragment_path(frag_vcf, frag_file="data/variantcalling/fragments.txt", 
                                full_vcf="data/variantcalling/2.0.realigned_genotypes.vcf"):
    import ipdb;ipdb.set_trace()
    callset = allel.read_vcf(full_vcf)
    POS = callset["variants/POS"]
    callset_frag = allel.read_vcf(frag_vcf)
    POS_fragment = callset_frag["variants/POS"]
    index_st =  np.where(POS == POS_fragment[0])[0][0]
    index_en = index_st + POS_fragment.shape[0]
    fn = filter_fragment_for_range(frag_file, index_st, index_en)
    frag_file = "/".join(frag_vcf.split("/")[:-1]) + "/fragments.txt"
    import ipdb;ipdb.set_trace()
    os.rename(fn, frag_file)
    _, frags, __ = read_fragments(frag_file)
    assert frags.shape[1] == POS_fragment.shape[0]


def script_create_data_fragments(s, e, create=False):
    def run_cmd(cmd):
        import os
        os.system(cmd)
        # import subprocess
        # process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
        # output, error = process.communicate()
    import ipdb;ipdb.set_trace()
    rng = f"{s}M_{e}M"
    pdir = "data/variantcalling/"
    dir_name = f'{pdir}{rng}'
    cmds =[
    f"rm -r {dir_name}",    
    f"mkdir {dir_name}",
    f"tabix  -h {pdir}2.0.realigned_genotypes.vcf.gz chr20:{s},000,000-{e},000,000 > {dir_name}/2.0.realigned_genotypes_{rng}.vcf",
    f"bgzip -c {dir_name}/2.0.realigned_genotypes_{rng}.vcf > {dir_name}/2.0.realigned_genotypes_{rng}.vcf.gz",
    f"tabix  -p vcf {dir_name}/2.0.realigned_genotypes_{rng}.vcf.gz",

    f"tabix  -h {pdir}chr20.GIAB_highconfidencecalls.vcf.gz chr20:{s},000,000-{e},000,000 > {dir_name}/chr20.GIAB_highconfidencecalls_{rng}.vcf",
    f"bgzip -c {dir_name}/chr20.GIAB_highconfidencecalls_{rng}.vcf > {dir_name}/chr20.GIAB_highconfidencecalls_{rng}.vcf.gz",
    f"tabix -p vcf {dir_name}/chr20.GIAB_highconfidencecalls_{rng}.vcf.gz",

    f"tabix  -h {pdir}chr20.GIAB_highconfidenceregions.bed.gz chr20:{s},000,000-{e},000,000 > {dir_name}/chr20.GIAB_highconfidenceregions_{rng}.bed",
    f"bgzip -c {dir_name}/chr20.GIAB_highconfidenceregions_{rng}.bed > {dir_name}/chr20.GIAB_highconfidenceregions_{rng}.bed.gz",
    f"tabix -p bed {dir_name}/chr20.GIAB_highconfidenceregions_{rng}.bed.gz"]
    if create:
        for cmd in cmds:
            run_cmd(cmd)
    ipdb.set_trace()        
    get_index_for_fragment_path(f'data/variantcalling/{rng}/2.0.realigned_genotypes_{rng}.vcf')
    test_real_data(rng)

   

def test_script_final_genotypes(s, e, create=False):
    import ipdb;ipdb.set_trace()
    def run_cmd(cmd):
        import os
        os.system(cmd)

    rng = f"{s}M_{e}M"
    pdir = "data/variantcalling/"
    dir_name = f'{pdir}{rng}'
    cmds =[
    f"tabix  -h {pdir}4.0.final_genotypes.vcf.gz chr20:{s},000,000-{e},000,000 > {dir_name}/4.0.final_genotypes_{rng}.vcf",
    f"bgzip -c {dir_name}/4.0.final_genotypes_{rng}.vcf > {dir_name}/4.0.final_genotypes_{rng}.vcf.gz",
    f"tabix  -p vcf {dir_name}/4.0.final_genotypes_{rng}.vcf.gz",
    ]
    if create:
        for cmd in cmds:
            run_cmd(cmd)

    benchmark_df = benchmark(longshot_vcf_path = f'data/variantcalling/{rng}/4.0.final_genotypes_{rng}.vcf' , 
             longshot_vcf_path_pre = f'data/variantcalling/{rng}/2.1.realigned_genotypes_{rng}.vcf', 
             true_variants= f'data/variantcalling/{rng}/chr20.GIAB_highconfidencecalls_{rng}.vcf', 
             high_confidence_bed= f'data/variantcalling/{rng}/chr20.GIAB_highconfidenceregions_{rng}.bed')
    benchmark_df.to_csv(f"data/variantcalling/{rng}/results_4.0.csv")         




def test_real_data(rng = "1M_2M"):
    import ipdb;ipdb.set_trace()
    fragments_path=f'data/variantcalling/{rng}/fragments.txt'
    longshot_vcf_path=f'data/variantcalling/{rng}/2.0.realigned_genotypes_{rng}.vcf'
    longshot_vcf_path_f=f'data/variantcalling/{rng}/2.1.realigned_genotypes_{rng}.vcf'
    true_variants = f"data/variantcalling/{rng}/chr20.GIAB_highconfidencecalls_{rng}.vcf"
    high_confidence_bed = f"data/variantcalling/{rng}/chr20.GIAB_highconfidenceregions_{rng}.bed"
    _, fragments, quals = read_fragments(fragments_path)
    callset = allel.read_vcf(longshot_vcf_path)
    filter = callset["variants/FILTER_PASS"]
    ls_01 = np.all(np.equal(callset['calldata/GT'], [0,1]), axis=2).T[0]
    ls_10 = np.all(np.equal(callset['calldata/GT'], [1,0]), axis=2).T[0]
    ls_00 = np.all(np.equal(callset['calldata/GT'], [0,0]), axis=2).T[0]
    ls_11 = np.all(np.equal(callset['calldata/GT'], [1,1]), axis=2).T[0]
    ls_homo = ls_00 | ls_11
    ls_hetero = ls_01 | ls_10

    filter_homo = filter & ls_homo
    # filter should be filter & filter_hetero since the algo works using the heterozygous 
    # alleles
    filter = filter & ls_hetero

    false_homo = get_false_homozygous_sites(fragments, quals, filter_homo, filter)
    false_homo = sorted(false_homo, key=lambda v:-1*v[1])
    ipdb.set_trace()
    #[1562, 1563, 429, 749, 1860, 1902, 2174, 1687, 1863, 1864, 1975, 1626, 1680]
    fragments_filter, quals_filter = fragments[:, filter], quals[:, filter]
    reads, st_en = cluster_fragments(*compress_fragments(fragments_filter, quals_filter))
    # fv = [854, 132, 754, 198, 26, 199, 477, 613, 117, 2, 767, 11, 39, 761, 35, 536, 228]
    # for v in fv:
    #     import ipdb;ipdb.set_trace()
    #     visualize_overlapping_reads_at(reads, st_en, v
    reads, st_en, false_vars = remove_false_variants(reads, st_en, fragments_filter.shape[1], logging=True)
    
    false_vars = dict(sorted(list(false_vars.items()), key= lambda v:(-1*v[1][1][0], -1*v[1][1][1])))
    # 1-1M: 1-2257
    # 1M-2M 2257:4468
    # 2M-3M 4468:6538
    # 3M-4M 6538:8390
    # 4M-5M 8390:10762

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

    callset = allel.read_vcf(longshot_vcf_path)
    POS = [str(callset["variants/POS"][idx]) for idx in loc_gt]
    print(POS)        

    with open(longshot_vcf_path, "r") as f:
        lines = f.readlines()

    dlines = []
    hlines = []
    for line in lines:
        if line.startswith("chr20"):
            dlines.append(line)
        else:
            hlines.append(line)
    
    for pos, (idx, gt) in zip(POS, loc_gt.items()):
        ld = dlines[idx].split("\t")            
        # ld1 = f"{gt}/{gt}" + ld[-1][3:]
        # ld[-1] = ld1
        #dlines[idx] = "\t".join(ld)
        # Assert positions are the same
        assert ld[1] == pos

    #import ipdb;ipdb.set_trace() 
    # ndlines = []
    false_var_idxs = set(list(loc_gt.keys()))
    # for i, ln in enumerate(dlines):
    #     if i in false_var_idxs:
    #         continue
    #     ndlines.append(ln)
    # dlines = ndlines
    dlines = [ln for i, ln in enumerate(dlines) if i not in false_var_idxs]    
    with open(longshot_vcf_path_f, "w") as f:
        f.writelines(hlines)
        f.writelines(dlines)    
    # {1877: 1, 479: 1, 1698: 1, 587: 0, 104: 0, 588: 1, 1039: 1, 1482: 1, 427: 1, 2: 0, 1740: 1, 23: 1, 289: 0, 1707: 0, 248: 1, 1119: 0, 644: 1}
    # ['1877119', '1302364', '1757678', '1352184', '1048931', '1352200', '1574507', '1684406', '1278060', '1000490', '1799385', '1008797',
    #  '1150139', '1766116', '1125851', '1577470', '1383746']
    benchmark_df = benchmark(longshot_vcf_path, longshot_vcf_path_f, true_variants, high_confidence_bed)
    benchmark_df.to_csv(f"data/variantcalling/{rng}/results.csv")

    # with open(true_variants, "r") as f:
    #     lines = f.readlines()
    # correct = []
    # false = []
    # dlines = []
    # hlines = []
    # for line in lines:
    #     if line.startswith("chr20"):
    #         dlines.append(line)
    #     else:
    #         hlines.append(line)

    # tp, fp = 0, 0
    # for dline in dlines:
    #     pos = dline.split("\t")[1]
    #     if pos in POS:
    #         genotype = dline.split("\t")[-1][:3]
    #         print(pos, genotype)
    #         if genotype == "1/1" or genotype == "0/0":
    #             tp += 1
    #         else:
    #             fp += 1                

        