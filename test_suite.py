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
    ls_homo_00 = ls_00 
    ls_homo_11 = ls_11
    filter_het = filter & ls_hetero
    filter_homo_00 = filter & ls_homo_00
    filter_homo_11 = filter & ls_homo_11
    hetero, homo_00, homo_11 = callset["variants/POS"][filter_het], callset["variants/POS"][filter_homo_00], callset["variants/POS"][filter_homo_11]
    return set(hetero), set(homo_00), set(homo_11)


def benchmark(longshot_vcf_path = 'data/variantcalling/1M_2M/2.0.realigned_genotypes_1M_2M.vcf' , 
             longshot_vcf_path_pre = 'data/variantcalling/1M_2M/2.1.realigned_genotypes_1M_2M.vcf',
             longshot_final_path =  'data/variantcalling/1M_2M/4.0.final_genotypes_1M_2M.vcf', 
             true_variants='data/variantcalling/1M_2M/chr20.GIAB_highconfidencecalls_1M_2M.vcf', 
             high_confidence_bed='data/variantcalling/1M_2M/chr20.GIAB_highconfidenceregions_1M_2M.bed'):

    import pandas as pd

    input_het, input_homo_00, input_homo_11 = get_genotypes_from_vcf(longshot_vcf_path, high_confidence_bed)
    output_het, output_homo_00, output_homo_11 = get_genotypes_from_vcf(longshot_vcf_path_pre, high_confidence_bed)
    true_vals_het, true_vals_homo_00, true_values_homo_11 = get_genotypes_from_vcf(true_variants, high_confidence_bed)
    fin_het, fin_homo_00, fin_homo_11 = get_genotypes_from_vcf(longshot_final_path, high_confidence_bed)

    true_vals_homo = true_vals_homo_00.union(true_values_homo_11)
    
    true_vals_pass = true_vals_homo.union(true_vals_het)
    
    input_pass = input_het.union(input_homo_00).union(input_homo_11)
    
    def get_precision_recall_f1(het, homo_11, homo_00):
        tp = len(het.intersection(true_vals_het)) + len(homo_11.intersection(true_values_homo_11))
        fp = len(het - het.intersection(true_vals_het)) + len(homo_11 - homo_11.intersection(true_values_homo_11))
        fn = len(homo_00.intersection(true_values_homo_11.union(true_vals_het)))
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = tp / (tp + 0.5 * (fp + fn))
        return precision, recall, f1
    
    input_precision, input_recall, input_f1 = get_precision_recall_f1(input_het, input_homo_11, input_homo_00)
    inp_tcmp = len(input_pass.intersection(true_vals_pass))
    inp_frac_na = len(true_vals_pass - true_vals_pass.intersection(input_pass) ) / len(true_vals_pass)

    
    output_pass = output_het.union(output_homo_00).union(output_homo_11)
    output_precision, output_recall, output_f1 = get_precision_recall_f1(output_het, output_homo_11, output_homo_00)

    output_tcmp = len(output_pass.intersection(true_vals_pass))
    output_frac_na = len(true_vals_pass - true_vals_pass.intersection(output_pass) ) / len(true_vals_pass)

    
    fin_pass = fin_het.union(fin_homo_00).union(fin_homo_11)
    fin_precision, fin_recall, fin_f1 = get_precision_recall_f1(fin_het, fin_homo_11, fin_homo_00)
    fin_tcmp = len(fin_pass.intersection(true_vals_pass))
    fin_frac_na = len(true_vals_pass - true_vals_pass.intersection(fin_pass)) / len(true_vals_pass)


    matrix = np.array([[inp_tcmp, input_precision, input_recall, input_f1, inp_frac_na],
                       [output_tcmp, output_precision, output_recall, output_f1, output_frac_na],
                       [fin_tcmp, fin_precision, fin_recall, fin_f1, fin_frac_na]])
                       

    df = pd.DataFrame( matrix,
                           columns = ["tot_comparisons", "precision", "recall", "f1 score", "frac_na"], 
                           index=["input", "pre_processed", "longshot_final"]) 

    
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

    f"tabix  -h {pdir}4.0.final_genotypes.vcf.gz chr20:{s},000,000-{e},000,000 > {dir_name}/4.0.final_genotypes_{rng}.vcf",
    f"bgzip -c {dir_name}/4.0.final_genotypes_{rng}.vcf > {dir_name}/4.0.final_genotypes_{rng}.vcf.gz",
    f"tabix  -p vcf {dir_name}/4.0.final_genotypes_{rng}.vcf.gz",

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

    benchmark_df = benchmark(longshot_vcf_path = f'data/variantcalling/{rng}/2.0.realigned_genotypes_{rng}.vcf' , 
             longshot_vcf_path_pre = f'data/variantcalling/{rng}/2.1.realigned_genotypes_{rng}.vcf',
             longshot_final_path = f'data/variantcalling/{rng}/4.0.final_genotypes_{rng}.vcf',  
             true_variants= f'data/variantcalling/{rng}/chr20.GIAB_highconfidencecalls_{rng}.vcf', 
             high_confidence_bed= f'data/variantcalling/{rng}/chr20.GIAB_highconfidenceregions_{rng}.bed')
    benchmark_df.to_csv(f"data/variantcalling/{rng}/results_4.0.csv")         

#longshot_vcf_path, longshot_vcf_path_f, het_pos=POS, 
#              correctly_classified_as_het=correctly_classified_as_het, falsely_clasified_as_het=false_classified_as_het, false_homo=posis_false_homo_vars, )

def create_confusion_matrix(longshot_vcf_path, longshot_vcf_path_f, false_vars_00_01, false_vars_01_00, false_vars_11_01):
    import pandas as pd
    def get_num_per_snp(path):
        callset = allel.read_vcf(path)    
        filter = callset["variants/FILTER_PASS"]
        ls_01 = np.all(np.equal(callset['calldata/GT'], [0,1]), axis=2).T[0]
        ls_10 = np.all(np.equal(callset['calldata/GT'], [1,0]), axis=2).T[0]
        ls_00 = np.all(np.equal(callset['calldata/GT'], [0,0]), axis=2).T[0]
        ls_11 = np.all(np.equal(callset['calldata/GT'], [1,1]), axis=2).T[0]
        ls_00, ls_11 = ls_00 & filter , ls_11 & filter
        ls_hetero = ls_01 | ls_10 & filter
        num_00 = len(np.nonzero(ls_00)[0])
        num_11 = len(np.nonzero(ls_11)[0])
        num_hetero = len(np.nonzero(ls_hetero)[0])
        return num_00, num_hetero, num_11

    input_00, input_01, input_11 =  get_num_per_snp(longshot_vcf_path)
    M = [[0 for _ in range(3)] for __ in range(3)]
    M[0][0] = input_00 - len(false_vars_00_01)
    M[0][1] = len(false_vars_00_01)
    M[0][2] = 0 

    M[1][0] = len(false_vars_01_00)
    M[1][1] = input_01 - len(false_vars_01_00)
    M[1][2] = 0
    
    M[2][0] = 0 
    M[2][1] = len(false_vars_11_01)
    M[2][2] = input_11 - len(false_vars_11_01)
    df = pd.DataFrame(M, columns=["0/0", "0/1", "1/1"], index=["0/0", "0/1", "1/1"])
    f_name = os.path.join(os.path.dirname(longshot_vcf_path) , "confusion_matrix.csv") 
    print(df.to_string())
    df.to_csv(f_name)    



def write_to_vcf_like(longshot_vcf_file, positions_dict, position_vars, file_name=None):
    nlines = []
    position_vars = set(position_vars)
    with open(longshot_vcf_file, "r") as fd:
        lines = [line.split("\t") for line in fd.readlines()]

    for line in lines:
        if line[0] == "chr20":
            pos = int(line[1])
            if pos in position_vars:
                line1 = deepcopy(line)
                line1[-1] = line1[-1][:len(line[-1])-1] + "\t"
                line1.append(positions_dict[pos])
                line1.append("\n")
                nlines.append("\t".join(line1))
    import ipdb;ipdb.set_trace()
    with open(file_name, "w") as fd:
        fd.write("\n".join(nlines))    




def test_real_data(rng = "1M_2M", write_homo=False, write_hetero=True):
    import ipdb;ipdb.set_trace()
    fragments_path=f'data/variantcalling/{rng}/fragments.txt'
    longshot_vcf_path=f'data/variantcalling/{rng}/2.0.realigned_genotypes_{rng}.vcf'
    longshot_vcf_path_f=f'data/variantcalling/{rng}/2.1.realigned_genotypes_{rng}.vcf'
    longshot_vcf_path_final = f'data/variantcalling/{rng}/4.0.final_genotypes_{rng}.vcf'
    true_variants = f"data/variantcalling/{rng}/chr20.GIAB_highconfidencecalls_{rng}.vcf"
    high_confidence_bed = f"data/variantcalling/{rng}/chr20.GIAB_highconfidenceregions_{rng}.bed"
    _, fragments, quals = read_fragments(fragments_path)
    callset = allel.read_vcf(longshot_vcf_path, fields="*")
    filter = callset["variants/FILTER_PASS"]
    tcallset = allel.read_vcf(true_variants)
    ls_01 = np.all(np.equal(callset['calldata/GT'], [0,1]), axis=2).T[0]
    ls_10 = np.all(np.equal(callset['calldata/GT'], [1,0]), axis=2).T[0]
    ls_00 = np.all(np.equal(callset['calldata/GT'], [0,0]), axis=2).T[0]
    ls_11 = np.all(np.equal(callset['calldata/GT'], [1,1]), axis=2).T[0]
    ls_homo = ls_00 | ls_11
    ls_hetero = ls_01 | ls_10

    tls_01 = np.all(np.equal(tcallset['calldata/GT'], [0,1]), axis=2).T[0]
    tls_10 = np.all(np.equal(tcallset['calldata/GT'], [1,0]), axis=2).T[0]
    tls_00 = np.all(np.equal(tcallset['calldata/GT'], [0,0]), axis=2).T[0]
    tls_11 = np.all(np.equal(tcallset['calldata/GT'], [1,1]), axis=2).T[0]
    tfilter = tcallset["variants/FILTER_PASS"]
    tls_homo = tls_00 | tls_11
    tls_hetero =  tls_01 | tls_10
    tfilter_homo = tls_homo & tfilter
    tfilter_hetero = tls_hetero & tfilter

    thomo_pos = tcallset["variants/POS"][tfilter_homo]
    thetero_pos = tcallset["variants/POS"][tfilter_hetero]

    filter_homo = filter & ls_homo
    # filter should be filter & filter_hetero since the algo works using the heterozygous 
    # alleles
    filter = filter & ls_hetero
    import ipdb;ipdb.set_trace()

    false_homo = get_false_homozygous_sites(fragments, quals, filter_homo, filter)
    #false_homo = []
    import ipdb;ipdb.set_trace()
    false_homo = sorted(false_homo, key=lambda v:-1*v[1])
    posis = [x for x ,_, __ in false_homo]
    vcallset = allel.read_vcf(longshot_vcf_path)
     
    posis_false_homo = [vcallset["variants/POS"][idx] for idx in posis]
    print(f"All homo sites classified as hetero {posis_false_homo}")
    genotype_ls_vcf = ["/".join([str(v) for v in vcallset["calldata/GT"][idx][0]]) for idx in posis]
    ratio_ambig_alt_by_ref = []
    confid_ratio = []
    def sigmoid(v):
        return 1/(1+np.exp(-1*v))

    log_confid_mean = np.mean([np.log(confid) for _,__, confid in false_homo])
    import ipdb;ipdb.set_trace()
    nposis = []
    nratio = []
    ac0__ac1_am_dp = []
    labels = []
    for i, idx in enumerate(posis):
        ac, am, dp = callset["variants/AC"][idx], callset["variants/AM"][idx], callset["variants/DP"]
        pos = vcallset["variants/POS"][idx]
        if pos in tcallset["variants/POS"]:
            index = np.where(tcallset["variants/POS"] == pos)
            labels.append(True)
        label = True if pos in tcallset["variants/POS"] else False
        labels.append(label)

        ac0__ac1_am_dp.append(np.array([ac[0], ac[1], am, dp]))
        ratio = (ac[1] + am) / ac[0]
        ratio_ambig_alt_by_ref.append(str(ratio))
        logconfid = (false_homo[i][2] - log_confid_mean) /  log_confid_mean 
        summ = sigmoid(logconfid) + sigmoid(ratio)
        prod = sigmoid(logconfid) * sigmoid(ratio)
        confid_ratio.append(str( (summ, prod)) )
        if ratio > 0.5:
            nposis.append(idx)
            nratio.append(str((ratio, summ, prod)))
    import ipdb;ipdb.set_trace()
    # import sklearn
    # features = sklearn.feature_selection.f_regression( np.array(ac0__ac1_am_dp), labels, )
    nposis_false_homo = [vcallset["variants/POS"][idx] for idx in nposis]
    n_false_homo = []
    n_genotype_ls_vcf = ["/".join([str(v) for v in vcallset["calldata/GT"][idx][0]]) for idx in nposis]    
    for  v, r in zip(false_homo, ratio_ambig_alt_by_ref):
        if float(r) >0.5:
            n_false_homo.append(v) 
   
    ipdb.set_trace()
    #ratio_ambig_alt_by_ref = [[ str((ac[1] + am) / ac[0]) for ac,am in zip(][idx])  ][0] for idx in posis]
    #positions = {vcallset["variants/POS"][idx]:"\t".join([":".join([n_genotype_ls_vcf[i],"0/1"]), str(confid), str(coverage)]) for 
                    # i, (idx ,confid,  coverage) in enumerate(false_homo)}
    positions = {vcallset["variants/POS"][idx]:"\t".join([":".join([n_genotype_ls_vcf[i],"0/1"]), str(confid), str(coverage), str(nratio[i])]) for 
                    i, (idx ,confid,  coverage) in enumerate(n_false_homo)}
    tpos = tcallset["variants/POS"]
    posis_false_homo_in_tv = set(nposis_false_homo).intersection(tpos)
    
    posis_false_homo_vars = set(nposis_false_homo) - set(nposis_false_homo).intersection(tpos)
    print(f"Homo sites which are not present in true_variants {posis_false_homo_vars}")
    # Thse variants were originally classified as homo, but are classified as hetero and is hetero
    correctly_classified_as_het = set(posis_false_homo_in_tv).intersection(thetero_pos)
    print(f"Homo sites correctly classified as hetero {correctly_classified_as_het}")
    # These variants are originally classified as homo and predicted as hetero, but are actually homo itself. 
    false_classified_as_het = set(posis_false_homo_in_tv).intersection(thomo_pos)
 
    # These variants are not present : False homo variants
    # posis_false_homo_vars = set(posis_false_homo) - set(posis_false_homo).intersection(tpos)
    # print(f"Homo sites which are not present in true_variants {posis_false_homo_vars}")
    # # Thse variants were originally classified as homo, but are classified as hetero and is hetero
    # correctly_classified_as_het = set(posis_false_homo_in_tv).intersection(thetero_pos)
    # print(f"Homo sites correctly classified as hetero {correctly_classified_as_het}")
    # # These variants are originally classified as homo and predicted as hetero, but are actually homo itself. 
    # false_classified_as_het = set(posis_false_homo_in_tv).intersection(thomo_pos)
 



    print(f"homo sites which are classified as hetero, but is homo {false_classified_as_het}")
    
    

    write_to_vcf_like(longshot_vcf_path, positions, correctly_classified_as_het, f'data/variantcalling/{rng}/homo_correct_classified_hetero.txt')
    write_to_vcf_like(longshot_vcf_path, positions, false_classified_as_het, f'data/variantcalling/{rng}/homo_false_classified_hetero.txt')
    write_to_vcf_like(longshot_vcf_path, positions, posis_false_homo_vars,  f'data/variantcalling/{rng}/false_homo_vars.txt')
    

    ipdb.set_trace()
    fragments_filter, quals_filter = fragments[:, filter], quals[:, filter]
    reads, st_en = cluster_fragments(*compress_fragments(fragments_filter, quals_filter))
    reads, st_en, false_vars = remove_false_variants(reads, st_en, fragments_filter.shape[1], logging=True)
    
    false_vars = dict(sorted(list(false_vars.items()), key= lambda v:(-1*v[1][1][0], -1*v[1][1][1])))
    ipdb.set_trace()

    mp, loc_gt = {}, {}
    cnt = 0
    for i,  v in enumerate(filter):
        if v:
            mp[cnt] = i 
            cnt += 1
    for k, v in false_vars.items():
        index = mp[k]
        gt = v[0]
        loc_gt[index] = gt

    POS = [str(callset["variants/POS"][idx]) for idx in loc_gt]
    print(f"Hetero sites classified as homo {POS}")
    
    false_vars_00_01 = [vcallset["variants/POS"][idx] for idx in nposis if len(np.nonzero(vcallset["calldata/GT"][idx][0])[0])==0]
    false_vars_11_01 = [vcallset["variants/POS"][idx] for idx in nposis if len(np.nonzero(vcallset["calldata/GT"][idx][0])[0])==2]
    
    
    false_vars_01_00 = POS
    
    create_confusion_matrix(longshot_vcf_path, longshot_vcf_path_f, false_vars_00_01, false_vars_01_00, false_vars_11_01)
    with open(longshot_vcf_path, "r") as f:
        lines = f.readlines()

    dlines = []
    hlines = []
    for line in lines:
        if line.startswith("chr20"):
            dlines.append(line)
        else:
            hlines.append(line)
    
    false_var_idxs = set(list(loc_gt.keys()))
    if write_hetero:
        dlines = [ln for i, ln in enumerate(dlines) if i not in false_var_idxs]
    if write_homo:
        import ipdb;ipdb.set_trace()
        ndlines = []
        for line in dlines:
            ln = line.split("\t")
            if int(ln[1]) in nposis_false_homo:
                gt = "0/1" + ln[-1][3:]
                ln[-1] = gt
                ndlines.append("\t".join(ln))
            else:
                ndlines.append(line)          
        dlines = ndlines 

    with open(longshot_vcf_path_f, "w") as f:
        f.writelines(hlines)
        f.writelines(dlines)

    benchmark_df = benchmark(longshot_vcf_path, longshot_vcf_path_f, longshot_vcf_path_final, true_variants, high_confidence_bed)
    benchmark_df.to_csv(f"data/variantcalling/{rng}/results.csv")
        