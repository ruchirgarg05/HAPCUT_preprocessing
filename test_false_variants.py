# Create a test suite to test all the possibilities
from utils import *
from preprocess_utils import *
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
    RS, ST_EN = cluster_fragments(hap_samples, start_end)
    _, __, false_vars = remove_false_variants(RS, ST_EN, 0.02, ref_H)
    assert not len(set(false_vars) - set(false_variant_locs))