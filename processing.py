__author__ = 'brain'

import warnings
warnings.filterwarnings("always")

import core as p
import definitions as d


def processing(path_dwi_input, path_binary_mask, path_output, path_bval, path_bvec):
    f_tensor_fitevecs, f_tensor_fitevals = p.to_estimate_dti(path_dwi_input, path_binary_mask, path_output, path_bval,
                                                             path_bvec)

    p.to_estimate_dti_maps(path_dwi_input, path_output, f_tensor_fitevecs, f_tensor_fitevals)

    #p.to_generate_tractography(path_dwi_input, path_binary_mask, path_output, path_bval, path_bvec)

    indexHarvardOxfortCortical = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                                  24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45,
                                  46, 47]

    atlas_dict = {'Morel': d.morel_atlas, 'AAN': d.aan_atlas, 'HarvardOxfordCort': d.harvard_oxford_cort_atlas,
                  'HypothalamusAtlas': d.hypothalamus_atlas}
    # (('AAN', [1, 2, 3, 4, 5, 6, 7, 8, 9]),('Morel', [4, 18, 42, 56])),

    rules = [(('AAN', [1, 3]), ('Morel', [4, 18, 42, 56]), ('HarvardOxfordCort', indexHarvardOxfortCortical)),
             (('AAN', [1, 3]), ('HypothalamusAtlas', [1, 2, 3]), ('HarvardOxfordCort', indexHarvardOxfortCortical)),
             (('Morel', [4, 18, 42, 56]), ('HarvardOxfordCort', indexHarvardOxfortCortical))]

    p.to_generate_bunddle(path_dwi_input, path_output, path_binary_mask, path_bval, path_bvec, rules, atlas_dict)
