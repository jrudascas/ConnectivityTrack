__author__ = 'brain'

import warnings

warnings.filterwarnings("always")

import core as p
import definitions as d
import nibabel as nib


def processing(path_dwi_input, path_binary_mask, path_output, path_bval, path_bvec, mapping):
    f_tensor_fitevecs, f_tensor_fitevals = p.to_estimate_dti(path_dwi_input, path_binary_mask, path_output, path_bval,
                                                             path_bvec)

    list_maps = p.to_estimate_dti_maps(path_dwi_input, path_output, f_tensor_fitevecs, f_tensor_fitevals)

    # p.to_generate_tractography(path_dwi_input, path_binary_mask, path_output, path_bval, path_bvec)

    affine = nib.load(path_dwi_input).affine
    list_path_atlas_1 = p.registration_atlas_to(d.aan_atlas, path_output, affine, mapping)
    list_path_atlas_2 = p.registration_atlas_to(d.morel_atlas, path_output, affine, mapping)
    list_path_atlas_3 = p.registration_atlas_to(d.hypothalamus_atlas, path_output, affine, mapping)
    list_path_atlas_4 = p.registration_atlas_to(d.harvard_oxford_cort_atlas, path_output, affine, mapping)

    indexHarvardOxfortCortical = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                                  24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45,
                                  46, 47]

    atlas_dict = {'Morel': list_path_atlas_2, 'AAN': list_path_atlas_1, 'HarvardOxfordCort': list_path_atlas_4,
                  'HypothalamusAtlas': list_path_atlas_3}

    bunddle_rules = [(('AAN', [1, 3]), ('Morel', [4, 18, 42, 56]), ('HarvardOxfordCort', indexHarvardOxfortCortical)),
             (('AAN', [1, 3]), ('HypothalamusAtlas', [1, 2, 3]), ('HarvardOxfordCort', indexHarvardOxfortCortical)),
             (('Morel', [4, 18, 42, 56]), ('HarvardOxfordCort', indexHarvardOxfortCortical))]

    list_bunddle = p.to_generate_bunddle(path_dwi_input, path_output, path_binary_mask, path_bval, path_bvec, bunddle_rules, atlas_dict)

    roi_rules = {'AAN':[1, 3], 'Morel':[4, 18, 42, 56], 'HypothalamusAtlas':[1, 2, 3]}
    features_list = p.to_generate_report_aras(list_bunddle, list_maps, roi_rules, atlas_dict)

    return features_list