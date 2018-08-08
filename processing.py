__author__ = 'brain'

import core as p


def processing(path_dwi_input, path_binary_mask, path_out, path_bval, path_bvec):
    f_tensor_fitevecs, f_tensor_fitevals = p.to_estimate_dti(path_dwi_input, path_binary_mask, path_out, path_bval,
                                                             path_bvec)

    p.to_estimate_dti_maps(path_dwi_input, path_binary_mask, f_tensor_fitevecs, f_tensor_fitevals, path_out,
                           path_bvec, path_bval)
    p.to_generate_tractography(path_dwi_input, path_binary_mask, path_out, path_bval, path_bvec)

    #dictAtlas = {'Morel': morelAtlas, 'AAN': aanAtlas, 'HarvardOxfordCort': harvardOxfordCortAtlas,
    #             'HypothalamusAtlas': hypothalamusAtlas}
    # (('AAN', [1, 2, 3, 4, 5, 6, 7, 8, 9]),('Morel', [4, 18, 42, 56])),

    #listRules = [(('AAN', [1, 3]), ('Morel', [4, 18, 42, 56]), ('HarvardOxfordCort', indexHarvardOxfortCortical)),
    #             (('AAN', [1, 3]), ('HypothalamusAtlas', [1, 2, 3]), ('HarvardOxfordCort', indexHarvardOxfortCortical)),
    #             (('Morel', [4, 18, 42, 56]), ('HarvardOxfordCort', indexHarvardOxfortCortical))]

    #p.to_generate_bunddle(listRules, dictAtlas, dataDWI, largeMask, gtab, affineDWI, affineROI)
