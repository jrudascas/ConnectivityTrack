__author__ = 'Jrudascas'

import core as processData
import definitions as d
from dipy.core.gradients import gradient_table
from dipy.io import read_bvals_bvecs
import numpy as np

import core as p


def preprocessing(path_dwi_input, path_out, path_bvec, path_bval):
    # bvals, bvecs = read_bvals_bvecs(bval_path, bvec_path)
    # ref_bo = str(np.where(gradient_table(bvals, bvecs).b0s_mask == True)[0])
    ref_bo = str(0)

    process = {}

    process['pathEddy'] = processData.eddy_correction(path_dwi_input, path_out, ref_bo)
    process['pathNonLocalMean'] = processData.nonLocalMean(process['pathEddy'], path_out)
    process['pathReslicing'] = processData.reslicing(process['pathNonLocalMean'], path_out, d.vox_sz)
    # maskedVolume, binaryMask = processData.medianOtsu(process[process.__len__() - 1], outPath)
    maskedVolume, binaryMask = processData.betDWI(process['pathReslicing'], path_out)

    process['pathDWIMasked'] = maskedVolume
    process['pathBinaryMask'] = binaryMask

    warped_dwi, MNI_T2_affine, mapping_dwi, path_normalized = p.to_register_dwi_to_mni(process['pathDWIMasked'], path_out,
                                                                                       path_bvec, path_bval)

    process['pathNormalized'] = path_normalized
    process['mappingToNMI'] = mapping_dwi

    return process
