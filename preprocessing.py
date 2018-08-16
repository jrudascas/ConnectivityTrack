__author__ = 'Jrudascas'

import warnings
warnings.filterwarnings("always")

import core as c
import definitions as d


def preprocessing(path_dwi_input, path_out, path_bvec, path_bval):
    # bvals, bvecs = read_bvals_bvecs(bval_path, bvec_path)
    # ref_bo = str(np.where(gradient_table(bvals, bvecs).b0s_mask == True)[0])
    ref_bo = str(0)

    process = {}

    process['pathEddy'] = c.eddy_correction(path_dwi_input, path_out, ref_bo)
    process['pathNonLocalMean'] = c.nonLocalMean(process['pathEddy'], path_out)
    process['pathReslicing'] = c.reslicing(process['pathNonLocalMean'], path_out, d.vox_sz)
    # maskedVolume, binaryMask = processData.medianOtsu(process[process.__len__() - 1], outPath)
    dwi_masked, binary_mask, b0_masked = c.betDWI(process['pathReslicing'], path_out)

    process['pathDWIMasked'] = dwi_masked
    process['pathBinaryMask'] = binary_mask
    process['pathb0Masked'] = b0_masked

    path_normalized, mapping = c.to_register_dwi_to_mni(process['pathDWIMasked'], path_out, path_bvec, path_bval)

    process['pathNormalized'] = path_normalized
    process['mapping_b0_to_NMI'] = mapping

    return process
