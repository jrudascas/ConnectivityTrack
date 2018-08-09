__author__ = 'Jrudascas'

import nibabel as nib
from dipy.io import read_bvals_bvecs
import os
import fsl_wrapper as fsl
import preprocessing as pre
import processing as pro
import utils as ut
import definitions as d
import core as p
import utils as utils
from dipy.core.geometry import vector_norm

def Main (path_in, path_out):
    import numpy as np
    print('...................................')
    print('        Starting Processing        ')
    print('...................................')
    print(' ')

    lstDir = os.walk(path_in)
    validExtentions = ['bvec', 'bval', 'nii', 'gz']
    files_found = {}

    for root, dirs, files in lstDir:
        for fichero in files:
            (file_name, extension) = os.path.splitext(fichero)
            fullPath = path_in + file_name + extension
            if utils.toValidateExtention(fullPath, validExtentions):
                files_found[utils.whatKindFileIs(fullPath)] = fullPath

    print('Were found these files and will be processed: ')

    for key, value in files_found.items():
        print('    - ' + key + ': ' + value)

    print(' ')
    print('...................................')
    print('       running Pre-processing      ')
    print('...................................')
    print(' ')

    img = nib.load(files_found['dwi'])
    data = img.get_data()
    affine = img.affine
    bvals, bvecs = read_bvals_bvecs(files_found['bval'], files_found['bvec'])
    bvecs = np.where(np.isnan(bvecs), 0, bvecs)
    bvecs_close_to_1 = abs(vector_norm(bvecs) - 1) <= 0.1

    bvecs_close_to_1[..., 0] = True
    bvals = bvals[bvecs_close_to_1]
    bvecs = bvecs[bvecs_close_to_1]
    data = data[..., bvecs_close_to_1]

    if not np.all(bvecs_close_to_1) == True:
        np.savetxt(files_found['bval'], bvals, delimiter='    ', fmt='%s')
        np.savetxt(files_found['bvec'], bvecs, delimiter='    ', fmt='%s')
        nib.save(nib.Nifti1Image(data, affine), files_found['dwi'])

    if os.path.exists(d.path_temporal):
        ut.delete_Files(d.path_temporal)
        os.removedirs(d.path_temporal)

    os.mkdir(d.path_temporal)

    if 't1' in files_found:
        print()
        print('-> Starting preprocessing of structural image')
        print()
        refName = utils.extractFileName(files_found['t1'])

        if not (os.path.exists(path_out + refName + '_BET.nii')):
            fsl.BET(files_found['t1'], path_out + refName + '_BET.nii', '-f .4')

        if not(os.path.exists(path_out + refName + '_Normalized.nii')):
           warped_t1, MNI_T2_affine, mapping_t1 = p.registrationtoNMI(path_out + refName + '_BET.nii.gz', path_out)
           nib.save(nib.Nifti1Image(warped_t1.astype(np.float32), MNI_T2_affine), path_out + refName + '_Normalized.nii')

        print('-> Ending preprocessing of structural image')
        print()

    print()
    print('-> Starting preprocessing of diffution image')

    preprocessing_output = pre.preprocessing(files_found['dwi'], path_out, files_found['bvec'], files_found['bval'])

    print('-> Ending preprocessing of diffution image')

    print(' ')
    print('....................................')
    print('        running Processing          ')
    print('....................................')
    print(' ')

    pro.processing(preprocessing_output['pathNormalized'], d.brain_mask_nmi, path_out, files_found['bval'], files_found['bvec'])

    #print("Building connectivity matrix " + time.strftime("%H:%M:%S") )
    #M = p.connectivity_matrix2(streamlines, warped_atlas, affine=streamlines_affine, shape=mask_data.shape)
    #M[:1, :] = 0
    #M[:, :1] = 0

    #np.savetxt(pathOut + 'connectivity.out', M, delimiter=' ', fmt='%s')
    #print("End Building connectivity matrix " + time.strftime("%H:%M:%S") )

    print('...................................')
    print('         Ending Processing         ')
    print('...................................')
    print(' ')