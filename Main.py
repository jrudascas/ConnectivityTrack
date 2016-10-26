import numpy as np
import nibabel as nib
from dipy.io import read_bvals_bvecs
import os
import FSLWrapper as fsl
import Preprocessing as preprocessing
import Processing as processing
import Utils as ut
import Definitions as definitions
import ProcessData as p
import Utils as utils
from dipy.core.geometry import vector_norm

print('...................................')
print('    Inicia procesamiento de DTI    ')
print('...................................')
print(' ')

lstFiles = []
lstDir = os.walk(definitions.Definitions.pathIN)

fdwiName1 = ""
fdwiName2 = ""
ext1 = ""
ext2 = ""

for root, dirs, files in lstDir:
    for fichero in files:
        (nombreFichero, extension) = os.path.splitext(fichero)
        if((extension == ".nii") | (extension == ".gz")):

            if ((fdwiName1 == "")):
                ext1 = extension
                fdwiName1 = nombreFichero
            elif (fdwiName2 == ""):
                fdwiName2 = nombreFichero
                ext2 = extension
        if(extension == ".bvec"):
            fbvec = definitions.Definitions.pathIN + nombreFichero + extension
            refName = nombreFichero
        if(extension == ".bval"):
            fbval = definitions.Definitions.pathIN + nombreFichero + extension
            refName = nombreFichero

if (refName != fdwiName1):
    fdwiT1 = definitions.Definitions.pathIN + fdwiName1 + ext1
    fdwi   = definitions.Definitions.pathIN + fdwiName2 + ext2
else:
    fdwiT1 = definitions.Definitions.pathIN + fdwiName2 + ext2
    fdwi   = definitions.Definitions.pathIN + fdwiName1 + ext1

print('Se procesaran los siguientes arhivos: ')
print(nombreFichero)

print(' ')
print('...................................')
print('          Preprocesamiento         ')
print('...................................')
print(' ')

img = nib.load(fdwi)
data = img.get_data()
affine = img.get_affine()
bvals, bvecs = read_bvals_bvecs(fbval, fbvec)
dwi_mask = bvals > 0
bvecs = np.where(np.isnan(bvecs), 0, bvecs)
bvecs_close_to_1 = abs(vector_norm(bvecs) - 1) <= 0.1

bvecs_close_to_1[..., 0] = True
bvals = bvals[bvecs_close_to_1]
bvecs = bvecs[bvecs_close_to_1]
data = data[..., bvecs_close_to_1]

if not np.all(bvecs_close_to_1) == True:
    np.savetxt(fbval, bvals, delimiter='    ', fmt='%s')
    np.savetxt(fbvec, bvecs, delimiter='    ', fmt='%s')
    nib.save(nib.Nifti1Image(data, affine), fdwi)

if os.path.exists(definitions.Definitions.tempPath):
    ut.delete_Files(definitions.Definitions.tempPath)
    os.removedirs(definitions.Definitions.tempPath)

os.mkdir(definitions.Definitions.tempPath)

if (fdwiName2 != ""):
    print('-> Preprocesado image estructural')
    fsl.BET(fdwiT1, definitions.Definitions.pathOUT + refName + '_Structural_BET.nii', '-f .4')

    if not(os.path.exists(definitions.Definitions.pathOUT + refName + '_Structural_Normalized.nii')):
       warped_Structural, MNI_T2_affine, mapping_Structural = p.registrationtoNMI(definitions.Definitions.pathOUT + refName + '_Structural_BET.nii.gz', definitions.Definitions.pathOUT)
       nib.save(nib.Nifti1Image(warped_Structural.astype(np.float32), MNI_T2_affine), definitions.Definitions.pathOUT + refName + '_Structural_Normalized.nii')

listProcess = preprocessing.Preprocessing_DWI(fdwi, definitions.Definitions.pathOUT, 0)

mask = nib.load(listProcess[listProcess.__len__() - 1])
mask_data = (mask.get_data() == 0)

dwiPreprocessed = nib.load(listProcess[listProcess.__len__() - 3])
dwiPreprocessed_data = dwiPreprocessed.get_data()

dwiPreprocessed_data[mask_data,:] = 0

ni_b0 = nib.Nifti1Image(dwiPreprocessed_data, dwiPreprocessed.get_affine())
refNameOnly = utils.extractFileName(listProcess[listProcess.__len__() - 3])
filename = definitions.Definitions.pathOUT + refNameOnly + '_DWImask.nii.gz'
fggg = filename
ni_b0.to_filename(filename)

warped, MNI_T2_affine, mapping = p.registrationDWItoNMI(filename, fbvec, fbval, definitions.Definitions.pathOUT)

print(' ')
print('...................................')
print('           Procesamiento           ')
print('...................................')
print(' ')
print('--> Calculando el modelo por tensor')

streamlines, streamlines_affine = processing.Processing_DWI(listProcess[listProcess.__len__() - 3], listProcess[listProcess.__len__() - 1], definitions.Definitions.pathOUT, fbval, fbvec, mapping, MNI_T2_affine)

filename = definitions.Definitions.pathOUT + utils.extractFileName(listProcess[listProcess.__len__() - 2]) + '_MedianOtsu_b0.nii.gz'

p.registerAffine_atlas(definitions.Definitions.atlas, definitions.Definitions.standard, definitions.Definitions.pathOUT, definitions.Definitions.tempPath, dwiPreprocessed.get_affine(), filename)

warped_atlas = p.register_atlas(definitions.Definitions.atlas, definitions.Definitions.pathOUT, dwiPreprocessed.get_affine(), mapping)

nib.save(nib.Nifti1Image(warped_atlas, dwiPreprocessed.get_affine()), definitions.Definitions.pathOUT + refNameOnly + '_ATLAS' + definitions.Definitions.extension)

import numpy as np
import time

print("Building connectivity matrix " + time.strftime("%H:%M:%S") )

M = p.connectivity_matrix2(streamlines, warped_atlas, affine=streamlines_affine)

print("End Building connectivity matrix " + time.strftime("%H:%M:%S") )


M[:1, :] = 0
M[:, :1] = 0

np.savetxt(definitions.Definitions.pathOUT + 'connectivity.out', M, delimiter=' ', fmt='%s')

print(' ')
print('...................................')
print('  Proceso Finalizado Exitosamente  ')
print('...................................')
