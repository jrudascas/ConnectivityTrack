__author__ = 'Jrudascas'

from dipy.tracking._utils import (_mapping_to_voxel, _to_voxel_coordinates)
from dipy.core.gradients import gradient_table
from dipy.data import get_sphere
from dipy.io import read_bvals_bvecs
from dipy.reconst.dti import color_fa, fractional_anisotropy, quantize_evecs
from dipy.tracking.eudx import EuDX
import dipy.reconst.dti as dti
import sys
import fsl_wrapper as fsl
import utils as utils
import os
import nibabel as nib
from dipy.align.reslice import reslice
import numpy as np
from dipy.denoise.nlmeans import nlmeans
from dipy.segment.mask import median_otsu
import definitions as d
import tools as tools
import scipy.ndimage as ndim
from dipy.io.trackvis import save_trk
from dipy.denoise.noise_estimate import estimate_sigma
from time import time


def eddy_correction(file_in, outPath, ref_bo):
    """
    Prueba documental.

    In:

    file_in: akakakakka
    outPath: kakakasjdjdlllf kklkd
    ref_bo. kskskdejien skkd  dllkd

    Out:

    """
    t = time()
    print('    - running Eddy Correction...')
    refNameOnly = utils.extractFileName(file_in)

    if not (os.path.exists(outPath + utils.extractFileName(file_in) + d.idEddyCorrect + d.extension)):
        refName = utils.extractFileNameExt(file_in)

        os.system('cp ' + file_in + ' ' + d.path_temporal)  # Copiamos archivo de difusion a la carpeta temporal
        fsl.EDDYCORRECT(d.path_temporal + refName, d.path_temporal + refNameOnly + d.idEddyCorrect + '.nii', ref_bo)
        os.system(
            'cp ' + d.path_temporal + refNameOnly + d.idEddyCorrect + d.extension + ' ' + outPath)  # Copiamos archivo de difusion desde carpeta temporal

    print("Total time: ", time() - t)
    print()

    return outPath + refNameOnly + d.idEddyCorrect + d.extension


def reslicing(file_in, outPath, vox_sz):
    t = time()
    print('    - runnning Reslice...')

    finalFileName = outPath + utils.extractFileName(file_in) + d.idReslice + d.extension
    if not (os.path.exists(finalFileName)):
        img = nib.load(file_in)
        data = img.get_data()
        affine = img.affine

        old_vox_sz = img.header.get_zooms()[:3]

        new_vox_sz = (vox_sz, vox_sz, vox_sz)

        # Si el tamano del voxel es isotropico, no es necesario hacer el reslice
        print('Old size:' + str(old_vox_sz))
        print('New size:' + str(new_vox_sz))

        data, affine = reslice(data, affine, old_vox_sz, new_vox_sz)

        nib.save(nib.Nifti1Image(data, affine), finalFileName)
        # nib.save(nib.Nifti1Image(data[:,:,:,0], affine), outPath + refNameOnly + '_only_b0_' + d.idReslice + d.extension)

        # img = nib.load(outPath + refNameOnly + d.idReslice + d.extension)
        # vox_sz = img.get_header().get_zooms()[:3]
        # a = 1
    print("Total time: ", time() - t)
    print()

    return finalFileName


def betDWI(file_in, outPath):
    t = time()
    print('    - running BET with FSL...')

    finalFileName = outPath + utils.extractFileName(file_in) + d.id_bet + '_dwi_masked' + d.extension
    binaryMaskFileName = outPath + utils.extractFileName(file_in) + d.id_bet + '_b0_masked_mask' + d.extension
    b0MaskedFileName = outPath + utils.extractFileName(file_in) + d.id_bet + '_b0_masked' + d.extension

    if not (os.path.exists(b0MaskedFileName)):
        fsl.BET(file_in, b0MaskedFileName, '-m -f .4')

        imgMask = nib.load(binaryMaskFileName)
        dataMask = imgMask.get_data()

        img = nib.load(file_in)
        data = img.get_data()

        data[dataMask == 0] = 0
        nib.save(nib.Nifti1Image(data.astype(np.float32), img.affine), finalFileName)

    print("Total time: ", time() - t)
    print()

    return finalFileName, binaryMaskFileName


def nonLocalMean(file_in, outPath):
    t = time()
    print('    - running NonLocal Mean algoritm...')

    finalFileName = outPath + utils.extractFileName(file_in) + d.idNonLocalMean + d.extension

    if not (os.path.exists(finalFileName)):
        img = nib.load(file_in)
        data = img.get_data()

        newData = np.zeros(data.shape)
        gradientDirections = data.shape[-1]

        for index in range(gradientDirections):
            print(index)
            sigma = estimate_sigma(data[:, :, :, index], N=8)
            newData[:, :, :, index] = nlmeans(data[:, :, :, index], sigma=sigma)

        nib.save(nib.Nifti1Image(newData.astype(np.float32), img.affine), finalFileName)

    print("Total time: ", time() - t)
    print()

    return finalFileName


def medianOtsu(file_in, outPath, median_radius=4, num_pass=4):
    t = time()
    print('    - running Median Otsu algoritm...')

    finalFileName = outPath + utils.extractFileName(file_in) + d.idMedianOtsu + '_maskedVolume' + d.extension
    binaryMaskFileName = outPath + utils.extractFileName(file_in) + d.idMedianOtsu + '_binaryMask' + d.extension
    b0MaskedFileName = outPath + utils.extractFileName(file_in) + d.idMedianOtsu + '_b0Masked' + d.extension

    if not (os.path.exists(finalFileName)):
        img = nib.load(file_in)
        data = img.get_data()
        maskedvolume, mask = median_otsu(data, median_radius, num_pass)

        nib.save(nib.Nifti1Image(maskedvolume.astype(np.float32), img.affine), finalFileName)
        nib.save(nib.Nifti1Image(mask.astype(np.float32), img.affine), binaryMaskFileName)
        nib.save(nib.Nifti1Image(maskedvolume[:, :, :, d.defaultb0Reference].astype(np.float32), img.affine),
                 b0MaskedFileName)

    print("Total time: ", time() - t)
    print()

    return finalFileName, binaryMaskFileName


def to_estimate_dti(file_in, file_inMask, outPath, fbval, fbvec):
    t = time()
    print('building DTI Model...')

    refNameOnly = utils.extractFileName(file_in)

    if (not (os.path.exists(outPath + refNameOnly + d.idEvecs + d.extension))) | (
            not (os.path.exists(outPath + refNameOnly + d.idEvals + d.extension))):
        try:
            os.remove(outPath + refNameOnly + d.idEvecs + d.extension)
            os.remove(outPath + refNameOnly + d.idEvals + d.extension)
        except:
            print("Unexpected error:", sys.exc_info()[0])

        img = nib.load(file_in)
        data = img.get_data()
        mask = nib.load(file_inMask)
        mask = mask.get_data()

        bvals, bvecs = read_bvals_bvecs(fbval, fbvec)
        gtab = gradient_table(bvals, bvecs)

        tenmodel = dti.TensorModel(gtab)
        tenfit = tenmodel.fit(data, mask)

        evecs_img = nib.Nifti1Image(tenfit.evecs.astype(np.float32), img.affine)
        evals_img = nib.Nifti1Image(tenfit.evals.astype(np.float32), img.affine)
        nib.save(evecs_img, outPath + refNameOnly + d.idEvecs + d.extension)
        nib.save(evals_img, outPath + refNameOnly + d.idEvals + d.extension)

    print("Total time: ", time() - t)
    return outPath + refNameOnly + d.idEvecs + d.extension, outPath + refNameOnly + d.idEvals + d.extension


def to_estimate_dti_maps(path_dwi_input, path_output, file_tensor_fitevecs, file_tensor_fitevals):
    import utils as u
    ref_name_only = u.extractFileName(file_tensor_fitevecs)
    ref_name_only = ref_name_only[:-9]

    img_tensorFitevecs = nib.load(file_tensor_fitevecs)
    img_tensorFitevals = nib.load(file_tensor_fitevals)

    evecs = img_tensorFitevecs.get_data()
    evals = img_tensorFitevals.get_data()

    affine = img_tensorFitevecs.affine

    print('--> Calculando el mapa de anisotropia fraccional')
    FA = fractional_anisotropy(evals)
    FA[np.isnan(FA)] = 0

    print('--> Calculando el mapa de anisotropia fraccional RGB')
    FA2 = np.clip(FA, 0, 1)
    RGB = color_fa(FA2, evecs)

    print('--> Calculando el mapa de difusividad media')
    MD = dti.mean_diffusivity(evals)

    print('--> Calculando el mapa de difusividad axial')
    AD = dti.axial_diffusivity(evals)

    print('--> Calculando el mapa de difusividad radial')
    RD = dti.radial_diffusivity(evals)

    print('--> Guardando el mapa de FA')
    nib.save(nib.Nifti1Image(FA.astype(np.float32), affine), path_output + ref_name_only + '_FA' + d.extension)

    print('--> Guardando el mapa de FA a Color')
    nib.save(nib.Nifti1Image(np.array(255 * RGB, 'uint8'), affine),
             path_output + ref_name_only + '_FA_RGB' + d.extension)

    print('--> Guardando el mapa de difusion media')
    nib.save(nib.Nifti1Image(MD.astype(np.float32), affine), path_output + ref_name_only + '_MD' + d.extension)

    print('--> Guardando el mapa de difusividad axial')
    nib.save(nib.Nifti1Image(AD.astype(np.float32), affine), path_output + ref_name_only + '_AD' + d.extension)

    print('--> Guardando el mapa de difusividad radial')
    nib.save(nib.Nifti1Image(RD.astype(np.float32), affine), path_output + ref_name_only + '_RD' + d.extension)

    print('--> Guardando la Tractografia')

    sphere = get_sphere('symmetric724')
    peak_indices = quantize_evecs(evecs, sphere.vertices)

    eu = EuDX(FA.astype('f8'), peak_indices, seeds=300000, odf_vertices=sphere.vertices, a_low=0.15)
    tensor_streamlines = [streamline for streamline in eu]

    hdr = nib.trackvis.empty_header()
    hdr['voxel_size'] = nib.load(path_dwi_input).get_header().get_zooms()[:3]
    hdr['voxel_order'] = 'LAS'
    hdr['dim'] = FA.shape

    tensor_streamlines_trk = ((sl, None, None) for sl in tensor_streamlines)

    nib.trackvis.write(path_output + ref_name_only + '_tractography_EuDx.trk', tensor_streamlines_trk, hdr,
                       points_space='voxel')

    return tensor_streamlines, eu.affine


def to_generate_tractography(path_dwi_input, path_binary_mask, path_out, path_bval, path_bvec):
    from dipy.reconst.shm import CsaOdfModel
    from dipy.data import default_sphere
    from dipy.direction import peaks_from_model
    from dipy.tracking.local import LocalTracking
    from dipy.tracking import utils
    from dipy.tracking.local import ThresholdTissueClassifier

    print('    - Starting reconstruction of Tractography...')

    if not os.path.exists(path_out + '_tractography_CsaOdf' + '.trk'):
        dwi_img = nib.load(path_dwi_input)
        dwi_data = dwi_img.get_data()
        dwi_affine = dwi_img.affine

        dwi_mask_data = nib.load(path_binary_mask).get_data()

        g_tab = gradient_table(path_bval, path_bvec)

        csa_model = CsaOdfModel(g_tab, sh_order=6)

        csa_peaks = peaks_from_model(csa_model, dwi_data, default_sphere, sh_order=6,
                                     relative_peak_threshold=.8,
                                     min_separation_angle=25, mask=dwi_mask_data.astype(bool))

        classifier = ThresholdTissueClassifier(csa_peaks.gfa, .2)

        seeds = utils.seeds_from_mask(dwi_mask_data.astype(bool), density=[3, 3, 3], affine=dwi_affine)

        streamlines = LocalTracking(csa_peaks, classifier, seeds, dwi_affine, step_size=2)

        streamlines = [s for s in streamlines if s.shape[0] > 10]

        streamlines = list(streamlines)

        save_trk(path_out + '_tractography_CsaOdf' + '.trk', streamlines, dwi_affine, dwi_data.shape)

    print('    - Ending reconstruction of Tractography...')


def to_register_dwi_to_mni(path_in, path_out, path_bvec, path_bval):
    img_DWI = nib.load(path_in)
    data_DWI = img_DWI.get_data()
    affine_DWI = img_DWI.affine

    bvals, bvecs = read_bvals_bvecs(path_bval, path_bvec)
    gtab = gradient_table(bvals, bvecs)

    b0 = data_DWI[..., gtab.b0s_mask]

    mean_b0 = np.mean(b0, -1)

    mni_t2 = nib.load(d.standardT2)
    mni_t2_data = mni_t2.get_data()
    MNI_T2_affine = mni_t2.affine

    directionWarped = np.zeros((mni_t2_data.shape[0], mni_t2_data.shape[1], mni_t2_data.shape[2], data_DWI.shape[-1]))
    rangos = range(data_DWI.shape[-1])

    affine, starting_affine = tools.affine_registration(mean_b0, mni_t2_data, moving_grid2world=affine_DWI,
                                                        static_grid2world=MNI_T2_affine)

    warped_moving, mapping = tools.syn_registration(mean_b0, mni_t2_data,
                                                    moving_grid2world=affine_DWI,
                                                    static_grid2world=MNI_T2_affine,
                                                    # step_length=0.1,
                                                    # sigma_diff=2.0,
                                                    metric='CC',
                                                    dim=3, level_iters=[10, 10, 5],
                                                    # prealign=affine.affine)
                                                    prealign=starting_affine)

    for gradientDirection in rangos:
        # print(gradientDirection)
        directionWarped[:, :, :, gradientDirection] = mapping.transform(
            data_DWI[:, :, :, gradientDirection].astype(int), interpolation='nearest')

    nib.save(nib.Nifti1Image(directionWarped, MNI_T2_affine), path_out + 'dwiNormalized' + d.extension)

    return directionWarped, MNI_T2_affine, mapping, path_out + 'dwiNormalized' + d.extension


def to_register_t1_to_nmi(path_in, path_out):
    img = nib.load(path_in)
    data = img.get_data()
    affineStructural = img.affine

    MNI_T1 = nib.load(d.standardT1)
    MNI_T1_data = MNI_T1.get_data()
    MNI_T1_affine = MNI_T1.affine

    affine, starting_affine = tools.affine_registration(data, MNI_T1_data, moving_grid2world=affineStructural,
                                                        static_grid2world=MNI_T1_affine)

    warped_moving, mapping_t1 = tools.syn_registration(data, MNI_T1_data,
                                                       moving_grid2world=affineStructural,
                                                       static_grid2world=MNI_T1_affine,
                                                       # step_length=0.1,
                                                       # sigma_diff=2.0,
                                                       metric='CC',
                                                       dim=3, level_iters=[5, 5, 3],
                                                       # dim=3, level_iters = [10, 10, 5],
                                                       # prealign=affine.affine)
                                                       prealign=starting_affine)

    nib.save(nib.Nifti1Image(warped_moving, MNI_T1_affine), path_out + 'dwiNormalized' + d.extension)

    return warped_moving, MNI_T1_affine, mapping_t1


def registrationtoNMI(file_in, outPath):
    img = nib.load(file_in)
    data = img.get_data()
    affineStructural = img.affine

    MNI_T2 = nib.load(d.standardT2)
    MNI_T2_data = MNI_T2.get_data()
    MNI_T2_affine = MNI_T2.affine

    affine, starting_affine = tools.affine_registration(data, MNI_T2_data, moving_grid2world=affineStructural,
                                                        static_grid2world=MNI_T2_affine)

    warped_moving, mapping = tools.syn_registration(data, MNI_T2_data,
                                                    moving_grid2world=affineStructural,
                                                    static_grid2world=MNI_T2_affine,
                                                    # step_length=0.1,
                                                    # sigma_diff=2.0,
                                                    metric='CC',
                                                    dim=3, level_iters=[5, 5, 3],
                                                    # dim=3, level_iters = [10, 10, 5],
                                                    # prealign=affine.affine)
                                                    prealign=starting_affine)

    return warped_moving, MNI_T2_affine, mapping


def registrationto(file_in, file_reg):
    img = nib.load(file_in)
    data = img.get_data()
    affineStructural = img.affine

    MNI_T2 = nib.load(file_reg)
    MNI_T2_data = MNI_T2.get_data()
    MNI_T2_affine = MNI_T2.affine

    affine, starting_affine = tools.affine_registration(data, MNI_T2_data, moving_grid2world=affineStructural,
                                                        static_grid2world=MNI_T2_affine)

    warped_moving, mapping = tools.syn_registration(data, MNI_T2_data,
                                                    moving_grid2world=affineStructural,
                                                    static_grid2world=MNI_T2_affine,
                                                    # step_length=0.1,
                                                    # sigma_diff=2.0,
                                                    metric='CC',
                                                    dim=3, level_iters=[5, 5, 3],
                                                    # dim=3, level_iters = [10, 10, 5],
                                                    # prealign=affine.affine)
                                                    prealign=starting_affine)

    return warped_moving, MNI_T2_affine, mapping


def registerAffine_atlas(pathAtlas, pathStandard, outPath, tempPath, affineSubject, Subject):
    atlas = nib.load(pathAtlas)
    atlas_data = atlas.get_data()

    indexs = np.unique(atlas_data)

    refNameOnly = utils.extractFileName(pathAtlas)

    file_outSubject, omatSubject = fsl.FLIRT(pathStandard, tempPath + 'Aux_FLIRT' + d.extension, Subject,
                                             tempPath + 'Aux_FLIRT_omat.mat')
    fsl.hexTodec(omatSubject, omatSubject + '.mat2')
    omatSubject = omatSubject + '.mat2'

    for index in indexs:
        roi = (atlas_data == index)
        nib.save(nib.Nifti1Image(roi.astype(np.float32), affineSubject),
                 tempPath + refNameOnly + '_ROI_' + str(index) + d.extension)

        fsl.FLIRT_xfm(tempPath + refNameOnly + '_ROI_' + str(index) + d.extension,
                      outPath + refNameOnly + '_ROI_' + str(index) + '_FLIRT' + d.extension, Subject, omatSubject)


def register_atlas(pathAtlas, outPath, affineSubject, mapping):
    atlas = nib.load(pathAtlas)
    atlas_data = atlas.get_data()

    indexs = np.unique(atlas_data)

    refNameOnly = utils.extractFileName(pathAtlas)

    for index in indexs:
        roi = (atlas_data == index)
        # warped_roi = mapping.transform_inverse(roi.astype(int)*255, interpolation='nearest')
        warped_roi = mapping.transform_inverse(ndim.binary_dilation(roi).astype(int), interpolation='nearest')

        warped_roi = ndim.binary_dilation(warped_roi)
        warped_roi = ndim.binary_erosion(warped_roi)

        bin_warped_roi = np.ceil(warped_roi)

        filled_warped_roi = ndim.binary_fill_holes(bin_warped_roi.astype(int)).astype(int)

        nib.save(nib.Nifti1Image(filled_warped_roi.astype(np.float32), affineSubject),
                 outPath + refNameOnly + '_ROI_' + str(index) + d.extension)

        print("ROI # " + str(index) + " for " + refNameOnly + " Atlas, has been saved")

        if not ('registeredAtlas' in locals()):
            registeredAtlas = np.zeros(filled_warped_roi.shape)

        registeredAtlas[filled_warped_roi != 0] = index

    nib.save(nib.Nifti1Image(registeredAtlas, affineSubject), outPath + refNameOnly + '_ATLAS2' + d.extension)
    return registeredAtlas.astype(np.int32)


def connectivity_matrix2(streamlines, label_volume, affine, shape, voxel_size=None):
    endpoints = [sl for sl in streamlines]
    lin_T, offset = _mapping_to_voxel(affine, voxel_size)
    # endpoints = _to_voxel_coordinates(streamlines, lin_T, offset)
    # endpoints = endpoints.astype(int)
    # streamlines = list(endpoints)
    # endlabels2 = label_volume[i2, j2, k2]
    myList = []
    indexROI = np.unique(label_volume)
    indexROI.sort(0)
    matriz = np.zeros(shape=(len(indexROI), len(indexROI)))
    from decimal import Decimal

    print("ROI Number = " + str(len(indexROI)))

    for ROI in indexROI:
        ROIimg = (label_volume == ROI)
        ROIimg = ROIimg.astype(int)

        for ROI2 in indexROI:
            # if ((ROI == 1) & (ROI2 == 2)):
            if (1):
                if (ROI2 > ROI):
                    ROI2img = (label_volume == ROI2)
                    ROI2img = ROI2img.astype(int)

                    for sl in streamlines:
                        # sl += offset
                        sl_Aux = sl
                        sl = _to_voxel_coordinates(sl, lin_T, offset)
                        i, j, k = sl.T
                        # i2, j2, k2 = endpoints.T

                        labelsROI = ROIimg[i, j, k]
                        labelsROI2 = ROI2img[i, j, k]

                        if ((sum(labelsROI) > 0) & (sum(labelsROI2) > 0)):
                            matriz[ROI, ROI2] = matriz[ROI, ROI2] + 1
                            # myList.append(sl_Aux)
        print(ROI)

    return matriz.astype(int)


def to_generate_bunddle(path_dwi_input, path_binary_mask, path_bval, path_bvec, rules, atlas_dict):
    from dipy.reconst.shm import CsaOdfModel
    from dipy.data import default_sphere
    from dipy.direction import peaks_from_model
    from dipy.tracking.local import LocalTracking
    from dipy.tracking import utils
    from dipy.tracking.local import ThresholdTissueClassifier

    print('Starting the model')

    t = time()

    dwi_img = nib.load(path_dwi_input)
    dwi_data = dwi_img.get_data()
    dwi_affine = dwi_img.affine

    dwi_mask_data = nib.load(path_binary_mask).get_data()

    g_tab = gradient_table(path_bval, path_bvec)

    lin_T, offset = _mapping_to_voxel(dwi_affine, None)

    csa_model = CsaOdfModel(g_tab, sh_order=6)

    csa_peaks = peaks_from_model(csa_model, dwi_data, default_sphere, sh_order=6,
                                 relative_peak_threshold=.8,
                                 min_separation_angle=25, mask=dwi_mask_data.astype(bool))

    classifier = ThresholdTissueClassifier(csa_peaks.gfa, .2)

    print('Finished the model: ' + str(time() - t))

    ruleNumber = 1
    for rule in rules:

        t = time()

        print('Starting ROI reconstruction')

        for elementROI in rule[0][1]:
            if not ('roi' in locals()):
                roi = nib.load(atlas_dict[rule[0][0]][elementROI - 1]).get_data().astype(bool)
            else:
                roi = roi | nib.load(atlas_dict[rule[0][0]][elementROI - 1]).get_data().astype(bool)

        nib.save(nib.Nifti1Image(roi.astype(np.float32), dwi_affine),
                 '/home/jrudascas/Desktop/DWITest/Datos_Salida/roi_rule_' + str(ruleNumber) + '.nii')

        for elementROI in rule[1][1]:
            if not ('target' in locals()):
                target = nib.load(atlas_dict[rule[1][0]][elementROI - 1]).get_data().astype(bool)
            else:
                target = target | nib.load(atlas_dict[rule[1][0]][elementROI - 1]).get_data().astype(bool)

        nib.save(nib.Nifti1Image(target.astype(np.float32), dwi_affine),
                 '/home/jrudascas/Desktop/DWITest/Datos_Salida/target_rule_' + str(ruleNumber) + '.nii')

        seeds = utils.seeds_from_mask(roi.astype(bool), density=[3, 3, 3], affine=dwi_affine)

        streamlines = LocalTracking(csa_peaks, classifier, seeds, dwi_affine, step_size=.5)

        streamlines = [s for s in streamlines if s.shape[0] > 5]

        streamlines = list(streamlines)

        save_trk('/home/jrudascas/Desktop/DWITest/Datos_Salida/BundleROI_rule_' + str(ruleNumber) + '.trk', streamlines,
                 dwi_affine, roi.shape)

        print('Finished ROI reconstruction: ' + str(time() - t))

        t = time()

        print('Starting TARGET filtering')

        bunddle = []

        t = time()
        for sl in streamlines:
            # sl += offset
            # sl_Aux = np.copy(sl)
            sl_Aux = sl
            sl = _to_voxel_coordinates(sl, lin_T, offset)
            i, j, k = sl.T
            labelsROI = target[i, j, k]

            if sum(labelsROI) > 0:
                bunddle.append(sl_Aux)

        save_trk('/home/jrudascas/Desktop/DWITest/Datos_Salida/BundleROI_to_TARGET_rule_' + str(ruleNumber) + '.trk',
                 bunddle, dwi_affine, roi.shape)

        print('Finished TARGET filtering: ' + str(time() - t))

        if len(rule) == 3:  # If necessary other filtering (exclusition)
            print('Starting exclusive filtering')
            t = time()

            for elementROI in rule[2][1]:
                if not ('roiFiltered' in locals()):
                    roiFiltered = nib.load(atlas_dict[rule[2][0]][elementROI - 1]).get_data().astype(bool)
                else:
                    roiFiltered = roiFiltered | nib.load(atlas_dict[rule[2][0]][elementROI - 1]).get_data().astype(bool)

            bunddleFiltered = []

            for b in bunddle:
                b_Aux = b
                b = _to_voxel_coordinates(b, lin_T, offset)
                i, j, k = b.T
                labelsROI = roiFiltered[i, j, k]

                if sum(labelsROI) == 0:
                    bunddleFiltered.append(b_Aux)

            save_trk('/home/jrudascas/Desktop/DWITest/Datos_Salida/BundleROI_to_TARGET_Filtered_rule_' + str(
                ruleNumber) + '.trk',
                     bunddleFiltered, dwi_affine, roi.shape)

            print('Finished exclusive filtering: ' + str(time() - t))
            del roiFiltered

        del roi
        del target

        ruleNumber = ruleNumber + 1


def toGenerateBunddle(roi1, roi2, data, gtab, affine):
    print('Starting Bundles generator')

    from dipy.reconst.shm import CsaOdfModel
    from dipy.data import default_sphere
    from dipy.direction import peaks_from_model
    from dipy.tracking.local import LocalTracking
    from dipy.tracking import utils
    from dipy.tracking.local import ThresholdTissueClassifier

    lin_T, offset = _mapping_to_voxel(affine, None)

    volueMaskPath = '/home/jrudascas/Desktop/DWITest/Datos_Salida/Subject_Reslice_MedianOtsu_b0Masked_mask.nii.gz'
    masked = nib.load(volueMaskPath)
    dataMasked = masked.get_data()

    maskI = dataMasked

    # tensor_model = dti.TensorModel(gtab)
    # tenfit = tensor_model.fit(data, mask=maskI)

    # FA = fractional_anisotropy(tenfit.evals)
    # classifier = ThresholdTissueClassifier(FA, .2)

    '''
    print('Starting the ConstrainedSphericalDeconvModel')
    t = time()
    csd_model = ConstrainedSphericalDeconvModel(gtab, None, sh_order=6)
    csd_fit = csd_model.fit(data, mask=maskI)

    detmax_dg = DeterministicMaximumDirectionGetter.from_shcoeff(csd_fit.shm_coeff,
                                                                 max_angle=30.,
                                                                 sphere=default_sphere)

    streamlines1 = LocalTracking(detmax_dg, classifier, seeds, affine, step_size=3)

    save_trk("/home/jrudascas/Desktop/DWITest/Datos_Salida/ConstrainedSphericalDeconvModel.trk", streamlines1, affine, roi1.shape)

    print(time() - t)

    
    print('Starting the CsaOdfModel')
    t = time()

    print('Initialization the seeds')
    seeds = utils.seeds_from_mask(maskI, density=[2, 2, 2], affine=affine)

    csa_model = CsaOdfModel(gtab, sh_order=6)
    csa_peaks = peaks_from_model(csa_model, data, default_sphere, sh_order=6,
                                 relative_peak_threshold=.8,
                                 min_separation_angle=45, mask=maskI)

    classifier = ThresholdTissueClassifier(csa_peaks.gfa, .25)

    streamlines = LocalTracking(csa_peaks, classifier, seeds, affine, step_size=2)

    streamlines = [s for s in streamlines if s.shape[0]>80]

    streamlines = list(streamlines)

    save_trk("/home/jrudascas/Desktop/DWITest/Datos_Salida/CsaOdfModel.trk", streamlines, affine, roi1.shape)

    print(time() - t)
'''
    print('Starting the CsaOdfModel ROI')

    t = time()

    ROIPath = '/home/jrudascas/Desktop/DWITest/Datos_Salida/AAN_1mm_ROI_1.0.nii.gz'
    ROImasked = nib.load(ROIPath)
    maskROI = ROImasked.get_data()

    seeds = utils.seeds_from_mask(maskROI.astype(bool), density=[3, 3, 3], affine=affine)

    csa_model = CsaOdfModel(gtab, sh_order=6)

    csa_peaks = peaks_from_model(csa_model, data, default_sphere, sh_order=6,
                                 relative_peak_threshold=.8,
                                 min_separation_angle=45, mask=dataMasked.astype(bool))

    classifier = ThresholdTissueClassifier(csa_peaks.gfa, .25)

    streamlines = LocalTracking(csa_peaks, classifier, seeds, affine, step_size=.5)

    streamlines = [s for s in streamlines if s.shape[0] > 5]

    streamlines = list(streamlines)

    save_trk("/home/jrudascas/Desktop/DWITest/Datos_Salida/CsaOdfModelROI.trk", streamlines, affine, roi1.shape)

    print('Reconstruction: ' + time() - t)

    bunddle = []

    t = time()
    for sl in streamlines:
        # sl += offset
        # sl_Aux = np.copy(sl)
        sl_Aux = sl
        sl = _to_voxel_coordinates(sl, lin_T, offset)
        i, j, k = sl.T
        labelsROI = roi2[i, j, k]

        if sum(labelsROI) > 0:
            bunddle.append(sl_Aux)

        save_trk('/home/jrudascas/Desktop/DWITest/Datos_Salida/BundleROI_to_ROI.trk', bunddle, affine, roi2.shape)

    print('ROI to ROI: ' + time() - t)


# def toEstimateBunddle(streamlines, atlas, dict):
#    for bunddle in dict:

def connectivity_matrix(streamlines, label_volume, voxel_size=None,
                        affine=None, symmetric=True, return_mapping=False,
                        mapping_as_streamlines=False):
    """Counts the streamlines that start and end at each label pair.

    Parameters
    ----------
    streamlines : sequence
        A sequence of streamlines.
    label_volume : ndarray
        An image volume with an integer data type, where the intensities in the
        volume map to anatomical structures.
    voxel_size :
        This argument is deprecated.
    affine : array_like (4, 4)
        The mapping from voxel coordinates to streamline coordinates.
    symmetric : bool, False by default
        Symmetric means we don't distinguish between start and end points. If
        symmetric is True, ``matrix[i, j] == matrix[j, i]``.
    return_mapping : bool, False by default
        If True, a mapping is returned which maps matrix indices to
        streamlines.
    mapping_as_streamlines : bool, False by default
        If True voxel indices map to lists of streamline objects. Otherwise
        voxel indices map to lists of integers.

    Returns
    -------
    matrix : ndarray
        The number of connection between each pair of regions in
        `label_volume`.
    mapping : defaultdict(list)
        ``mapping[i, j]`` returns all the streamlines that connect region `i`
        to region `j`. If `symmetric` is True mapping will only have one key
        for each start end pair such that if ``i < j`` mapping will have key
        ``(i, j)`` but not key ``(j, i)``.

    """
    # Error checking on label_volume
    kind = label_volume.dtype.kind
    labels_positive = ((kind == 'u') or
                       ((kind == 'i') and (label_volume.min() >= 0)))
    valid_label_volume = (labels_positive and label_volume.ndim == 3)
    if not valid_label_volume:
        raise ValueError("label_volume must be a 3d integer array with"
                         "non-negative label values")

    print(streamlines.__len__())
    # If streamlines is an iterators
    if return_mapping and mapping_as_streamlines:
        streamlines = list(streamlines)
    # take the first and last point of each streamline
    endpoints = [sl[0::len(sl) - 1] for sl in streamlines]

    print(streamlines.__len__())
    print(endpoints.__len__())

    from dipy.tracking._utils import (_mapping_to_voxel, _to_voxel_coordinates)
    from collections import defaultdict

    # Map the streamlines coordinates to voxel coordinates
    lin_T, offset = _mapping_to_voxel(affine, voxel_size)
    endpoints = _to_voxel_coordinates(endpoints, lin_T, offset)

    # get labels for label_volume
    i, j, k = endpoints.T
    endlabels = label_volume[i, j, k]
    if symmetric:
        endlabels.sort(0)
    mx = label_volume.max() + 1
    matrix = ndbincount(endlabels, shape=(mx, mx))
    if symmetric:
        matrix = np.maximum(matrix, matrix.T)

    if return_mapping:
        mapping = defaultdict(list)
        for i, (a, b) in enumerate(endlabels.T):
            mapping[a, b].append(i)

        # Replace each list of indices with the streamlines they index
        if mapping_as_streamlines:
            for key in mapping:
                mapping[key] = [streamlines[i] for i in mapping[key]]

        # Return the mapping matrix and the mapping
        return matrix, mapping
    else:
        return matrix


def ndbincount(x, weights=None, shape=None):
    """Like bincount, but for nd-indicies.

    Parameters
    ----------
    x : array_like (N, M)
        M indices to a an Nd-array
    weights : array_like (M,), optional
        Weights associated with indices
    shape : optional
        the shape of the output
    """
    x = np.asarray(x)
    if shape is None:
        shape = x.max(1) + 1

    x = ravel_multi_index(x, shape)
    # out = np.bincount(x, weights, minlength=np.prod(shape))
    # out.shape = shape
    # Use resize to be compatible with numpy < 1.6, minlength new in 1.6
    out = np.bincount(x, weights)
    out.resize(shape)

    return out


def ravel_multi_index(multi_index, dims, mode='raise', order='C'):  # real signature unknown; restored from __doc__
    """
    ravel_multi_index(multi_index, dims, mode='raise', order='C')

        Converts a tuple of index arrays into an array of flat
        indices, applying boundary modes to the multi-index.

        Parameters
        ----------
        multi_index : tuple of array_like
            A tuple of integer arrays, one array for each dimension.
        dims : tuple of ints
            The shape of array into which the indices from ``multi_index`` apply.
        mode : {'raise', 'wrap', 'clip'}, optional
            Specifies how out-of-bounds indices are handled.  Can specify
            either one mode or a tuple of modes, one mode per index.

            * 'raise' -- raise an error (default)
            * 'wrap' -- wrap around
            * 'clip' -- clip to the range

            In 'clip' mode, a negative index which would normally
            wrap will clip to 0 instead.
        order : {'C', 'F'}, optional
            Determines whether the multi-index should be viewed as indexing in
            C (row-major) order or FORTRAN (column-major) order.

        Returns
        -------
        raveled_indices : ndarray
            An array of indices into the flattened version of an array
            of dimensions ``dims``.

        See Also
        --------
        unravel_index

        Notes
        -----
        .. versionadded:: 1.6.0
    """
    pass
