from docutils.nodes import definition

__author__ = 'Jrudascas'

from dipy.core.gradients import gradient_table
from dipy.data import get_sphere
from dipy.io import read_bvals_bvecs
from dipy.reconst.dti import color_fa, fractional_anisotropy, quantize_evecs
from dipy.tracking.eudx import EuDX
import dipy.reconst.dti as dti
import sys
import FSLWrapper as fsl
import Utils as utils
import os
import nibabel as nib
from dipy.align.reslice import reslice
import numpy as np
from dipy.denoise.nlmeans import nlmeans
from dipy.segment.mask import median_otsu
import Definitions as definitions
import Tools as tools
import scipy.ndimage as ndim

def EddyCorrect_DWI (file_in, outPath, ref_bo):

    print('Eddy Correction...')
    refNameOnly = utils.extractFileName(file_in)

    if not(os.path.exists(outPath + utils.extractFileName(file_in) + definitions.Definitions.idEddyCorrect + definitions.Definitions.extension)):
       refName = utils.extractFileNameExt(file_in)

       os.system('cp ' + file_in + ' ' + definitions.Definitions.tempPath) #Copiamos archivo de difusion a la carpeta temporal
       fsl.EDDYCORRECT(definitions.Definitions.tempPath + refName, definitions.Definitions.tempPath + refNameOnly + definitions.Definitions.idEddyCorrect + '.nii', ref_bo)
       os.system('cp ' + definitions.Definitions.tempPath + refNameOnly + definitions.Definitions.idEddyCorrect + definitions.Definitions.extension + ' ' + outPath) #Copiamos archivo de difusion desde carpeta temporal

    return outPath + refNameOnly + definitions.Definitions.idEddyCorrect + definitions.Definitions.extension

def Reslice_DWI (file_in, outPath, vox_sz):

    print('Reslice...')

    refNameOnly = utils.extractFileName(file_in)
    if not(os.path.exists(outPath + utils.extractFileName(file_in) + definitions.Definitions.idReslice + definitions.Definitions.extension)):

       img = nib.load(file_in)
       data = img.get_data()
       affine = img.get_affine()

       old_vox_sz = img.get_header().get_zooms()[:3]

       new_vox_sz = (vox_sz, vox_sz, vox_sz)

       #Si el tamano del voxel es isotropico, no es necesario hacer el reslice
       data, affine = reslice(data, affine, old_vox_sz, new_vox_sz)
       nib.save(nib.Nifti1Image(data, affine), outPath + refNameOnly + definitions.Definitions.idReslice + definitions.Definitions.extension)

       #img = nib.load(outPath + refNameOnly + definitions.Definitions.idReslice + definitions.Definitions.extension)
       #vox_sz = img.get_header().get_zooms()[:3]
       #a = 1
    return outPath + refNameOnly + definitions.Definitions.idReslice + definitions.Definitions.extension

def NonLocalMean_DWI (file_in, outPath, threshold, ref_b0):

    print('NonLocal Mean...')

    refNameOnly = utils.extractFileName(file_in)
    if not(os.path.exists(outPath + utils.extractFileName(file_in) + definitions.Definitions.idNonLocalMean + definitions.Definitions.extension)):

       img = nib.load(file_in)
       data = img.get_data()
       mask = data[..., ref_b0] > threshold
       data2 = data[..., ref_b0]
       sigma = np.std(data2[~mask])
       den = nlmeans(data2, sigma=sigma, mask=mask)

       nib.save(nib.Nifti1Image(den.astype(np.float32), img.get_affine()), outPath + refNameOnly + definitions.Definitions.idNonLocalMean + definitions.Definitions.extension)

    return outPath + refNameOnly + definitions.Definitions.idNonLocalMean + definitions.Definitions.extension

def Median_Otsu_DWI (file_in, outPath, median_radius, num_pass):

    print('Median Otsu...')

    refNameOnly = utils.extractFileName(file_in)

    if not(os.path.exists(outPath + utils.extractFileName(file_in) + definitions.Definitions.idMedianOtsu + '_mask' + definitions.Definitions.extension)):

       img = nib.load(file_in)
       data = img.get_data()
       b0_mask, mask = median_otsu(data, median_radius, num_pass)

       nib.save(nib.Nifti1Image(b0_mask.astype(np.float32), img.get_affine()), outPath + refNameOnly + definitions.Definitions.idMedianOtsu + '_b0' + definitions.Definitions.extension)
       nib.save(nib.Nifti1Image(mask.astype(np.float32), img.get_affine()), outPath + refNameOnly + definitions.Definitions.idMedianOtsu + '_mask' + definitions.Definitions.extension)

    return outPath + refNameOnly + definitions.Definitions.idMedianOtsu + '_mask' + definitions.Definitions.extension

def DTIModel(file_in, file_inMask, outPath, fbval, fbvec):

    refNameOnly = utils.extractFileName(file_in)

    if (not(os.path.exists(outPath + refNameOnly + definitions.Definitions.idEvecs + definitions.Definitions.extension))) | (not(os.path.exists(outPath + refNameOnly + definitions.Definitions.idEvals + definitions.Definitions.extension))):
        try:
            os.remove(outPath + refNameOnly + definitions.Definitions.idEvecs + definitions.Definitions.extension)
            os.remove(outPath + refNameOnly + definitions.Definitions.idEvals + definitions.Definitions.extension)
        except:
            print "Unexpected error:", sys.exc_info()[0]

        img = nib.load(file_in)
        data = img.get_data()
        vox_sz = img.get_header().get_zooms()[:3]
        mask = nib.load(file_inMask)
        mask = mask.get_data()

        bvals, bvecs = read_bvals_bvecs(fbval, fbvec)
        gtab = gradient_table(bvals, bvecs)

        tenmodel = dti.TensorModel(gtab)
        tenfit = tenmodel.fit(data, mask)

        evecs_img = nib.Nifti1Image(tenfit.evecs.astype(np.float32), img.get_affine())
        evals_img = nib.Nifti1Image(tenfit.evals.astype(np.float32), img.get_affine())
        nib.save(evecs_img, outPath + refNameOnly + definitions.Definitions.idEvecs + definitions.Definitions.extension)
        nib.save(evals_img, outPath + refNameOnly + definitions.Definitions.idEvals + definitions.Definitions.extension)

    list = [outPath + refNameOnly + definitions.Definitions.idEvecs + definitions.Definitions.extension, outPath + refNameOnly + definitions.Definitions.idEvals + definitions.Definitions.extension]
    return list

def EstimateMapsDTI(file_in, file_inMask, file_tensorFitevecs, file_tensorFitevals, outPath, fbvec, fbval, mapping = None, affine_mapping = None):

    import Utils as u
    refNameOnly = u.extractFileName(file_tensorFitevecs)
    refNameOnly = refNameOnly[:-9]

    img_tensorFitevecs = nib.load(file_tensorFitevecs)
    img_tensorFitevals = nib.load(file_tensorFitevals)

    evecs = img_tensorFitevecs.get_data()
    evals = img_tensorFitevals.get_data()

    affine = img_tensorFitevecs.get_affine()

    print('--> Calculando el mapa de anisotropia fraccional')
    FA = fractional_anisotropy(evals)
    FA[np.isnan(FA)] = 0

    print('--> Calculando el mapa de anisotropia fraccional RGB')
    FA2 = np.clip(FA, 0, 1)
    RGB = color_fa(FA2, evecs)

    print('--> Calculando el mapa de difusividad media')
    MD1 = dti.mean_diffusivity(evals)

    print('--> Calculando el mapa de difusividad axial')
    AD = dti.axial_diffusivity(evals)

    print('--> Calculando el mapa de difusividad radial')
    RD = dti.radial_diffusivity(evals)

    print('--> Guardando el mapa de FA')
    nib.save(nib.Nifti1Image(FA.astype(np.float32), affine), outPath + refNameOnly + '_FA' + definitions.Definitions.extension)

    #fsl.FLIRT(pathOUT + refName + '_FA.nii.gz', pathOUT + refName + '_FA_Normalized.nii', template, 'transformation_FA.mat')
    if (not(mapping is None)) & (not(affine_mapping is None)):
       warped = mapping.transform(FA)
       nib.save(nib.Nifti1Image(warped.astype(np.float32), affine_mapping),  outPath + refNameOnly + '_FA_Normalized.nii.gz')

    print('--> Guardando el mapa de FA a Color')
    nib.save(nib.Nifti1Image(np.array(255 * RGB, 'uint8'), affine), outPath + refNameOnly + '_FA_RGB' + definitions.Definitions.extension)

    #if (not(mapping is None)) & (not(affine_mapping is None)):
    #   warped = mapping.transform(RGB)
    #   nib.save(nib.Nifti1Image(warped.astype(np.float32), affine_mapping),  outPath + refNameOnly + '_FA_RGB_Normalized.nii.gz')

    #fsl.FLIRT(pathOUT + refName + '_FA_RGB.nii.gz', pathOUT + refName + '_FA_RGB_Normalized.nii', template, 'transformation_FA_RGB.mat')

    print('--> Guardando el mapa de difusion media')
    nib.save(nib.Nifti1Image(MD1.astype(np.float32), affine), outPath + refNameOnly + '_MD' + definitions.Definitions.extension)

    if (not(mapping is None)) & (not(affine_mapping is None)):
       warped = mapping.transform(MD1)
       nib.save(nib.Nifti1Image(warped.astype(np.float32), affine_mapping),  outPath + refNameOnly + '_MD_Normalized.nii.gz')

    #fsl.FLIRT(pathOUT + refName + '_MD.nii.gz', pathOUT + refName + '_MD_Normalized.nii', template, 'transformation_MD.mat')

    print('--> Guardando el mapa de difusividad axial')
    nib.save(nib.Nifti1Image(AD.astype(np.float32), affine), outPath + refNameOnly + '_AD' + definitions.Definitions.extension)

    if (not(mapping is None)) & (not(affine_mapping is None)):
       warped = mapping.transform(AD)
       nib.save(nib.Nifti1Image(warped.astype(np.float32), affine_mapping),  outPath + refNameOnly + '_AD_Normalized.nii.gz')

    #fsl.FLIRT(pathOUT + refName + '_AD.nii.gz', pathOUT + refName + '_AD_Normalized.nii', template, 'transformation_AD.mat')

    print('--> Guardando el mapa de difusividad radial')
    nib.save(nib.Nifti1Image(RD.astype(np.float32), affine), outPath + refNameOnly + '_RD' + definitions.Definitions.extension)

    if (not(mapping is None)) & (not(affine_mapping is None)):
       warped = mapping.transform(RD)
       nib.save(nib.Nifti1Image(warped.astype(np.float32), affine_mapping),  outPath + refNameOnly + '_RD_Normalized.nii.gz')

    #fsl.FLIRT(pathOUT + refName + '_RD.nii.gz', pathOUT + refName + '_RD_Normalized.nii', template, 'transformation_RD.mat')

    #print('--> Guardando imagen estructural')
    #nib.save(nib.Nifti1Image(imgT1.astype(np.float32), imgT1.get_affine()), pathOUT + refName + '_Structure.nii.gz')

    print('--> Guardando la Tractografia')

    sphere = get_sphere('symmetric724')
    peak_indices = quantize_evecs(evecs, sphere.vertices)

    eu = EuDX(FA.astype('f8'), peak_indices, seeds=100000, odf_vertices = sphere.vertices, a_low=0.15)
    tensor_streamlines = [streamline for streamline in eu]
    new_vox_sz = (definitions.Definitions.vox_sz, definitions.Definitions.vox_sz, definitions.Definitions.vox_sz)
    hdr = nib.trackvis.empty_header()
    hdr['voxel_size'] = new_vox_sz
    hdr['voxel_order'] = 'LAS'
    hdr['dim'] = FA.shape

    tensor_streamlines_trk = ((sl, None, None) for sl in tensor_streamlines)
    streamlines = list(eu)
    ten_sl_fname = outPath + refNameOnly + '_Tractografia.trk'
    nib.trackvis.write(ten_sl_fname, tensor_streamlines_trk, hdr, points_space='voxel')

    """
    from dipy.reconst.shm import CsaOdfModel
    from dipy.data import default_sphere
    from dipy.direction import peaks_from_model

    bvals, bvecs = read_bvals_bvecs(fbval, fbvec)
    gtab = gradient_table(bvals, bvecs)

    dataDWI = nib.load(file_in)
    data = dataDWI.get_data()
    affine = dataDWI.get_affine()

    old_vox_sz = dataDWI.get_header().get_zooms()[:3]
    if np.unique(old_vox_sz).size != 1:
        new_vox_sz = (2.0, 2.0, 2.0)
        data, affine = reslice(data, affine, old_vox_sz, new_vox_sz)

    dataMask = nib.load(file_inMask)
    mask = dataMask.get_data()

    csa_model = CsaOdfModel(gtab, sh_order=6)
    csa_peaks = peaks_from_model(csa_model, data, default_sphere,
                                 relative_peak_threshold=.7,
                                 min_separation_angle=45, mask=mask)

    from dipy.tracking.local import ThresholdTissueClassifier

    classifier = ThresholdTissueClassifier(csa_peaks.gfa, .25)

    from dipy.tracking import utils

    #seed_mask = labels == 2
    seeds = utils.seeds_from_mask(mask, density=[2, 2, 2], affine=affine)

    from dipy.tracking.local import LocalTracking

    # Initialization of LocalTracking. The computation happens in the next step.

    #San Jose Data
    streamlines = LocalTracking(csa_peaks, classifier, seeds, affine, step_size=2)

    streamlines = [s for s in streamlines if s.shape[0]>20]

    streamlines = list(streamlines)
    save_trk("CSA_deterministic.trk", streamlines, affine, white_matter.shape)

    print("End: " + time.strftime("%H:%M:%S"))

    print("Start: " + time.strftime("%H:%M:%S"))

    from dipy.reconst.csdeconv import (ConstrainedSphericalDeconvModel,
                                       auto_response)

    response, ratio = auto_response(gtab, data, roi_radius=10, fa_thr=0.2)
    csd_model = ConstrainedSphericalDeconvModel(gtab, response, sh_order=6)
    csd_fit = csd_model.fit(data, mask=mask)

    from dipy.direction import ProbabilisticDirectionGetter

    prob_dg = ProbabilisticDirectionGetter.from_shcoeff(csd_fit.shm_coeff,
                                                        max_angle=30.,
                                                        sphere=default_sphere)

    classifier = ThresholdTissueClassifier(csa_peaks.gfa, .25)

    # San Jose Data
    streamlines = LocalTracking(prob_dg, classifier, seeds, affine, step_size=2, max_cross=1)
    streamlines = [s for s in streamlines if s.shape[0]>20]

    # 3T Data
    #streamlines = LocalTracking(prob_dg, classifier, seeds, affine, step_size=1, max_cross=1)
    #streamlines = [s for s in streamlines if s.shape[0]>100]

    # Compute streamlines and store as a list.
    streamlines = list(streamlines)

    save_trk("CSD_probabilistic.trk", streamlines, affine, white_matter.shape)

    print("End: " + time.strftime("%H:%M:%S"))
    """
    return streamlines, eu.affine

"""def affine_registration(file_in, bvec, bval, outPath):
    img_DWI = nib.load(file_in)
    data_DWI = img_DWI.get_data()
    affine_DWI = img_DWI.get_affine()

    bvals, bvecs = read_bvals_bvecs(bval, bvec)
    gtab = gradient_table(bvals, bvecs)

    b0 = data_DWI[..., gtab.b0s_mask]

    mean_b0 = np.mean(b0, -1)

    MNI_T2 = nib.load(definitions.Definitions.standard)
    MNI_T2_data = MNI_T2.get_data()
    MNI_T2_affine = MNI_T2.get_affine()

    return tools.affine_registration(mean_b0, MNI_T2_data, moving_grid2world=affine_DWI, static_grid2world=MNI_T2_affine)
"""
def registrationDWItoNMI(file_in, bvec, bval, outPath):

    img_DWI = nib.load(file_in)
    data_DWI = img_DWI.get_data()
    affine_DWI = img_DWI.get_affine()

    bvals, bvecs = read_bvals_bvecs(bval, bvec)
    gtab = gradient_table(bvals, bvecs)

    b0 = data_DWI[..., gtab.b0s_mask]

    mean_b0 = np.mean(b0, -1)

    MNI_T2 = nib.load(definitions.Definitions.standard)
    MNI_T2_data = MNI_T2.get_data()
    MNI_T2_affine = MNI_T2.get_affine()

    affine, starting_affine = tools.affine_registration(mean_b0, MNI_T2_data, moving_grid2world=affine_DWI, static_grid2world=MNI_T2_affine)

    warped_moving, mapping = tools.syn_registration(mean_b0, MNI_T2_data,
                                  moving_grid2world=affine_DWI,
                                  static_grid2world=MNI_T2_affine,
                                  #step_length=0.1,
                                  #sigma_diff=2.0,
                                  metric='CC',
                                  dim=3, level_iters = [10, 10, 5],
                                  #prealign=affine.affine)
                                  prealign=starting_affine)

    return warped_moving, MNI_T2_affine, mapping

def registrationtoNMI(file_in, outPath):

    img = nib.load(file_in)
    data = img.get_data()
    affineStructural = img.get_affine()

    MNI_T2 = nib.load(definitions.Definitions.standard)
    MNI_T2_data = MNI_T2.get_data()
    MNI_T2_affine = MNI_T2.get_affine()

    affine, starting_affine = tools.affine_registration(data, MNI_T2_data, moving_grid2world=affineStructural, static_grid2world=MNI_T2_affine)

    warped_moving, mapping = tools.syn_registration(data, MNI_T2_data,
                                  moving_grid2world=affineStructural,
                                  static_grid2world=MNI_T2_affine,
                                  #step_length=0.1,
                                  #sigma_diff=2.0,
                                  metric='CC',
                                  dim=3, level_iters = [5, 5, 3],
                                  #dim=3, level_iters = [10, 10, 5],
                                  #prealign=affine.affine)
                                  prealign=starting_affine)

    return warped_moving, MNI_T2_affine, mapping

def registrationto(file_in, file_reg):

    img = nib.load(file_in)
    data = img.get_data()
    affineStructural = img.get_affine()

    MNI_T2 = nib.load(file_reg)
    MNI_T2_data = MNI_T2.get_data()
    MNI_T2_affine = MNI_T2.get_affine()

    affine, starting_affine = tools.affine_registration(data, MNI_T2_data, moving_grid2world=affineStructural, static_grid2world=MNI_T2_affine)

    warped_moving, mapping = tools.syn_registration(data, MNI_T2_data,
                                  moving_grid2world=affineStructural,
                                  static_grid2world=MNI_T2_affine,
                                  #step_length=0.1,
                                  #sigma_diff=2.0,
                                  metric='CC',
                                  dim=3, level_iters = [5, 5, 3],
                                  #dim=3, level_iters = [10, 10, 5],
                                  #prealign=affine.affine)
                                  prealign=starting_affine)

    return warped_moving, MNI_T2_affine, mapping

def registerAffine_atlas(pathAtlas, pathStandard, outPath, tempPath, affineSubject, Subject):
    atlas                   = nib.load(pathAtlas)
    atlas_data              = atlas.get_data()

    indexs = np.unique(atlas_data)

    refNameOnly = utils.extractFileName(pathAtlas)

    file_outSubject, omatSubject = fsl.FLIRT(pathStandard, tempPath + 'Aux_FLIRT' + definitions.Definitions.extension, Subject, tempPath + 'Aux_FLIRT_omat.mat')
    fsl.hexTodec(omatSubject, omatSubject + '.mat2')
    omatSubject = omatSubject + '.mat2'
    for index in indexs:
        roi = (atlas_data == index)
        nib.save(nib.Nifti1Image(roi.astype(np.float32), affineSubject), tempPath + refNameOnly + '_ROI_' + str(index) + definitions.Definitions.extension)

        fsl.FLIRT_xfm(tempPath + refNameOnly + '_ROI_' + str(index) + definitions.Definitions.extension, outPath + refNameOnly + '_ROI_' + str(index) + '_FLIRT' + definitions.Definitions.extension, Subject, omatSubject)

def register_atlas(pathAtlas, outPath, affineSubject, mapping):

    atlas                   = nib.load(pathAtlas)
    atlas_data              = atlas.get_data()

    indexs = np.unique(atlas_data)

    refNameOnly = utils.extractFileName(pathAtlas)

    for index in indexs:
        roi = (atlas_data == index)
        warped_roi = mapping.transform_inverse(roi.astype(int)*255, interpolation='nearest')

        bin_warped_roi = np.ceil(warped_roi)

        filled_warped_roi = ndim.binary_fill_holes(bin_warped_roi.astype(int)).astype(int)

        """
        warped_roi = mapping.transform_inverse(ndim.binary_dilation(roi).astype(int), interpolation='nearest')
        warped_roi = ndim.binary_erosion(warped_roi)

        bin_warped_roi = np.ceil(warped_roi)

        filled_warped_roi = ndim.binary_fill_holes(bin_warped_roi.astype(int)).astype(int)
        """
        nib.save(nib.Nifti1Image(filled_warped_roi.astype(np.float32), affineSubject), outPath + refNameOnly + '_ROI_' + str(index) + definitions.Definitions.extension)

        print("ROI # " + str(index) + " for " + refNameOnly + " Atlas, has been saved")
        if not('warped_atlas' in locals()):
           warped_atlas = np.zeros(filled_warped_roi.shape)

        warped_atlas = warped_atlas + (filled_warped_roi*index)

        f = np.unique(warped_atlas)
        print(f.shape)

        if (f.shape != (index + 1)):
            warped_atlas[warped_atlas == np.max(f)] = 0

        #print('Unique')
        #print(f)

    return warped_atlas.astype(np.int32)
    #corpus_callosum = (atlas_data == 5) | (atlas_data == 4) | (atlas_data == 3)

    #warped_corpus_callosum = mapping.transform_inverse(ndim.binary_dilation(corpus_callosum).astype(int), interpolation='nearest')

    #warped_corpus_callosum = ndim.binary_erosion(warped_corpus_callosum)

    #bin_warped_corpus_callosum = np.ceil(warped_corpus_callosum)

    #filled_warped_corpus_callosum= ndim.binary_fill_holes(bin_warped_corpus_callosum.astype(int)).astype(int)
    #nib.save(nib.Nifti1Image(filled_warped_corpus_callosum.astype(np.float32), dwiPreprocessed.get_affine()),  definitions.Definitions.pathOUT + 'warped_corpus_callosum.nii.gz')

def connectivity_matrix2(streamlines, label_volume, affine, voxel_size=None):
    from dipy.tracking._utils  import (_mapping_to_voxel, _to_voxel_coordinates)

    lin_T, offset = _mapping_to_voxel(affine, voxel_size)

    indexROI = np.unique(label_volume)
    matriz = np.zeros(shape=(len(indexROI),len(indexROI)))

    for ROI in indexROI:
        ROIimg = (label_volume == ROI)
        ROIimg = ROIimg.astype(int)

        for ROI2 in indexROI:
            if (ROI2 >= ROI ):
                ROI2img = (label_volume == ROI2)
                ROI2img = ROI2img.astype(int)

                for sl in streamlines:
                    #sl += offset
                    sl_mapping = sl.astype(int)
                    i, j, k = sl_mapping.T
                    labelsROI  = ROIimg[i, j, k]
                    labelsROI2 = ROI2img[i, j, k]

                    if ((sum(labelsROI) > 0) & (sum(labelsROI2) > 0)):
                        matriz[ROI,ROI2] = matriz[ROI,ROI2] + 1
    return matriz.astype(int)


    #for sl in streamlines:

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
    endpoints = [sl[0::len(sl)-1] for sl in streamlines]

    print(streamlines.__len__())
    print(endpoints.__len__())

    from dipy.tracking._utils  import (_mapping_to_voxel, _to_voxel_coordinates)
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

def ravel_multi_index(multi_index, dims, mode='raise', order='C'): # real signature unknown; restored from __doc__
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
