__author__ = 'brain'


class Definitions(object):
    tempPath        = '/home/jrudascas/Desktop/DWITest/Temporal/'
    pathIN          = '/home/jrudascas/Desktop/DWITest/Datos_Entrada/'
    pathOUT         = '/home/jrudascas/Desktop/DWITest/Datos_Salida/'
    standard        = '/home/jrudascas/Desktop/DWITest/Additionals/Standards/MNI152_T1_2mm_brain.nii.gz'
    atlas           = '/home/jrudascas/Desktop/DWITest/Additionals/Atlas/JHU-ICBM-labels-2mm.nii.gz'
                     #'/home/jrudascas/Desktop/DWITest/Additionals/Atlas/2mm/AAN_2mm.nii'
                     #'/home/jrudascas/Desktop/DWITest/Additionals/Atlas/JHU-ICBM-labels-2mm.nii.gz'
                     #'/home/jrudascas/Desktop/DWITest/Additionals/Atlas/2mm/ThalamicNuclei_2mm.nii'
    extension       = '.nii.gz'
    idEddyCorrect   = '_EddyCorrect'
    idReslice       = '_Reslice'
    idNonLocalMean  = '_NonLocalMean'
    idMedianOtsu    = '_MedianOtsu'
    idEvecs         = '_DTIEvecs'
    idEvals         = '_DTIEvals'
    vox_sz          = 2.0
    threshold       = 100
    median_radius   = 4
    num_pass        = 4

