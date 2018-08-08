__author__ = 'Jrudascas'

path_temporal = '/home/jrudascas/Desktop/DWITest/Temporal/'
# pathIN                 = '/home/jrudascas/Desktop/DWITest/Datos_Entrada/'
# pathOUT                = '/home/jrudascas/Desktop/DWITest/Datos_Salida/'
standardT2 = '/home/jrudascas/Desktop/DWITest/Additionals/Standards/MNI152_T2_1mm_brain.nii.gz'
standardT1 = '/home/jrudascas/Desktop/DWITest/Additionals/Standards/MNI152_T1_1mm_brain.nii.gz'
brain_mask_nmi = '/usr/share/fsl/data/standard/MNI152_T1_1mm_brain_mask.nii.gz'
# atlas                  = '/home/jrudascas/Desktop/DWITest/Additionals/Atlas/JHU-ICBM-labels-2mm.nii.gz'
# '/home/jrudascas/Desktop/DWITest/Additionals/Atlas/2mm/AAN_2mm.nii'
# '/home/jrudascas/Desktop/DWITest/Additionals/Atlas/JHU-ICBM-labels-2mm.nii.gz'
atlas = '/home/jrudascas/Desktop/DWITest/Additionals/Atlas/1mm/AAN_1mm.nii'
# atlas                  = '/home/jrudascas/Desktop/DWITest/Additionals/Atlas/HarvardOxford-cort-maxprob-thr25-2mm.nii.gz'
defaultb0Reference = 0
extension = '.nii.gz'
idEddyCorrect = '_EddyCorrect'
idReslice = '_Reslice'
idNonLocalMean = '_NonLocalMean'
idMedianOtsu = '_MedianOtsu'
id_bet = '_BET'
idEvecs = '_DTIEvecs'
idEvals = '_DTIEvals'
separador = '    - '
vox_sz = 1.0
threshold = 100
median_radius = 4
num_pass = 4
