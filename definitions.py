__author__ = 'Jrudascas'

import os

path_input = '/home/runlab/data/SanJose/subject5/'
path_output = '/home/runlab/data/results/SanJose/'

path_temporal = '/home/runlab/data/results/Temporal/'

atlas = '/home/runlab/data/Atlas/1mm/AAN_1mm.nii'
aanAtlas = '/home/runlab/data/Atlas/1mm/1mm/AAN.nii'
morelAtlas = '/home/runlab/data/Atlas/1mm/ThalamicNucleiMorelAtlas.nii'
harvardOxfordCortAtlas = '/home/runlab/data/Atlas/1mm/HarvardOxfordCort.nii'
hypothalamusAtlas = '/home/runlab/data/Atlas/1mm/Hypothalamus.nii'

standardT2 = os.path.join(os.environ['FSLDIR'], 'data/standard/MNI152_T2_1mm_brain.nii.gz')
standardT1 = os.path.join(os.environ['FSLDIR'], 'data/standard/MNI152_T1_1mm_brain.nii.gz')
brain_mask_nmi = os.path.join(os.environ['FSLDIR'], 'data/standard/MNI152_T1_1mm_brain_mask.nii.gz')

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
