__author__ = 'Jrudascas'

import ProcessData as processData
import Definitions as definitions

def Preprocessing_DWI (file_in, outPath, ref_bo):
    ref_bo = str(ref_bo)

    process = []

    #directory = outPath + utils.extractFileName(file_in) + '/'
    #if not os.path.exists(directory): os.makedirs(directory)

    process.append(processData.Reslice_DWI(file_in, outPath, definitions.Definitions.vox_sz))

    process.append(processData.EddyCorrect_DWI(process[process.__len__() - 1], outPath, ref_bo))

    process.append(processData.NonLocalMean_DWI(process[process.__len__() - 1], outPath, definitions.Definitions.threshold, int(ref_bo)))

    process.append(processData.Median_Otsu_DWI(process[process.__len__() - 1], outPath, definitions.Definitions.median_radius, definitions.Definitions.num_pass))

    return process