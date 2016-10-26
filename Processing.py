__author__ = 'brain'

import ProcessData as processDatas

def Processing_DWI (file_in, file_inMask, outPath, fbval, fbvec, mapping, affine_mapping):
    listev = processData.DTIModel(file_in, file_inMask, outPath, fbval, fbvec)

    return processData.EstimateMapsDTI(file_in, file_inMask, listev[0], listev[1], outPath, fbvec, fbval, mapping, affine_mapping)