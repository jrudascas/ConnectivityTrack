__author__ = 'Jrudascas'

import os
from os import listdir
from os.path import isfile, join

def extractFileNameExt (file_in):
    return file_in.split('/')[file_in.split('/').__len__() - 1]

def extractFileName (file_in):
    refName = file_in.split('/')[file_in.split('/').__len__() - 1]
    return refName.split('.')[0]

def extractExtensionFile (file_in):
    refName = file_in.split('/')[file_in.split('/').__len__() - 1]
    return refName.split('.')[1]

def delete_Files(folder):
      files_dump = [join(folder, c) for c in listdir(folder)]
      files_dump = filter(lambda c: isfile(c), files_dump)
      [os.remove(c) for c in files_dump]

def toValidateExtention(path, extList):
    if extractExtensionFile(path) in extList:
        return True
    else:
        return False

def whatKindFileIs(path):
    import nibabel as nib

    def isT1(path):
        try:
            image = nib.load(path)
        except:
            return False

        if len(image.shape) == 3:
            return True
        else:
            return False

    def isDWI(path):
        try:
            image = nib.load(path)
        except:
            return False

        if len(image.shape) == 4 and image.shape[-1] > 1 and image.shape[-1] < 50:
            return True
        else:
            return False

    if extractExtensionFile(path) == 'bvec':
        return 'bvec'
    elif extractExtensionFile(path) == 'bval':
        return 'bval'
    elif toValidateExtention(path, ['nii', 'gz']):
        if isT1(path):
            return 't1'
        elif isDWI(path):
            return 'dwi'
        else: return 'unknown'




