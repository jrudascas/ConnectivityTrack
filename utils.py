__author__ = 'Jrudascas'

import os
from os import listdir
from os.path import isfile, join


def to_extract_filename_extention(path_input):
    return path_input.split('/')[path_input.split('/').__len__() - 1]


def to_extract_foldername(path_input):
    return path_input.split('/')[path_input.split('/').__len__() - 2]


def to_extract_filename(path_input):
    refName = to_extract_filename_extention(path_input)
    return refName.split('.')[0]


def to_extract_extension(path_input):
    refName = to_extract_filename_extention(path_input)
    return refName.split('.')[1]


def to_delete_files(folder):
    files_dump = [join(folder, c) for c in listdir(folder)]
    files_dump = filter(lambda c: isfile(c), files_dump)
    [os.remove(c) for c in files_dump]


def to_validate_extention(path, extList):
    if to_extract_extension(path) in extList:
        return True
    else:
        return False


def what_kind_neuroimage_is(path):
    import nibabel as nib

    def is_t1(path):
        try:
            image = nib.load(path)
        except:
            return False

        if len(image.shape) == 3:
            return True
        else:
            return False

    def is_dwi(path):
        try:
            image = nib.load(path)
        except:
            return False

        if len(image.shape) == 4 and image.shape[-1] > 1 and image.shape[-1] < 50:
            return True
        else:
            return False

    if to_extract_extension(path) == 'bvec':
        return 'bvec'
    elif to_extract_extension(path) == 'bval':
        return 'bval'
    elif to_validate_extention(path, ['nii', 'gz']):
        if is_t1(path):
            return 't1'
        elif is_dwi(path):
            return 'dwi'
        else:
            return 'unknown'
