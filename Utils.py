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

