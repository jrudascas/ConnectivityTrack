__author__ = 'Jrudascas'

import os

def FLIRT (file_in, file_out, ref, omat):
    command = 'flirt -in ' + file_in + ' -out ' + file_out + ' -ref ' + ref + ' -omat ' + omat + ' -dof 6'
    os.system(command)
    #os.wait()

    #os.system("flirt -in /home/jrudascas/Desktop/DWITest/Additionals/Standards/MNI152_T1_2mm_brain.nii.gz -out /home/jrudascas/Desktop/DWITest/Temporal/Aux_FLIRT.nii.gz -ref /home/jrudascas/Desktop/DWITest/Datos_Salida/20160301_113321TRACTOCOMPL4s012a1001_Reslice_EddyCorrect_NonLocalMean_MedianOtsu_b0.nii.gz -omat /home/jrudascas/Desktop/DWITest/Temporal/Aux_FLIRT_omatXXXXXXX.mat -dof 6")
    print(command)
    return file_out, omat

def FLIRT_xfm (file_in, file_out, ref, omat):
    #flirt -in newvol -ref refvol -out outvol -init invol2refvol.mat -applyxfm
    command = 'flirt -in ' + file_in + ' -out ' + file_out + ' -ref ' + ref + ' -init ' + omat + ' -applyxfm'
    os.system(command)
    print(command)
    return file_out, omat

def BET (file_in, file_out, parameters):
    command = 'bet ' + file_in + ' ' + file_out + ' ' + parameters
    os.system(command)

def EDDYCORRECT (file_in, file_out, referenceNo):
    command = 'eddy_correct ' + file_in + ' ' + file_out + ' ' + referenceNo
    print(command)
    os.system(command)

def hexTodec (file, file_out):
    command = 'sh /home/jrudascas/Desktop/DWITest/Additionals/Scripts/hexTodec ' + file + ' > ' + file_out
    os.system(command)

"""
    Transformacion Lineal

flirt -in /home/brain/DTI/Experiment/Share/Datos_Entrada/20150605_100402im0TRACTOGRAFIA10s011a1001bet.nii.gz -ref /home/brain/DTI/Experiment/Share/Datos_Entrada/MNI152_T1_2mm_brain.nii.gz -out /home/brain/DTI/Experiment/Share/Datos_Entrada/20150605_100402im0TRACTOGRAFIA10s011a1001FLIRT.nii.gz -omat /home/brain/DTI/Experiment/Share/Datos_Entrada/transformation.mat -dof 6

Transformacion No Lineal

fnirt --ref=/home/brain/DTI/Experiment/Share/Datos_Entrada/MNI152_T1_2mm_brain.nii.gz --in=/home/brain/DTI/Experiment/Share/Datos_Entrada/20150605_100402im0TRACTOGRAFIA10s011a1001bet.nii.gz --iout=/home/brain/DTI/Experiment/Share/Datos_Entrada/20150605_100402im0TRACTOGRAFIA10s011a1001FNIRT.nii.gz --aff=/home/brain/DTI/Experiment/Share/Datos_Entrada/transformation.mat --cout=warp_struct2mni.nii

Construir Transformacion Inversa

convert_xfm -omat /home/brain/DTI/Experiment/transformation_T1Inverse.mat -inverse /home/brain/DTI/Experiment/transformation_T1.mat

invwarp --ref=/home/brain/DTI/Experiment/Share/Datos_Entrada/20150605_100402im0TRACTOGRAFIA10s011a1001bet.nii.gz --warp=/home/brain/DTI/Experiment/Share/Datos_Entrada/warp_struct2mni.nii.gz --out=/home/brain/DTI/Experiment/Share/Datos_Entrada/warp_struct2mniInverse.nii.gz

applywarp --ref=/home/brain/DTI/Experiment/Share/Datos_Entrada/20150605_100402im0TRACTOGRAFIA10s011a1001bet.nii.gz --in=/home/brain/DTI/Experiment/Share/Datos_Entrada/TestLopez.nii.gz --out=/home/brain/DTI/Experiment/Share/Datos_Entrada/TestLopezFNIRPrealing.nii.gz --warp=/home/brain/DTI/Experiment/Share/Datos_Entrada/warp_struct2mniInverse.nii.gz --interp=nn

"""