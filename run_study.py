__author__ = 'Jrudascas'

import warnings

warnings.filterwarnings("always")

import os
import main as m
import definitions as d
import utils as u
import numpy as np

lstFiles = []
lstDir = os.walk(d.path_input)

for group in sorted(os.listdir(d.path_input)):
    path_input_group = os.path.join(d.path_input, group)

    features_list_group = []

    if os.path.isdir(path_input_group):
        for subject in sorted(os.listdir(path_input_group)):
            path_input_subject = os.path.join(path_input_group, subject)
            if os.path.isdir(path_input_subject):

                path_output_study = os.path.join(d.path_output, u.to_extract_foldername(d.path_input))
                if not (os.path.exists(path_output_study)):
                    os.mkdir(path_output_study)

                path_output_group = os.path.join(path_output_study, group)
                if not (os.path.exists(path_output_group)):
                    os.mkdir(path_output_group)

                path_output_subject = os.path.join(path_output_group, subject)
                if not (os.path.exists(path_output_subject)):
                    os.mkdir(path_output_subject)

                features_list = m.run_main(path_input_subject, path_output_subject + '/')

                features_list_group.append(features_list)

    np.savetxt(path_output_group + 'features.out', np.array(features_list_group), delimiter=' ', fmt='%s')
