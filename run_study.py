__author__ = 'Jrudascas'

import warnings

warnings.filterwarnings("always")

import os
import main as m
import definitions as d
import utils as u

lstFiles = []
lstDir = os.walk(d.path_input)

for group in sorted(os.listdir(d.path_input)):
    path_group = os.path.join(d.path_input, group)
    if os.path.isdir(path_group):
        for subject in sorted(os.listdir(path_group)):
            path_subject = os.path.join(path_group, subject)
            if os.path.isdir(path_subject):

                if not (os.path.exists(os.path.join(d.path_output, u.to_extract_foldername(d.path_input)))):
                    os.mkdir(os.path.join(d.path_output, u.to_extract_filename_extention(d.path_input)))

                if not (os.path.exists(os.path.join(d.path_output, group))):
                    os.mkdir(os.path.join(d.path_output, group))

                if not (os.path.exists(os.path.join(os.path.join(d.path_output, group), subject))):
                    os.mkdir(os.path.join(os.path.join(d.path_output, group), subject))

                m.run_main(path_subject, os.path.join(
                    os.path.join(os.path.join(d.path_output, u.to_extract_filename_extention(d.path_input)), group),
                    subject) + '/')
