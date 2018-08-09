__author__ = 'Jrudascas'

import warnings
warnings.filterwarnings("always")

import os
import main as m
import definitions as d

lstFiles = []
lstDir = os.walk(d.path_input)

for subject in sorted(os.listdir(d.path_input)):
    if os.path.isdir(os.path.join(d.path_input, subject)):
        if not (os.path.exists(os.path.join(d.path_output, subject))):
            os.mkdir(os.path.join(d.path_output, subject))

        print(subject)
        m.run_main(os.path.join(d.path_input, subject, os.sep), os.path.join(d.path_output, subject, os.sep))
