__author__ = 'Jrudascas'

import warnings
warnings.filterwarnings("always")

import os
import main as m
import definitions as d

lstFiles = []
lstDir = os.walk(d.path_input)

for root, dirs, files in lstDir:
    for dir in dirs:

        if not (os.path.exists(os.path.join(d.path_output, dir))):
            os.mkdir(os.path.join(d.path_output, dir))

        print(dir)
        m.run_main(os.path.join(d.path_inputput, dir), os.path.join(d.path_output, dir))
