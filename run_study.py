__author__ = 'Jrudascas'

import os
import main as m
import definitions as d

lstFiles = []
lstDir = os.walk(d.path_input)

for root, dirs, files in lstDir:
    for dir in dirs:

        if not (os.path.exists(d.path_output + os.sep + dir)):
            os.mkdir(d.path_output + os.sep + dir)
        print("Finding into " + d.path_input + os.sep + dir + os.sep)
        m.run_main(d.path_input + os.sep + dir + os.sep, d.path_output + os.sep + dir + os.sep)