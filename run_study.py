__author__ = 'Jrudascas'

import os
import main as main
import utils as utils3
pathInStudy = "/home/jrudascas/Desktop/MONTOYA/In/"
pathOutStudy = "/home/jrudascas/Desktop/MONTOYA/Out/"

lstFiles = []
lstDir = os.walk(pathInStudy)

for root, dirs, files in lstDir:
    for dir in dirs:

        if not (os.path.exists(pathOutStudy + os.sep + dir)):
            os.mkdir(pathOutStudy + os.sep + dir)
        print("Finding into " + pathInStudy + os.sep + dir + os.sep)
        main.Main(pathInStudy + os.sep + dir + os.sep, pathOutStudy + os.sep + dir + os.sep)