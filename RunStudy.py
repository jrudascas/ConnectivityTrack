__author__ = 'Jrudascas'

import os
import Main as main
import Utils as utils
pathInStudy = "/home/jrudascas/Desktop/TestMichael/In/"
pathOutStudy = "/home/jrudascas/Desktop/TestMichael/Out/"

lstFiles = []
lstDir = os.walk(pathInStudy)

for root, dirs, files in lstDir:
    for dir in dirs:

        if not (os.path.exists(pathOutStudy + os.sep + dir)):
            os.mkdir(pathOutStudy + os.sep + dir)
        print("Finding into " + pathInStudy + os.sep + dir + os.sep)
        main.Main(pathInStudy + os.sep + dir + os.sep, pathOutStudy + os.sep + dir + os.sep)