__author__ = 'Jrudascas'

import warnings
warnings.filterwarnings("always")

import main as main
import definitions as d
import time

t = time.time()

main.run_main(d.path_input, d.path_output)

print("Total duration: " + str(time.time() - t) + " sec")
