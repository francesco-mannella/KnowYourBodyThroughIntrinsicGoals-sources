#!/usr/bin/python
# -*- coding: utf-8 -*-


import os
import sys
import copy

# working dir is the base dir of this file
pathname = os.path.dirname(sys.argv[0])
if pathname: os.chdir(pathname)

sys.path.append("../")

################################################################
################################################################
# To force stop on exceptions

import traceback
def my_excepthook(type, value, tback): 
    traceback.print_exception(type,value,tback) 
    sys.exit(1)

sys.excepthook = my_excepthook

################################################################
################################################################

import numpy as np
import plotter 
from Robot import Robot

np.set_printoptions(edgeitems=3, linewidth=999,  precision=3,
        suppress=True, threshold=1000)


#################################################################
#################################################################

import progressbar

## Start Qt event loop unless running in interactive mode.
if __name__ == '__main__':
   
    import argparse
    parser = argparse.ArgumentParser() 
    parser.add_argument('-g','--graphics',
            help="Graphics on",
            action="store_true", default=False) 
    parser.add_argument('-t','--stime',
            help="Simpulation time (only for graphics off)",
            action="store", default=2000)  
    args = parser.parse_args()
    GRAPHICS = bool(args.graphics) 
    STIME = int(args.stime) 

    robot = Robot()

    if GRAPHICS :

        plotter.graph_main(robot)
        
    else:
        bar = progressbar.ProgressBar( 
                maxval=STIME, 
                widgets=[progressbar.Bar('=', '[', ']'), 
                    ' ', progressbar.Percentage()],
                term_width=30)
        
        bar.start()
        for t in range(STIME):
            robot.step()
            bar.update(t+1)
        bar.finish()

