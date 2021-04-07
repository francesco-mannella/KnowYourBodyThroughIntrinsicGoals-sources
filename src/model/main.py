#!/usr/bin/python
# -*- coding: utf-8 -*-
"""

The MIT License (MIT)

Copyright (c) 2015 Francesco Mannella <francesco.mannella@gmail.com> 

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""


import os
import sys
import copy
import pickle
import gzip

# working dir is the base dir of this file
pathname = os.path.dirname(sys.argv[0])
if pathname: os.chdir(pathname)

sys.path.append("../")

################################################################
################################################################
# To force stop on exceptions

import traceback
def my_excepthook(type, value, tback): 
    traceback.print(_exception(type,value,tback) )
    sys.exit(1)

sys.excepthook = my_excepthook

################################################################
################################################################

import numpy as np
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
    parser.add_argument('-d','--dump',
            help="dump the robot object",
            action="store_true", default=False) 
    parser.add_argument('-l','--load',
            help="load the robot object",
            action="store_true", default=False) 
    parser.add_argument('-s','--save_dir',
            help="storage directory",
            action="store", default="../../")      
    parser.add_argument('-t','--stime',
            help="Simulation time (only for graphics off)",
            action="store", default=2000)  
    args = parser.parse_args()
    GRAPHICS = bool(args.graphics) 
    STIME = int(args.stime)  
    SDIR = args.save_dir
    if SDIR[-1]!='/': SDIR += '/'
    SDIR=os.getcwd()+'/'+SDIR
     
    DUMP = int(args.dump) 
    LOAD = int(args.load) 
   
    log_sensors = open(SDIR+"log_sensors", "w")
    log_position = open(SDIR+"log_position", "w")
    log_predictions = open(SDIR+"log_predictions", "w")
    log_targets = open(SDIR+"log_targets", "w")
    log_weights = open(SDIR+"log_weights", "w")
    

    dumpfile = SDIR+"dumped_robot"
    
    if LOAD :
        print("loading ...")
        with gzip.open(dumpfile, 'rb') as f:
            robot = pickle.load(f)
    else :
        robot = Robot()

    robot.log_sensors = log_sensors
    robot.log_position = log_position
    robot.log_predictions = log_predictions
    robot.log_targets = log_targets
    robot.log_weights = log_weights
    
    print("simulating ...")
    if GRAPHICS :
        import plotter 
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

    if DUMP :
        
        print("dumping ...")
        with gzip.open(dumpfile, 'wb') as f:
            robot.init_streams()
            robot = pickle.dump(robot, f)


