#!/usr/bin/env python 
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

import sys
sys.path.append("../")

import numpy as np
import numpy.random as rnd
import time
import os


from utils.gauss_utils import *

#------------------------------------------------------------
GaussianMaker = OptimizedGaussianMaker

class Kohonen(object) :

    def __init__(self,
            stime               = 400,
            n_dim_out           = 1,
            bins                = 1,
            n_output            = 25,
            n_input             = 10,
            neighborhood        = 30,
            neighborhood_decay  = 300,
            neighborhood_bl     = 1.0,
            eta                 = 0.1,
            eta_decay           = 300,
            eta_bl              = 0.001,
            weight_bl           = 0.00,
            average_decay       = 0.1,
            normalize           = lambda x : x
            ) :  
        """ 
            stime                  (int): time of simulation
            n_dim_out              (int): number of dimensions of output topology
            bins               list(int): list of bins for each dimension
            n_output               (int): number of output units
            n_input                (int): number of input elements
            neighborhood           (int): radius of neighbor-to-winner output units
            neighborhood_decay   (float): neighborhood decay
            neighborhood_bl      (float): neighborhood bl
            eta                  (float): learning rate
            eta_decay            (float): learning rate decay
            average_decay        (float): decat of the raw output moving average
            normalize         (function): normalizing function
        """
        
        # time-step counter
        self.t = 0
        
        self.STIME = stime
        self.ETA = eta
        self.ETA_BL = eta_bl
        self.ETA_DECAY = eta_decay
        self.N_DIM_OUT = n_dim_out
        self.normalize = normalize
        self.AVERAGE_DECAY = average_decay
       
        if np.isscalar(bins) :
            self.BINS = np.ones(self.N_DIM_OUT)*bins 
        else :
            self.BINS = bins 

        self.N_OUTPUT = n_output
        self.N_INPUT = n_input
        self.neighborhood = neighborhood
        self.neighborhood_DECAY = neighborhood_decay
        self.neighborhood_BL = neighborhood_bl

        self.inp = np.zeros(self.N_INPUT)
        self.out = np.zeros(self.N_OUTPUT)
        self.out_raw = np.zeros(self.N_OUTPUT)
        self.inp_min = 0
        self.inp_max = 0

         
        lims = [ [0,self.BINS[x]-1,self.BINS[x]]
                for x in range(self.N_DIM_OUT) ]
        self.gmaker = GaussianMaker(lims)
        
        # timing
        self.t = 0

        # initial weights  
        self.inp2out_w = weight_bl*rnd.randn(self.N_OUTPUT,self.N_INPUT)
        
        # data storage
        self.data = dict()
        self.l_inp = 0
        self.l_out = 1
        self.l_out_raw = 2
        self.data[self.l_inp] = np.zeros([self.N_INPUT,self.STIME])
        self.data[self.l_out] = np.zeros([self.N_OUTPUT,self.STIME])
        self.data[self.l_out_raw] = np.zeros([self.N_OUTPUT,self.STIME])

    def step(self, inp) :
        """ spreading """

        self.t += 1
        
        # Current neighbourhood
        curr_neighborhood =  self.neighborhood_BL + self.neighborhood * \
                np.exp(-self.t/float(self.neighborhood_DECAY))     

        # # input
        if not all(inp==0) :

            x = self.normalize(inp)
            self.inp = x
            w = self.inp2out_w
            #print y.shape
            y = np.dot(w,x) -0.5*np.diag(np.dot(w,w.T))

            # Calculate neighbourhood
            max_index = np.argmax(y) # index of maximum
            self.idx = max_index

            # output:
            self.out_raw = y
            point = map1DND(max_index, self.N_DIM_OUT, self.BINS)
            self.out,_ = self.gmaker(point,
                    np.ones(self.N_DIM_OUT)*(curr_neighborhood**2))
        else:
            x = np.zeros(inp.shape)
            self.out = np.zeros(self.N_OUTPUT)

    def learn(self) :
        """ Learning step """

        eta = self.ETA_BL + self.ETA* np.exp(-self.t/self.ETA_DECAY)
        
        # Update weights
        x = self.inp
        y = self.out
        w = self.inp2out_w 

        w += eta* (np.outer(y,x) -  np.outer(y, np.ones(self.N_INPUT)) *w )

    def store(self):
        """ storage """

        tt = self.t%self.STIME
        self.data[self.l_inp][:,tt] = self.inp
        self.data[self.l_out][:,tt] = self.out

        out = self.out_raw*2
        datax = self.data[self.l_out_raw]
        win = self.idx

        datax[win,tt] = self.AVERAGE_DECAY*out[win]
        if tt >0 :
            datax[:,tt] += datax[:,tt-1]
            datax[win,tt] -= self.AVERAGE_DECAY*datax[win,tt-1]

    
    def reset_data(self):
        """ Reset """

        for k in self.data :
            self.data[k] = self.data[k]*0
            



