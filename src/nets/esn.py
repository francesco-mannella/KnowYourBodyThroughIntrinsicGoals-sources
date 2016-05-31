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

import numpy as np
import math
import time


def evaluate_rho(m):

    if m.shape [0] != m.shape[1] :
        raise ValueError( "evaluate_rho(m) requires square matrix" );

    # find eigenvalues
    eigenvalues = np.linalg.eigvals(m)

    # the spectral radius is the maximum between the
    # absolute values of eigenvalues.
    return  np.abs(eigenvalues).max();


#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
#-- RESERVOIR ---------------------------------------------------------------
#----------------------------------------------------------------------------
#----------------------------------------------------------------------------

class ESN(object):
    '''
    A firing-rate esn.
    Internal weights are normalized following the echo-state algorithm
    plus a correction of the infinitesimal rotation/expansion ratio.
    The output function of all units is a positive-truncated tanh.
    '''

    def __init__(
            self,
            name         = "r",
            stime        = 1000,
            dt           = 0.001,
            tau          = 0.01,
            N            = 100,
            alpha        = 0.5,
            beta         = 0.5,
            epsilon      = .5e-8,
            sparseness   = 1.0,
            weight_mean  = 0.0,
            th           = 0.0,
            amp          = 1.0,
            radius_amp   = 1.0,
            trunk        = True,
            noise        = False,
            noise_std    = 0.1
            ) :
        """
            name         string:    name of the object
            stime        float:     time of storage interval (number of timesteps)
            dt           float:     integration step (sec)
            tau          float:     decay of leaky units (sec)
            N            int:       number of units
            alpha        float:     infinitesimal expansion
            beta         float:     infinitesimal rotation
            epsilon      float:     epsilon - spectral radius between is  epsilon and 1
            sparseness   float:     sparseness of weights
            weight_mean  float:     initial mean of random weights
            th           float:     output threshold
            amp          float:     output amplitude
            radius_amp   float:     spectral radius amplification after normalization
            trunk        bool:      output function trunctation
            noise        bool:      internal gaussian noise
            noise_std    float:     standard deviation of internal noise
        """

        # Consts
        self.NAME = name
        self.DT = dt
        self.TAU = tau
        self.STIME = int(stime)
        self.N = N
        self.ALPHA = alpha
        self.BETA = beta
        self.TH = th
        self.AMP = amp
        self.RADIUS_AMP = radius_amp
        self.TRUNK = trunk
        self.NOISE = noise
        self.NOISE_STD = noise_std
        self.SPARSENESS = sparseness
        self.WEIGHT_MEAN = weight_mean

        # Variables
        self.pot = np.zeros(self.N)    # potentials
        self.out = np.zeros(self.N)    # outputs
        self.w = np.zeros([self.N, self.N])    # inner weights

        self.normalize_to_echo( epsilon)

        # define labels
        (
            self.pot_lab,
            self.out_lab,
            self.inp_lab,
            self.w_norm_lab,
            self.out_mean_lab
        ) = range(5)

        self.data = dict()    # dictionary of stores

        # initialize to zeros
        self.data[self.pot_lab] = np.zeros([self.N, self.STIME])
        self.data[self.out_lab] = np.zeros([self.N, self.STIME])
        self.data[self.inp_lab] = np.zeros([self.N, self.STIME])
        self.data[self.w_norm_lab] = np.zeros(self.STIME)
        self.data[self.out_mean_lab] = np.zeros(self.STIME)

    def normalize_to_echo(self, epsilon) :
        '''
        find the optimized spectral radius of W so that rho(1-epsilon) < Wd  < 1,
        where Wd = (dt/tau)*W+(1-(dt/tau))*eye. See Proposition 2 in Jaeger et al. (2007) http://goo.gl/bqGAJu.

        epsilon    float:      spectral radius must be between epsilon and 1
        '''

        self.w = self.randomize()

        # normalize W so that rho = 1
        self.w = self.normalize(self.w, 1.0)

        # the iteration has to reach this value
        target = 1.0 - epsilon/2.0

        # integration step (dt/tau)
        if np.isscalar(self.TAU):
            h = self.DT/self.TAU
        else:
            h = self.DT/self.TAU.max()

        # convenience alias for the identity matrix
        I = np.eye(self.N,self.N)

        e = np.linalg.eigvals(self.w)
        rho = abs(e).max()
        x = e.real
        y = e.imag

        # solve quadratic equations
        a = x**2*h**2 + y**2*h**2
        b = 2*x*h - 2*x*h**2
        c = 1 + h**2 - 2*h - target**2
        # just get the positive solutions
        sol = (-b + np.sqrt(b**2 - 4*a*c))/(2*a)
        # and take the minor amongst them
        effective_rho = sol.min()

        self.w *= effective_rho

    def randomize(self) :
        '''
        Make a random sparse matrix with modulated infinitesimal rotation and translation
        '''

        # random sparse weigths
        M = (np.random.randn(self.N, self.N) + self.WEIGHT_MEAN )* (
                (np.random.rand(self.N, self.N) < self.SPARSENESS) )

        # decompose rotation and translation
        M = self.ALPHA*(M+ M.T) + self.BETA*(M - M.T)

        return M

    def normalize(self, M, rho) :
        '''
        Normalize the matrix M so that the spectral radius is rho
        '''
        # normalize to spectral radius 1
        return rho * M / evaluate_rho(M)

    def reset(self):
        '''
        Reset internal units to the initial state
        '''
        self.pot *= 0
        self.out *= 0

    def step(self, inp):
        '''
        Activation step
        inp     array:  external input
        '''

        self.inp = inp

        # Collect inputs
        increment = np.dot(self.w, self.out) + inp

        if  self.NOISE :
            increment +=  self.NOISE_STD*np.random.rand(self.N)

        # Integrate
        self.pot += (self.DT/self.TAU) * \
                (
                - self.pot
                + increment
                )

        # Transfer function
        self.out = np.tanh(self.AMP*(self.pot-self.TH))
        if self.TRUNK :
            self.out = np.maximum(0, self.out)

    def store(self, tt):
        '''
        Store activity
        '''
        t = tt%self.STIME
        self.data[self.pot_lab][:, t] = self.pot
        self.data[self.out_lab][:, t] = self.out
        self.data[self.inp_lab][:, t] = self.inp
        self.data[self.w_norm_lab][t] = np.linalg.norm(self.w)
        self.data[self.out_mean_lab][t] = self.out.mean()

    def reset_data(self):
        '''
        Reset stores
        '''
        for k in self.data :
            self.data[k] = self.data[k]*0

#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
#-- TEST --------------------------------------------------------------------
#----------------------------------------------------------------------------
#----------------------------------------------------------------------------


if __name__ == "__main__":

    import matplotlib.pyplot as plt
    
    plt.ion()

    N = 200
    N_OUT = 2
    TRIALS = 200 
    ETA = 50000.0

    echo = ESN(
        N       = N,
        stime   = 200,
        dt      = 1.0,
        tau     = 20.0,
        alpha   = 0.1,
        beta    = 0.9,
        epsilon = 1.0e-5
        )

    ro_w = 0.01*np.random.randn(N_OUT,N)
    
    ro_d = np.zeros([echo.STIME, N_OUT])

    target = np.array( [
        [0.5, 0.25],
        [0.5, 0.5 ],
        [0.25,0.5 ],
        [0.25,0.25]])


    fig = plt.figure(figsize=(5,10))
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)

    ax1.scatter(*np.array(target).T, s=50, color="red")
    line, = ax1.plot(0,0)
    start, = ax1.plot(0,0,marker="o", mfc="blue")
    end, = ax1.plot(0,0, ms= 15, marker="o", mfc="green")
    ax1.set_xlim([0,1.5])
    ax1.set_ylim([0,1.5])
    
    echo_activity = ax2.imshow(np.zeros([echo.STIME, N]),
            interpolation='none', aspect='auto', vmin=0, vmax=1)
   
    
    l_target = len(target)

    inp_w = 0.5*np.ones([N, l_target])*(np.random.rand(N, l_target)<0.1)
    sigma = 0.6
    rout = np.zeros(N_OUT)
    rout_mean = np.zeros(N_OUT)
   
    rout_real = np.zeros(N_OUT) 
    for k in xrange(TRIALS) :
        for t in xrange(echo.STIME) :
            
            mask = np.zeros(l_target)
            for i in range(l_target) :
                if  echo.STIME*(i/float(l_target)) <= t \
                        < echo.STIME*((i+1)/float(l_target)) :
                    mask[i] = 1

            dist = np.linalg.norm(target - rout_real, axis=1) 
            inp = np.exp(-(dist**2)/(2*sigma**2))*mask
            
            echo.step(np.dot(inp_w, inp))
            echo.store(t)
        
            rout = echo.out[:N_OUT] 
            rout_real = rout + 0.5*np.random.rand(2)
            rout_mean += 0.1*(rout_real - rout_mean)
            ro_d[t] = rout_real
           
            if k < TRIALS/2 :
                w = echo.w[:N_OUT,:]
                # tg_idx = np.where(inp == max(inp))
                # tg = target[tg_idx]
                rew = sum(inp>0.1)
                w += ETA*rew*np.outer(rout - rout_mean, echo.out) 
           
        line.set_data(*ro_d.T)
        start.set_data(*ro_d[0])
        end.set_data(*ro_d[-1])
        echo_activity.set_array(echo.data[echo.out_lab])
        fig.canvas.draw()
        plt.pause(.001)

    raw_input("press for single test")

    for t in xrange(echo.STIME) :
                
        mask = np.zeros(l_target)
        for i in range(l_target) :
            if  echo.STIME*(i/float(l_target)) <= t \
                    < echo.STIME*((i+1)/float(l_target)) :
                mask[i] = 1

        dist = np.linalg.norm(target - rout, axis=1) 
        inp = np.exp(-(dist**2)/(2*sigma**2))*mask
        
        echo.step(np.dot(inp_w, inp))
        echo.store(t)
    
        rout = echo.out[:N_OUT] 
        ro_d[t] = rout
        
        line.set_data(*ro_d[:t].T)
        start.set_data(*ro_d[0])
        end.set_data(*ro_d[t-1])
        echo_activity.set_array(echo.data[echo.out_lab][:,:t])
        fig.canvas.draw()
        plt.pause(.01)
 
    raw_input("press to end")

