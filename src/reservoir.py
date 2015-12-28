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

from math import *
from pylab import *


#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
#-- RESERVOIR ---------------------------------------------------------------
#----------------------------------------------------------------------------
#----------------------------------------------------------------------------

class Reservoir:
    ''' 
    A firing-rate reservoir.
    
    Internal weights are normalized following the echo-state algorithm
    plus a correction of the infinitesimal rotation/expansion ratio.

    The output function of all units is a positive-truncated tanh.

    '''

    def __init__(
            self,
            name         = "r",
            stime        = 1000,
            dt           = 0.001,
            tau          = 0.04,
            N            = 20,
            alpha        = 0.2,
            beta         = 0.8,
            gamma        = 0.0,
            epsilon      = 0.00001,
            sparseness   = 1.0,
            weight_mean  = 0.0,
            th           = 0.0,
            amp          = 1.0,
            radius_amp   = 2.0,
            trunk        = True,
            noise        = False,
            noise_std    = 0.1,
            ) :
        """
            :name         string    name of the object 
            :stime        float     time of storage interval (sec)
            :dt           float     integration step (sec)
            :tau          float     decay of leaky units (sec)
            :N            int       number of units
            :alpha        float     infinitesimal expansion
            :beta         float     infinitesimal rotation
            :gamma        float     shift of real eigenvalues
            :epsilon      float     epsilon - spectral radius between
                                    epsilon and 1
            :sparseness   float     sparseness of weights
            :weight_mean  float     mean of random weights 
            :th           float     threshold to output
            :amp          float     output amplitude
            :radius_amp   float     spectral radius amplification
            :trunk        bool      output function trunctation
            :noise        bool      internal gaussian noise
            :noise_std    float     noise standard deviation
        """

        # Consts
        self.NAME = name 
        self.DT = dt 
        self.TAU = tau 
        self.STIME = int(stime)
        self.N = N 
        self.ALPHA = alpha
        self.BETA = beta
        self.GAMMA = gamma
        self.TH = th
        self.AMP = amp
        self.RADIUS_AMP = radius_amp
        self.TRUNK = trunk
        self.NOISE = noise
        self.NOISE_STD = noise_std
        self.SPARSENESS = sparseness 
        self.WEIGHT_MEAN = weight_mean

        # Variables
        self.pot = zeros(self.N)    # potentials 
        self.out = zeros(self.N)    # outputs 
        self.w = zeros([self.N, self.N])    # inner weights
        
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
        self.data[self.pot_lab] = zeros([self.N, self.STIME])
        self.data[self.out_lab] = zeros([self.N, self.STIME])
        self.data[self.inp_lab] = zeros([self.N, self.STIME])
        self.data[self.w_norm_lab] = zeros(self.STIME)
        self.data[self.out_mean_lab] = zeros(self.STIME)
  
        # optimize weights
        self.find_max_radius(epsilon)

    def find_max_radius(self, epsilon) :
        '''
       
        Iteratively find the optimized spectral radius of W so that
        epsilon < (dt/tau)*W +(1 - (dt/tau))*eye() < 1

        :epsilon    float           spectral radius to be
                                    found is between epsilon and 1

        '''        
        
        dynamic_spectral = 2    # initial value of the spectral radius of 
                                # < (dt/tau)*W +(1 - (dt/tau))*eye() >
        effective_spectral = .1    # initial value of the spectral radius of W
         
        # normalize spectral radius to 1
        self.normalize()

        # iterate to find epsilon < dynamic_spectral < 1
        while not (((1-epsilon) < dynamic_spectral) 
                and (dynamic_spectral < 1.0)) :
        
            # scale to effective_spectral
            self.wt = effective_spectral*self.scale*self.w 
            # find current dynamic_spectral 
            dynamic_spectral = max( abs( eigvals( 
                (self.DT/self.TAU)*self.wt + 
                (1 - (self.DT/self.TAU))*eye(self.N) ) ) )   
            # increase effective_spectral 
            # based oncurrent dynamic_spectral
            effective_spectral = \
                    effective_spectral/dynamic_spectral 
        
        # finally scale weights based on effective_spectral 
        # and requested amplitude
        self.w = self.RADIUS_AMP * effective_spectral * \
                self.scale * self.w 

    def normalize(self) :
        ''' 
        
        reservoir normalization leading to 
        spectral radius = 1.0
        
        '''
        
        # random sparse weigths 
        self.w = (randn(self.N, self.N) + self.WEIGHT_MEAN )* ( 
                ( rand(self.N, self.N) < self.SPARSENESS) )
        # decompose rotation and translation
        self.w = ( self.ALPHA*(self.w + self.w.T) + 
                  self.BETA*(self.w - self.w.T) +
                  self.GAMMA*eye(self.N) )
        # normalize to spectral radius 1
        self.scale = 1.0 / max(abs(eigvals(self.w)))     

    def reset(self):
        self.pot *= 0
        self.out *= 0

    def step(self, inp):  
        '''

        A single spreading step 


        inp     array   a vector of 'N' floating  
                        external inputs 

        '''

        self.inp = inp
       
        # Collect inputs
        if not self.NOISE :
            increment = ( 
                    + dot(self.w, self.out) 
                    + inp 
                    )
        else :
            increment = (
                    + dot(self.w, self.out) 
                    + self.NOISE_STD*rand(self.N)
                    + inp 
                    )

        # Integrate
        self.pot += (self.DT / self.TAU) * \
                ( \
                - self.pot 
                + increment
                )

        # Transfer function
        if self.TRUNK : 
            self.out = maximum(0, 
                    tanh(self.AMP * (self.pot-self.TH)))
        else :
            self.out = tanh(self.AMP*(self.pot-self.TH))

    def store(self, tt):
        '''
        Store activity 
        '''
        t = tt%self.STIME
        self.data[self.pot_lab][:, t] = self.pot
        self.data[self.out_lab][:, t] = self.out 
        self.data[self.inp_lab][:, t] = self.inp 
        self.data[self.w_norm_lab][t] = norm(self.w) 
        self.data[self.out_mean_lab][t] = self.out.mean()

    def reset_data(self):
        '''
        Reset stores 
        '''
        for k in self.data :
            self.data[k] = self.data[k]*0


#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
#-- MAIN --------------------------------------------------------------------
#----------------------------------------------------------------------------
#----------------------------------------------------------------------------


if __name__ == "__main__":

    close('all')

    reservoir = Reservoir( )      

    # prepare an exponenetial decay input function
    inp =  exp(-linspace(0, 10, reservoir.STIME))
    
    # the reservoir is randomly connected
    inp_to_reservoir_w = rand(reservoir.N) * (rand(reservoir.N) <.1)

    # integrate
    for t in xrange(reservoir.STIME) :
        reservoir.step(inp_to_reservoir_w*inp[t])
        reservoir.store(t)
    

    X = reservoir.data[reservoir.out_lab].T
    M = reservoir.w
    N = reservoir.N

    #plot outputs
    figure("OutputsVsInput")
    plot(X, linewidth=2)
    plot(inp*0.1, linewidth=15, color="red")
    plot(inp*0.1, linewidth=4, color="white")
    
    # plot the spectrogram of the M matrix 
    figure("spectrogram")
    EM, _ = eig( M )     
    radius = max(abs(EM))
    scatter(real(EM), imag(EM))
    xlim([-(radius*6./4.), (radius*6./4.)])
    ylim([-(radius*6./4.), (radius*6./4.)])
    
    #PCA
    B = X - X.mean(0)   # subtract mean
    C = dot(B.T, B)/float(N-1)    # covariance matrix
    E, W = eig( dot(C.T, C) )     # eigenvalues, eigenvectors
    E = real(E)    # throw off 0j imag part
    W = real(W[argsort(E)[::-1]])    # sort eigenvector 
                                     #      (throw off 0j imag part)
    W = W[:, :3]    # get first 3 eigenvectors
    T = dot(B, W)   # first 3 principal components
    
    # plot the trajectory made by the first 3 principal components    
    from mpl_toolkits.mplot3d import Axes3D    
    fig = figure("PCA")
    colors = cm.coolwarm(np.linspace(0, 1, reservoir.STIME))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(
            T[:, 0],
            T[:, 1],
            T[:, 2],
            linewidth = 0.5,
            color = "grey"
            )
    sc = ax.scatter(
            T[:, 0],
            T[:, 1],
            T[:, 2],
            s = 40,
            facecolor = colors,
            edgecolor = colors
            )
    ax.scatter(
            T[0, 0],
            T[0, 1],
            T[0, 2],
            s = 300,
            facecolor = colors[0,],
            edgecolor = colors[0,]
            )
    ax.scatter(
            T[-1::, 0],
            T[-1::, 1], 
            T[-1::, 2],
            s = 300,
            facecolor = colors[-1,],
            edgecolor = colors[-1,]
            )

    show()
