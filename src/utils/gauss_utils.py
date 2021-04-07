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
import numpy.random as rnd
import time
import os

#------------------------------------------------------------
def map1DND(x,nDim,nBin) :
    '''
    Map an index in one dimension with a tuple of indices in
    'nDim' dinemsions.

    x           (int): Original 1-dimensional index
    nDim        (int): Number of dimensions to map to
    nBin    list(int): length of the dimensions to map to
  '''

    idx_max = np.prod(nBin) -1

    if x > idx_max :
        raise(ValueError( "index is greater than Bin**nDim" ))

    idcs = np.zeros(nDim)
    for i in xrange(nDim) :
        idcs[nDim -1 -i] = x%nBin[nDim -1 -i]
        x /= nBin[nDim -1 -i]
    
    return idcs.astype("int")

def mapND1D(idcs,nBin) :
    '''
    Map a tuple of indices in 'nDim' dimensions with an index in
    one dinemsion.

    x           (int): Original 1-dimensional index
    nBin    list(int): length of the original dimensions
    '''
    nDim = len(idcs)

    return int(sum( [  idcs[nDim -1 -x]*(nBin[x]**x)
        for x in xrange(nDim) ] ))

def grid(bins) :
    
    ret = np.array([])

    factor = np.arange(bins[0])

    if len(bins) > 1 :

        block = grid(bins[1:])
        n = len(block)
 
        blocks = []
        for factor_value in factor :
       
            factor_index = factor_value*np.ones([ n ,1])
            blocks.append(np.hstack( [ factor_index, block ] ))       
       
        ret = np.vstack(blocks)

    else :
        ret = factor.reshape(int(bins[0]), 1)
    
    return ret

def scaled_grid(idcs, lims) :
    
    idcs = idcs.astype("int")
    scaled = np.zeros(idcs.shape)
    
    for i in xrange(len(lims)):
       
        x = np.linspace(*lims[i], num = (max(idcs[:,i])+1))
        scaled[:,i] += x[idcs[:,i]]
    
    return scaled 


class MultidimensionalGaussianMaker(object) :
    
    def __init__(self, lims) :

        lims = np.vstack(lims) 
        idcs = grid(lims[:,2])
        
        self.X = scaled_grid(idcs, lims[:,:2])
        self.ln = self.X.shape[0]
        self.nDim = self.X.shape[1] 

    def __call__(self, mu, sigma) :
         
        e = (self.X - mu).T
        S = np.eye(self.nDim, self.nDim)*(1.0/sigma) 
        y = np.exp( -np.diag(np.dot(e.T, np.dot(S,e) ) ))

        return (y, self.X)
    


class TwoDimensionalGaussianMaker(object) :

    def __init__(self, lims) :

        lims = np.vstack(lims) 
        x = np.linspace(*lims[0])
        y = np.linspace(*lims[1])
        self.X, self.Y = np.meshgrid(x,y)

    def __call__(self, mu, sigma, theta=0) :

        if np.isscalar(sigma) == 1 :
            sigma = [sigma,sigma]

        sx,sy = sigma
        mx,my = mu
        
        a = (np.cos(theta)**2)/(2*sx**2) +\
            (np.sin(theta)**2)/(2*sy**2);
        b = (-np.sin(2*theta))/(4*sx**2) +\
            (np.sin(2*theta))/(4*sy**2);
        c = (np.sin(theta)**2)/(2*sx**2) +\
            (np.cos(theta)**2)/(2*sy**2);
        
        res = np.exp( 
                -a*(self.X-mx)**2 
                -2*b*(self.X-mx)*(self.Y-my) 
            -c*(self.Y-my)**2)
        
        return res.T.ravel(), [self.X, self.Y]

class OneDimensionalGaussianMaker(object) :

    def __init__(self, lims) :
        self.x = np.linspace(*lims[0])

    def __call__(self, mu, sigma) :

        return np.exp((-(self.x-mu)**2)/(sigma**2)), self.x

class OptimizedGaussianMaker(object) :
   
    def __init__(self, lims) :
        L = len(lims)
        self.gm = None
        if L == 1 :
            self.gm = OneDimensionalGaussianMaker(lims)
        elif L == 2:
            self.gm = TwoDimensionalGaussianMaker(lims)
        else:
            self.gm = MultidimensionalGaussianMaker(lims)

    def __call__(self, mu, sigma) :

        return self.gm(mu, sigma)



def gauss2d_oriented(x,y,m1,m2,std_x, std_y, theta) :
   
    a = (np.cos(theta)**2)/(2*std_x**2) +\
        (np.sin(theta)**2)/(2*std_y**2);
    b = (-np.sin(2*theta))/(4*std_x**2) +\
        (np.sin(2*theta))/(4*std_y**2);
    c = (np.sin(theta)**2)/(2*std_x**2) +\
        (np.cos(theta)**2)/(2*std_y**2);
    
    return np.exp( -a*(x-m1)**2 -2*b*(x-m1)*(y-m2) -c*(y-m2)**2) 


if __name__ == "__main__" :
    from pylab import *

    m = arange(15).reshape(3,5)

    print m
    print

    for x in xrange(3):
        line = ""
        for y in xrange(5):
            line=line+ "{} ".format(mapND1D([x,y],[3,5]))
        print line

    print

    line = ""
    for x in xrange(15):
        if x%5 == 0 :
            print line
            line =""
        line=line+ "{} ".format(map1DND(x,2,[3,5]))
    print line
