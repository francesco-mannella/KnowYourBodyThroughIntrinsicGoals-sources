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


class MultidimensionalGaussianMaker:
    
    def __init__(self, lims) :

        lims = np.vstack(lims) 
        idcs = grid(lims[:,2])
        
        self.X = scaled_grid(idcs, lims[:,:2])
        self.ln = self.X.shape[0]
        self.nDim = self.X.shape[1] 

    def get_gaussian(self, mu, sigma) :
         
        e = (self.X - mu).T
        S = np.eye(self.nDim, self.nDim)*(1.0/sigma) 
        y = np.exp( -np.diag(np.dot(e.T, np.dot(S,e) ) ))

        return (y, self.X)

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