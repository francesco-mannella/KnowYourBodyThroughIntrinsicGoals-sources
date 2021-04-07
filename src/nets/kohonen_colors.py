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
from kohonen import Kohonen


def normalize(x) :
    return x/np.linalg.norm(x)

def run_colors() :     
    
   
    import matplotlib.pyplot as plt
    plt.ion()
    plt.close('all')

    k = Kohonen(
            n_output = 400, 
            n_dim_out = 2,
            bins = 20,
            n_input = 3, 
            eta = .5,
            eta_bl = 0.1,
            stime = 20000,
            neighborhood = 1000, 
            neighborhood_decay = 1000,
            eta_decay = 4000,
            normalize = normalize
            )

    #--------------------------------------------------------
    # Prepare input and network vars

    samples = np.floor(255*rnd.rand(100,3))
    N_SAMPLES = samples.shape[0]

    #--------------------------------------------------------
    # Prepare plot

    fig = plt.figure("kohonen", figsize = (12,4))
    ax1 = fig.add_subplot(141)
    ax2 = fig.add_subplot(142)
    ax3 = fig.add_subplot(143)
    ax4 = fig.add_subplot(144)

    n_row = int(np.sqrt(k.N_OUTPUT))


    image1 = ax1.imshow( np.zeros([n_row, n_row, 3] ),
            vmin = 0, vmax = 1, interpolation = 'none', aspect = 'equal' )
    image2 = ax2.imshow( np.zeros([n_row, n_row]),
            interpolation = 'none', aspect = 'equal',
            vmin = 0, vmax = 1, cmap = plt.cm.binary )
    image3 = ax3.imshow( np.zeros([1,1,3]),
            vmin = 0, vmax = 1, interpolation = 'none', aspect = 'equal' )


    line, = ax4.plot(np.zeros(k.STIME))
    ax4.set_ylim([0.,1.])

    #--------------------------------------------------------
    # spreading

    for t in range(k.STIME) :

        # shuffle samples
        if t%N_SAMPLES == 0 :
            rnd.shuffle(samples)

        # network spreading
        inp = samples[t%N_SAMPLES,:]/255.0
        k.step(inp)
        k.learn()
        k.store()

        k

        # plot
        if t%200 == 0 :

            w = k.inp2out_w.reshape(
                n_row,
                n_row,
                k.N_INPUT
            )

            out = k.out.reshape( n_row, n_row)

            inp = inp.reshape(1,1,3)

            image1.set_data(w)
            image2.set_data(out)
            image3.set_data(inp)
            line.set_ydata(k.data[k.l_out_raw][k.idx,:])

            fig.canvas.draw()
            plt.pause(0.001)

    return k

if __name__ == "__main__" : 

   run_colors()
   raw_input()
