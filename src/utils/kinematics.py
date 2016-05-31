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
from __future__ import division

from math import *
from pylab import *

from scipy.ndimage import gaussian_filter1d
import scipy.optimize  
import time 

#----------------------------------------------------------------------
#----------------------------------------------------------------------
class Arm:
    """
    
    Kinematics of a number_of_joint-degrees-of-freedom 
    2-dimensional arm.
    
    Given the increment of joint angles calculate
    the current positions of the edges of the
    arm segments.

    """

    def __init__(self,
            number_of_joint = 3,
            joint_lims = None, 
            segment_lengths = None,
            origin = [0,0],
            mirror = False
            ):
        """
        number_of_joint:     (int)    number of joints
        joint_angles       (list):    initial joint angles
        joint_lims         (list):    joint angles limits 
        segment_lengths    (list):    length of arm segmens
        origin             (list):    origin coords of arm
        """

        self.mirror = mirror
        self.number_of_joint = number_of_joint
        
        # initialize lengths
        if segment_lengths is None:
            segment_lengths = ones(number_of_joint)
        self.segment_lengths = array(segment_lengths)
       
        # initialize limits   
        if joint_lims is None:
            joint_lims = vstack([-ones(number_of_joint)*
                pi,ones(number_of_joint)*pi]).T
        self.joint_lims = array(joint_lims)
       
        # set origin coords   
        self.origin = array(origin)


          
 
    def get_joint_positions(self,  joint_angles  ):
        """    
        Finds the (x, y) coordinates
        of each joint   
        joint_angles   (vector):    current angles of the joints    
        return          (array):    'number of joint' [x,y] coordinates   
        """ 

        #current angles
        res_joint_angles = joint_angles.copy() 

        # detect limits
        maskminus= res_joint_angles > self.joint_lims[:,0]
        maskplus = res_joint_angles < self.joint_lims[:,1]
  
        res_joint_angles =  res_joint_angles*(maskplus*maskminus) 
        res_joint_angles += self.joint_lims[:,0]*(logical_not(maskminus) )
        res_joint_angles += self.joint_lims[:,1]*(logical_not(maskplus) )
 
        if self.mirror :
            res_joint_angles = -res_joint_angles
            res_joint_angles[0] += pi 


        
        # calculate x coords of arm edges.
        # the x-axis position of each edge is the 
        # sum of its projection on the x-axis
        # and all the projections of the 
        # previous edges 
        x = array([  
                    sum([ 
                            self.segment_lengths[j] *
                            cos( (res_joint_angles[:(k+1)]).sum() ) 
                            for k in range(j+1) 
                        ])
                        for j in range(self.number_of_joint) 
                  ])
        
        # trabslate to the x origin 
        x = hstack([self.origin[0], x+self.origin[0]])

        # calculate y coords of arm edges.
        # the y-axis position of each edge is the 
        # sum of its projection on the x-axis
        # and all the projections of the 
        # previous edges 
        y = array([  
            sum([ 
                    self.segment_lengths[j] *
                    sin( (res_joint_angles[:(k+1)]).sum() ) 
                    for k in range(j+1) 
                ])
                for j in range(self.number_of_joint) 
            ])
        
        # translate to the y origin 
        y = hstack([self.origin[1], y+self.origin[1]])

        pos = array([x, y]).T

        return (pos, res_joint_angles)
    
    
#----------------------------------------------------------------------
#----------------------------------------------------------------------
def get_angle(v1,v2) :
    """
    Calculate the angle between two vectors
    v1  (array):   first vector
    v2  (array):   second vector
    """

    if (norm(v1)*norm(v2)) != 0 :     
        cosangle = dot(v1,v2)/(norm(v1)*norm(v2))
        cosangle = maximum(-1,minimum(1, cosangle))
        angle = arccos(cosangle)   
        if cross(v1,v2) < 0 :
            angle = 2*pi - angle  
        return angle
    return None


# Manages distances on a polynomial chain
class Polychain :

    # floating point precision
    HVERSOR = array([1,0])

    def set_chain(self, chain) :
        '''
         chain      (array):    each row is a point (x,y) 
                                of the polygonal chain
        '''

        # ensure it is a numpy array
        self.chain = array(chain[:])
        # the length of the chain (number of vertices)
        self.ln = len(self.chain)
        # the chain must be greater than one point
        if self.ln < 2 :
            raise ValueError('Polychain initialized with only one point. Minimum required is two.')
       
        # calculate segments lengths
        self.seg_lens = [ 
                norm( self.chain[x] -  self.chain[x-1] ) \
                for x in xrange(1, self.ln) ]

        # calculate angles at the vertices
        self.seg_angles = []
        for x in xrange(1, self.ln ) :
            if x == 1 :
                ab = self.HVERSOR
            else :
                ab = self.chain[x-1] - self.chain[x-2]
            bc = self.chain[x] - self.chain[x-1]
            self.seg_angles.append(get_angle(ab, bc))

    def isPointInChain(self, point, epsilon = 0.1 ) :
        '''
            find out if a point belongs to the chain. 
            return:     a list of distances correspoding to the the line 
                        intersection with that point. Empty list if
                        the point does not belong to the chain
        '''

        distances = []
        for x in xrange(1,len(self.chain) ) :
            
            a = self.chain[x-1]
            b = self.chain[x]
            c = array(point)
            
            # check if the  point is within the same line
            if norm(cross(b-a, c-a)) < epsilon :
                
                abac = dot(b-a, c-a)
                ab = dot(b-a, b-a)
                if 0 <= abac <= ab :

                    distance = sum(self.seg_lens[:(x-1)])
                    distance += norm(point - self.chain[x-1])
                    distance = distance/sum(self.seg_lens)

                    distances.append( distance )

        return distances
      
    def get_point(self, distance) :
        '''
        get a point in the 2D space given a 
        distance from the first point of the chain 
        '''

        if distance > 1 :
            raise ValueError('distance must be a proportion of the polyline length (0,1)')

        distance = sum(self.seg_lens)*distance
        cum_ln = 0
            
        for l in xrange(self.ln-1) :
            s_ln = self.seg_lens[l]
            if cum_ln <= distance <= cum_ln+s_ln :
                break 
            cum_ln += self.seg_lens[l]

        rel_ln = distance - cum_ln
        
        return self.chain[l] + \
                ( rel_ln*cos( sum(self.seg_angles[:(l+1)]) ), \
                    rel_ln*sin( sum(self.seg_angles[:(l+1)]) ) )

        return -1

    def get_dense_chain(self, density) :

        tot_len = self.get_length()
        curr_len = 0
        dense_chain = []
        points = density
        dense_chain.append(self.get_point( 0 ))
        for x in xrange(1,len(self.chain) ) :
            a = self.chain[x-1]
            b = self.chain[x]
            len_seg = norm(a-b)
            gap = (1.0/float(points+1))*len_seg
            for d in xrange(points+1):
                dist = (curr_len +gap*(1+d))/ float(tot_len)
                dense_chain.append(self.get_point( dist ))
            curr_len += len_seg

        return vstack(dense_chain)

    def get_length(self) :
        '''
        return: the length of the current polyline
        '''
        return sum(self.seg_lens)


if __name__ == "__main__" :
    
    ion()
  
    arm = Arm(
            number_of_joint = 3,    # 3 joints 
            origin = [0.0,0.0], # origin at (1.0 , 0.5)
            joint_lims = [ 
                [0, pi*(3./2.)],    # first joint limits                   
                [0, pi*0.75],    # second joint limits             
                [0, pi*0.75]     # third joint limits
                ]  
            )
    
    angle = zeros(3)  
    pos,angle = arm.get_joint_positions(angle)
    poly = Polychain()
    poly.set_chain(pos)
    point = poly.get_point(0.75)
 
    
    # init plot 
    fig = figure("arm")
    ax = fig.add_subplot(111)
    segments, = ax.plot(*pos.T)    # plot arm segments
    edges = ax.scatter(*pos.T) # plot arm edges
    xl = [-4,4]    # x-axis limits
    yl = [-4,4]    #y-axis limits
    ax.plot(xl, [0,0], c = "black", linestyle = "--")    # plot x-axis
    ax.plot([0,0], yl, c = "black", linestyle = "--")    # plot y-axis     
    external_point = scatter(*point, s= 30, c="r") # plot arm edges
    dense_points = scatter(*zeros([2,25]), s= 20, c=[1,1,0]) # plot arm edges
    xlim(xl)
    ylim(yl) 

    # iterate over 100 timesteps
    for t in range(10):
      
        # set a random gaussian increment for each joint
        angle = ones(3)*(t*((2*pi)/100.0)) 
        
        # calculate current position given the increment
        pos, angle = arm.get_joint_positions(angle)
        poly.set_chain(pos)
      
        point = poly.get_point(0.75)

        dense = poly.get_dense_chain(2)
        print dense.shape
        # update plot
        segments.set_data(*pos.T)
        edges.set_offsets(pos)
        external_point.set_offsets(point)
        dense_points.set_offsets(dense)
        fig.canvas.draw()
        pause(.00001)

    raw_input()


