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

import numpy as np
import numpy.random as rnd

from scipy.ndimage import gaussian_filter1d
import scipy.optimize  
import time 


#----------------------------------------------------------------------
#----------------------------------------------------------------------
#----------------------------------------------------------------------
#----------------------------------------------------------------------
# UTILS ---------------------------------------------------------------
#----------------------------------------------------------------------
#----------------------------------------------------------------------

def get_angle(v1,v2) :
    """
    Calculate the angle between two vectors
    v1  (array):   first vector
    v2  (array):   second vector
    """

    if (np.linalg.norm(v1)*np.linalg.norm(v2)) != 0 :     
        cosangle = np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))
        cosangle = np.maximum(-1,np.minimum(1, cosangle))
        angle = np.arccos(cosangle)   
        if np.cross(v1,v2) < 0 :
            angle = 2*np.pi - angle  
        return angle
    return None


# Manages distances on a polynomial chain
class Polychain(object) :

    # floating point precision
    HVERSOR = np.array([1,0])

    def set_chain(self, chain) :
        '''
         chain      (array):    each row is a point (x,y) 
                                of the polygonal chain
        '''

        # ensure it is a numpy array
        self.chain = np.array(chain[:])
        # the length of the chain (number of vertices)
        self.ln = len(self.chain)
        # the chain must be greater than one point
        if self.ln < 2 :
            raise ValueError('Polychain initialized with only one point. Minimum required is two.')
       
        # calculate segments lengths
        self.seg_lens = [ 
                np.linalg.norm( self.chain[x] -  self.chain[x-1] ) \
                for x in range(1, self.ln) ]

        # calculate angles at the vertices
        self.seg_angles = []
        for x in range(1, self.ln ) :
            if x == 1 :
                ab = self.HVERSOR
            else :
                ab = self.chain[x-1] - self.chain[x-2]
            bc = self.chain[x] - self.chain[x-1]
            self.seg_angles.append(get_angle(ab, bc))

    def autocollision(self, epsilon = 0.1, is_set_collinear=False):
        self.intersect = None
        (start, end) = self.chain[[1,-1]]

        for x in range(1,len(self.chain) ) :
            
            p = self.chain[x-1]
            rp = self.chain[x]
            r = rp - p
            
            for y in range(1,len(self.chain) ) :
                q = self.chain[y-1]
                sq = self.chain[y] 
                s = sq - q                 
                
                not_junction = np.all(p != sq) and np.all(q != rp) 

                if x!=y and not_junction :

                    rxs = np.linalg.norm(np.cross(r,s))
                    qpxr = np.linalg.norm(np.cross(q-p, r))
                    qpxs = np.linalg.norm(np.cross(q-p, s))
 
                    rxs_zero = abs(rxs) < epsilon

                    if is_set_collinear:
                        test_collinear = ( rxs_zero and qpxr < epsilon )
                        if test_collinear: 
                            t0 = np.dot(q-p,r)/np.dot(r,r)
                            t1 = t0 + np.dot(s,r)/np.dot(r,r)
                            mint = min(t0, t1)
                            maxt = max(t0, t1)
                            if (mint > (0+epsilon) and mint < (1+epsilon)) \
                                    or (maxt >  (0+epsilon) and maxt < (1-epsilon)) \
                                    or (mint <= (0+epsilon) and maxt >= (1-epsilon)):
                                return True

                    if not rxs_zero :
                        t = qpxs / rxs  
                        u = qpxr / rxs   

                        test_intersect = ((0)<t<(1)) and ((0)<u<(1)) 
                        if test_intersect:
                            self.intersect = p +t*r
                            return True


        return False
                
    def isPointInChain(self, point, epsilon = 0.1 ) :
        '''
            find out if a point belongs to the chain. 
            return:     a list of distances correspoding to the the line 
                        intersection with that point. Empty list if
                        the point does not belong to the chain
        '''

        distances = []
        c = array(point)
        for x in range(1,len(self.chain) ) :
            
            a = self.chain[x-1]
            b = self.chain[x]
            
            # check if the  point is within the same line
            if np.all(c!=a) and np.all(c!=b) :
                if np.linalg.norm(np.cross(b-a, c-a)) < epsilon :
                    
                    abac = np.dot(b-a, c-a)
                    ab = np.dot(b-a, b-a)
                    if 0 <= abac <= ab :

                        distance = np.sum(self.seg_lens[:(x-1)])
                        distance += np.linalg.norm(point - self.chain[x-1])
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
            
        for l in range(self.ln-1) :
            s_ln = self.seg_lens[l]
            if cum_ln <= distance <= cum_ln+s_ln :
                break 
            cum_ln += self.seg_lens[l]

        rel_ln = distance - cum_ln
        
        return self.chain[l] + \
                ( rel_ln*np.cos( sum(self.seg_angles[:(l+1)]) ), \
                    rel_ln*np.sin( sum(self.seg_angles[:(l+1)]) ) )

        return -1

    def get_dense_chain(self, density) :

        tot_len = self.get_length()
        curr_len = 0
        dense_chain = []
        points = density
        dense_chain.append(self.get_point( 0 ))
        for x in range( density ) :
            dense_chain.append(self.get_point( (1+x)/float(density+1) ))

        dense_chain.append(self.get_point( 1 ))

        return np.vstack(dense_chain)

    def get_length(self) :
        '''
        return: the length of the current polyline
        '''
        return sum(self.seg_lens)





#----------------------------------------------------------------------
#----------------------------------------------------------------------
#----------------------------------------------------------------------
#----------------------------------------------------------------------
# ARM -----------------------------------------------------------------
#----------------------------------------------------------------------
#----------------------------------------------------------------------


class PID(object) :

    def __init__(self, n=1, dt=0.1, Kp=0.1, Ki=0.9, Kd=0.001 ):
       
        self.n = n
        self.dt = dt

        self.previous_error = np.zeros(n)
        self.integral = np.zeros(n)
        self.derivative = np.zeros(n)
        self.setpoint = np.zeros(n)
        self.output = np.zeros(n)
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd

    def reset(self):
        n = self.n
        self.previous_error = np.zeros(n)
        self.integral = np.zeros(n)
        self.derivative = np.zeros(n)
        self.setpoint = np.zeros(n)
        self.output = np.zeros(n)

    def step(self, measured_value, setpoint=None):
        
        if setpoint is not None:
            self.setpoint = np.array(setpoint)

        error = setpoint - measured_value
        self.integral =  self.integral + error*self.dt
        self.derivative = (error - self.previous_error)/self.dt
        self.output = self.Kp*error + \
                self.Ki*self.integral + \
                self.Kd*self.derivative
        
        self.previous_error = error

        return self.output

class Arm(object):
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
            segment_lengths = np.ones(number_of_joint)
        self.segment_lengths = np.array(segment_lengths)
       
        # initialize limits   
        if joint_lims is None:
            joint_lims = vstack([-np.ones(number_of_joint)*
                pi,ones(number_of_joint)*pi]).T
        self.joint_lims = np.array(joint_lims)
       
        # set origin coords   
        self.origin = np.array(origin)
        
    def get_joint_positions(self,  joint_angles  ):
        """    
        Finds the (x, y) coordinates
        of each joint   
        joint_angles   (vector):    current angles of the joints    
        return          (array):    'number of joint' [x,y] coordinates   
        """ 


        # current angles
        res_joint_angles = joint_angles.copy() 

        # detect limits
        maskminus= res_joint_angles > self.joint_lims[:,0]
        maskplus = res_joint_angles < self.joint_lims[:,1]
  
        res_joint_angles =  res_joint_angles*(maskplus*maskminus) 
        res_joint_angles += self.joint_lims[:,0]*(np.logical_not(maskminus) )
        res_joint_angles += self.joint_lims[:,1]*(np.logical_not(maskplus) )
 
        # mirror
        if self.mirror :
            res_joint_angles = -res_joint_angles
            res_joint_angles[0] += np.pi 
 
        # calculate x coords of arm edges.
        # the x-axis position of each edge is the 
        # sum of its projection on the x-axis
        # and all the projections of the 
        # previous edges 
        x = np.array([  
                    sum([ 
                            self.segment_lengths[k] *
                            np.cos( (res_joint_angles[:(k+1)]).sum() ) 
                            for k in range(j+1) 
                        ])
                        for j in range(self.number_of_joint) 
                  ])
        
        # trabslate to the x origin 
        x = np.hstack([self.origin[0], x+self.origin[0]])

        # calculate y coords of arm edges.
        # the y-axis position of each edge is the 
        # sum of its projection on the x-axis
        # and all the projections of the 
        # previous edges 
        y = np.array([  
            sum([ 
                    self.segment_lengths[k] *
                    np.sin( (res_joint_angles[:(k+1)]).sum() ) 
                    for k in range(j+1) 
                ])
                for j in range(self.number_of_joint) 
            ])
        
        # translate to the y origin 
        y = np.hstack([self.origin[1], y+self.origin[1]])

        pos = np.array([x, y]).T
 
        return (pos, res_joint_angles)
    
#----------------------------------------------------------------------
#----------------------------------------------------------------------
#----------------------------------------------------------------------
#----------------------------------------------------------------------
# TEST ----------------------------------------------------------------
#----------------------------------------------------------------------
#----------------------------------------------------------------------

    
if __name__ == "__main__" :
    
    import matplotlib.pyplot as plt
    plt.ion()
  
    arm = Arm(
            number_of_joint = 6,    # 3 joints 
            origin = [0.0,0.0], # origin at (1.0 , 0.5)
            segment_lengths = array([1,1,3,1,1,2])/2.,
            joint_lims = [ 
                [0, pi],    # first joint limits                   
                [0, pi],    # second joint limits             
                [0, pi],     # third joint limits
                [0, pi],     # third joint limits
                [0, pi],     # third joint limits
                [0, pi]     # third joint limits
                ],  
            mirror=True
            )
    
    angle = zeros(6)  
    pos,angle = arm.get_joint_positions(angle)
    poly = Polychain()
    poly.set_chain(pos)
    point = poly.get_point(0.75)
 
    
    # init plot 
    fig = plt.figure("arm")
    ax = fig.add_subplot(111,aspect="equal")
    segments, = ax.plot(*pos.T)    # plot arm segments
    edges = ax.scatter(*pos.T) # plot arm edges
    xl = [-8,8]    # x-axis limits
    yl = [-8,8]    #y-axis limits
    ax.plot(xl, [0,0], c = "black", linestyle = "--")    # plot x-axis
    ax.plot([0,0], yl, c = "black", linestyle = "--")    # plot y-axis     
    external_point = scatter(*point, s= 30, c="r") # plot arm edges
    dense_points = scatter(*zeros([2,25]), s= 20, c=[1,1,0]) # plot arm edges
    plt.xlim(xl)
    plt.ylim(yl) 

    # iterate over 100 timesteps
    for t in range(200):
      
        # set a random gaussian increment for each joint
        angle = ones(6)*(t*((2*pi)/400.0)) 
        angle[5] = 0 
        # calculate current position given the increment
        pos, angle = arm.get_joint_positions(angle)
        poly.set_chain(pos)
        
        collision = poly.autocollision(is_set_collinear=True)
        point = poly.get_point(0.75)

        dense = poly.get_dense_chain(6)

        # update plot
        segments.set_data(*pos.T)
        if collision:
            segments.set_color('red')
            segments.set_linewidth(2)
        else:
            segments.set_color('blue')
            segments.set_linewidth(1)
        edges.set_offsets(pos)
        #external_point.set_offsets(point)
        #dense_points.set_offsets(dense)
        fig.canvas.draw()
        raw_input()

    raw_input()


