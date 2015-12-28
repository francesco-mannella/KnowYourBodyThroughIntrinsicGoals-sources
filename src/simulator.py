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
import time 

import kinematics as KM
import reservoir as RS


def gauss_release(x) :

    s = 0.48
    p = 20.0
    return exp(-(x/(2*s))**100)


#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------

class Learner :
    """    
    A leaky-integrator dynamic reservoir is used as the encoder. 
    
    It gives an output vector of N_READOUTS elements

    It takes as external input a vector of NUMBER_OF_TARGETS goal positions. 
    This vector is preprocessed through gaussian filters for smoothing.  

    It also takes as feedback input its output, a vector of N_READOUTS elements 

    LEARNING

    """

    #-------------------------------------------------------------------------
    #-------------------------------------------------------------------------
    #-------------------------------------------------------------------------    
    def __init__(self, stime = 100) :

        # standard deviation of noise ----------------------------------------
        self.EPSILON                           = 0.3       
        # decaying time for epsilon ------------------------------------------
        self.EPSILON_DECAY                     = 20.0         
        # learning rate ------------------------------------------------------
        self.ETA                               = 0.4         
        # learning regularization --------------------------------------------
        self.RGLR                              = 0.1
        # number of goal points ----------------------------------------------
        self.NUMBER_OF_TARGETS                 = 30    
        # decay rate of target memories 
        self.MEMORY_DECAY                      = 0.4

        # number of reservoir units ------------------------------------------
        self.N_RESERVOIR                       = 200  
        # number of readout units --------------------------------------------            
        self.N_READOUTS                        = 6       
        # proportion of input vs feedback ------------------------------------            
        self.INP_FEED_PROP                     = 2./3.
        # variance of the input gaussians ------------------------------------            
        self.SIGMA                             = 0.4

        # sparseness of input connections ------------------------------------            
        self.SPARSENESS                        = 0.005 
        # sparseness of input connections ------------------------------------            
        self.STIME                             = stime

        # initialize the value of the noise standard deviation
        self.curr_epsilon = 0.0

        # RESERVOIR

        # init
        self.encoder = RS.Reservoir(N=self.N_RESERVOIR, stime = self.STIME)          

        # readout indices
        self.READOUT = range(self.N_RESERVOIR)[(-self.N_READOUTS):] 
        self.NOREADOUT = range(self.N_RESERVOIR)[:(-self.N_READOUTS)] 

        # create weights from sensory inputs to the reservoir
        ninp = int(self.N_RESERVOIR*self.INP_FEED_PROP) 
        self.inp_w = zeros([ninp, self.NUMBER_OF_TARGETS])
        x = linspace(0.0,1.0, ninp)
        m = linspace(0.0,1.0,self.NUMBER_OF_TARGETS)
        sigma = self.SIGMA
        X,M = meshgrid(x,m)
        self.inp_w = exp(- ((1.0/sigma**2)*(X-M)**2)).T      

        # create weights from proprioceptive inputs to the reservoir   
        finp = self.N_RESERVOIR - ninp
        self.feed_w = randn(finp, self.N_READOUTS)
        self.feed_w *= (rand(*self.feed_w.shape) < self.SPARSENESS)

        # initialize weights to the readout units to zero 
        # so that the contribution of weight update is clear  
        self.encoder.w[self.READOUT,:] = 0
        
        # initialize feedbacks form the readouts to zero.
        self.encoder.w[:,self.READOUT] = 0

        # initialize noise
        self.noise = 0 
        
        # init readout units
        self.readouts = zeros(self.N_READOUTS)        
        
        # init effective readout units
        self.noised = zeros(self.N_READOUTS) 
        
        # init target readout positions
        self.target_readouts = zeros([self.NUMBER_OF_TARGETS, self.N_READOUTS]) 
        
        # init target readout positions
        self.current_target_readouts = zeros( self.N_READOUTS) 
        
        # init target counter 
        self.target_occurrence_counter = zeros(self.NUMBER_OF_TARGETS) 
    
    #-------------------------------------------------------------------------
    #-------------------------------------------------------------------------
    #-------------------------------------------------------------------------    
    def reset(self) :
        '''
        Reset all variables
        '''

        self.encoder.reset()
        self.noise = 0
        self.readouts = zeros(self.N_READOUTS)        
        self.noised = zeros(self.N_READOUTS) 

    #-------------------------------------------------------------------------
    #-------------------------------------------------------------------------
    #-------------------------------------------------------------------------    
    def update_noise(self) :
        '''
        Update the current value of noise amplitude
        '''
    
        self.curr_epsilon = self.curr_epsilon * \
                exp(-1.0/float(self.EPSILON_DECAY))

    def reset_noise(self, amplitude = 1.0) :  
        '''
        Reset noise amplitude to an initial value.
        amplitude (float) ratio of the default initial value EPSILON
        '''       

        self.curr_epsilon = self.EPSILON*amplitude 

    #-------------------------------------------------------------------------
    #-------------------------------------------------------------------------
    #-------------------------------------------------------------------------
    def activation_step(self, t, inp) :
        '''
        A timestep calculation of the learner activity
        t    (int)     timestep
        inp  (vector)  input-state vacter
        '''

        # PREPARE ACTION
        self.inp_current = hstack(( 
            dot(self.inp_w, inp), 
            dot(self.feed_w, self.readouts) ))

        # iterate
        self.encoder.step(self.inp_current)   

        # calculate readouts  
        self.readouts = self.encoder.out[self.READOUT]

        # calculate noise   
        self.noise = self.curr_epsilon*randn(self.N_READOUTS)  
        
        # calculate actual commands 
        self.noised = self.readouts + self.noise
    
    #-------------------------------------------------------------------------
    #-------------------------------------------------------------------------
    #-------------------------------------------------------------------------
    def update_target(self, state) :
        '''
        Update the readout target for a given status 
        state    (vector)         the input-state
                                  for which the target has to 
                                  be updated
        '''

        # in case reward is part of the targets, update the target readouts       
        if sum(state)>0  : 
            
            idx = find(state == max(state))[0]
            # update the mean of the target position for a given test_sample 
            t_readouts = self.target_readouts[idx,:]
            ts = self.target_occurrence_counter[idx]   
            t_readouts += (1/(ts+1))*(-t_readouts + self.noised)
            self.current_target_readouts = t_readouts[:]

            # updater counter of target appearences
            self.target_occurrence_counter += state

    #-------------------------------------------------------------------------
    #-------------------------------------------------------------------------
    #-------------------------------------------------------------------------
    def learn(self) :
            '''
            learning step
            '''

            inp = self.inp_current
            x = self.encoder.out
            r = self.current_target_readouts - self.readouts
            e = self.ETA
            w = self.encoder.w[self.READOUT,:]

            #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
            
            dw = e*( outer(r, x) -self.RGLR*w)

            #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
            
            self.encoder.w[self.READOUT,:] = dw

#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
class Predictor :
    '''
    Decide goal setting based on interest (inverse of the rate of ability 
    to produce the state)
    '''

    #-------------------------------------------------------------------------
    #-------------------------------------------------------------------------
    #-------------------------------------------------------------------------
    def __init__(self) :

        # number of goal points ----------------------------------------------
        self.NUMBER_OF_STATES                 = 30    
        # decay rate of target memories --------------------------------------
        self.MEMORY_DECAY                     = 0.4
        # learning rate ------------------------------------------------------
        self.ETA                              = 0.1
             
        # init target memory 
        self.memory = zeros(self.NUMBER_OF_STATES) 

    #-------------------------------------------------------------------------
    #-------------------------------------------------------------------------
    #-------------------------------------------------------------------------
    def update(self, prop_occur, state) :
        '''
        Calculate how much the agent is able to each a state 
        '''
        
        goal = prop_occur >= 0.1

        self.memory += self.ETA*(goal-prop_occur)*state
        self.memory = maximum(0,minimum(1,self.memory))
    
    #-------------------------------------------------------------------------
    #-------------------------------------------------------------------------
    #-------------------------------------------------------------------------
      
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------

class Controller :

    ACTUAL = 1
    COMMANDED = 2 
    DESIRED = 3

    #-------------------------------------------------------------------------
    #-------------------------------------------------------------------------
    #-------------------------------------------------------------------------
    def __init__(self) :
        
        # number of arm joints -----------------------------------------------            
        self.NUMBER_OF_JOINTS                  = 3               
        
        # number of arm joints -----------------------------------------------            
        self.NUMBER_OF_SENSORS                 = 30               
        
        # sensor overlap -----------------------------------------------------            
        self.SENSOR_OVERLAP                    = 0.1   
        # sensitivity of the arm sensors -------------------------------------               
        self.SENSOR_SENSITIVITY                = 0.2
        # sensitivity of the hand/arm intersection ---------------------------               
        self.TOUCH_SENSITIVITY                 = 0.3

        # origin of arm (first joint)-----------------------------------------
        self.LEFT_ARM_ORIGIN                   = [2, 0.0]      
        # lenghts of arm segments --------------------------------------------
        self.LEFT_ARM_SEGMENT_LENGTHS          = [1.0, 1.0, 1.0]
        # limits of joint angles ---------------------------------------------
        self.LEFT_ARM_LIMITS                   = array([ [0, pi*0.75], 
                                                         [0, pi*0.75], 
                                                         [0, pi*0.75] ])

        # origin of arm (first joint)-----------------------------------------
        self.RIGHT_ARM_ORIGIN                  = [-2, 0.0]      
        # lenghts of arm segments --------------------------------------------
        self.RIGHT_ARM_SEGMENT_LENGTHS         = [1.0, 1.0, 1.0]
        # limits of joint angles ---------------------------------------------
        self.RIGHT_ARM_LIMITS                  = array([ [0, pi*0.75], 
                                                         [0, pi*0.75], 
                                                         [0, pi*0.75] ])
        
        self.left_arm_actual_position         = zeros(( self.NUMBER_OF_JOINTS, 2 ))
        self.left_arm_commanded_position      = zeros(( self.NUMBER_OF_JOINTS, 2 ))
        self.left_arm_desired_position        = zeros(( self.NUMBER_OF_JOINTS, 2 ))

        self.right_arm_actual_position        = zeros(( self.NUMBER_OF_JOINTS, 2 ))
        self.right_arm_commanded_position     = zeros(( self.NUMBER_OF_JOINTS, 2 ))
        self.right_arm_desired_position       = zeros(( self.NUMBER_OF_JOINTS, 2 ))
        
        # One-dimentional encoding of the body
        self.body = KM.Polychain()   
        self.body_sensors = zeros( self.NUMBER_OF_SENSORS )    
        self.touch_point= (0,0)
        self.touch_distance = -1

        # this object computes the arm 
        # kinematics given the angles
        self.rarm = KM.Arm(
                number_of_joint = self.NUMBER_OF_JOINTS,
                origin = self.LEFT_ARM_ORIGIN[:], 
                segment_lengths = self.LEFT_ARM_SEGMENT_LENGTHS[:],
                joint_lims = self.LEFT_ARM_LIMITS[:]

                )  

        # this object computes the arm 
        # kinematics given the angles
        self.larm = KM.Arm(
                number_of_joint = self.NUMBER_OF_JOINTS,
                origin = self.RIGHT_ARM_ORIGIN[:], 
                segment_lengths = self.RIGHT_ARM_SEGMENT_LENGTHS[:],
                joint_lims = self.RIGHT_ARM_LIMITS[:],
                mirror=True
                )
        
        # set zero positions
        self.set_arms_positions( 
                zeros(self.NUMBER_OF_JOINTS*2), 
                zeros(self.NUMBER_OF_JOINTS*2), 
                zeros(self.NUMBER_OF_JOINTS*2) )

    #-------------------------------------------------------------------------
    #-------------------------------------------------------------------------
    #-------------------------------------------------------------------------
    def set_arms_positions(self, actual_angles, 
            commanded_angles, desired_angles ) :

        actual_angles = 2*pi*array(actual_angles[:])
        commanded_angles = 2*pi*array(commanded_angles[:])
        desired_angles = 2*pi*array(desired_angles[:])

        self.left_arm_actual_position,_ = \
                self.larm.get_joint_positions(
                        actual_angles[:self.NUMBER_OF_JOINTS])
        self.left_arm_commanded_position,_ = \
                self.larm.get_joint_positions(
                        commanded_angles[:self.NUMBER_OF_JOINTS])
        self.left_arm_desired_position,_  = \
                self.larm.get_joint_positions(
                        desired_angles[:self.NUMBER_OF_JOINTS])
       
        self.right_arm_actual_position,_ = \
                self.rarm.get_joint_positions(
                        actual_angles[self.NUMBER_OF_JOINTS:])
        self.right_arm_commanded_position,_ = \
                self.rarm.get_joint_positions(
                        commanded_angles[self.NUMBER_OF_JOINTS:])
        self.right_arm_desired_position,_  = \
                self.rarm.get_joint_positions(
                        desired_angles[self.NUMBER_OF_JOINTS:])

        # One-dimentional encoding of the body
        self.body.set_chain(vstack((
            self.left_arm_actual_position[::-1],
            self.right_arm_actual_position)))

    #-------------------------------------------------------------------------
    #-------------------------------------------------------------------------
    #-------------------------------------------------------------------------
    def get_arms_positions(self, pos_type = 1 ) :
        
        if pos_type == self.ACTUAL :
            return vstack((self.left_arm_actual_position[::-1], 
                    self.right_arm_actual_position))
        elif pos_type == self.COMMANDED :
            return vstack((self.left_arm_commanded_position[::-1], 
                    self.right_arm_commanded_position))
        elif pos_type == self.DESIRED :
            return vstack((self.left_arm_desired_position[::-1], 
                    self.right_arm_desired_position))

    #-------------------------------------------------------------------------
    #-------------------------------------------------------------------------
    #-------------------------------------------------------------------------
    def get_hands_positions(self, pos_type = 1 ) :

        if pos_type == self.ACTUAL :
            return (self.left_arm_actual_position[-1], 
                    self.right_arm_actual_position[-1])
        elif pos_type == self.COMMANDED :
            return (self.left_arm_commanded_position[-1], 
                    self.right_arm_commanded_position[-1])
        elif pos_type == self.DESIRED :
            return (self.left_arm_desired_position[-1], 
                    self.right_arm_desired_position[-1])
    
    
    #-------------------------------------------------------------------------
    #-------------------------------------------------------------------------
    #-------------------------------------------------------------------------
    def get_sensors(self) :

        hands = self.get_hands_positions(self.ACTUAL)
        for hand,h_idx in zip(hands, range(len(hands))):
            dists = array(self.body.isPointInChain(hand,self.SENSOR_SENSITIVITY))    
            if len(dists) > 1 :
                dist = dists[ logical_and( dists!=0.0, dists!=1.0) ]
                if len(dist) > 0 :
                    dist = dist[0]
                    if abs(h_idx-dist) > self.TOUCH_SENSITIVITY : 
                        x = linspace(0.,1.,self.NUMBER_OF_SENSORS )
                        self.body_sensors = exp(-(x - dist )**2/self.SENSOR_OVERLAP**2)  
                        self.touch_distance = dist
                        self.touch_point = self.body.get_point(self.touch_distance)
                        state = 1.0*(self.body_sensors == max(self.body_sensors))
                        return state
    
        return zeros(self.NUMBER_OF_SENSORS)
    
    #-------------------------------------------------------------------------
    #-------------------------------------------------------------------------
    #-------------------------------------------------------------------------
    def reset(self) :

        self.left_arm_actual_position     = zeros(( self.NUMBER_OF_JOINTS, 2 ))
        self.left_arm_commanded_position  = zeros(( self.NUMBER_OF_JOINTS, 2 ))
        self.left_arm_desired_position    = zeros(( self.NUMBER_OF_JOINTS, 2 ))
        self.right_arm_actual_position    = zeros(( self.NUMBER_OF_JOINTS, 2 ))
        self.right_arm_commanded_position = zeros(( self.NUMBER_OF_JOINTS, 2 ))
        self.right_arm_desired_position   = zeros(( self.NUMBER_OF_JOINTS, 2 ))

    #-------------------------------------------------------------------------
    #-------------------------------------------------------------------------
    #-------------------------------------------------------------------------
    def update_sensor_poses(self) :

        return  array([ 
            self.body.get_point(x) 
            for x in linspace(0,1,self.NUMBER_OF_SENSORS)
            ])

#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------

class Simulator :

    #-------------------------------------------------------------------------
    #-------------------------------------------------------------------------
    #-------------------------------------------------------------------------
    def __init__(self) :

        # time start in trial ------------------------------------------------
        self.TRIAL_START                       = 2     
        # time interval in trial ---------------------------------------------
        self.TRIAL                             = 20     
        # learning rate ------------------------------------------------------
        self.GOAL_TH                           = 0.9


        self.learner = Learner(stime = self.TRIAL)
        self.predictor = Predictor()
        self.controller = Controller()
        
        assert( self.learner.NUMBER_OF_TARGETS == self.controller.NUMBER_OF_SENSORS)
        assert( self.learner.NUMBER_OF_TARGETS == self.predictor.NUMBER_OF_STATES)
        
        self.trial_counter = 0
        self.trial_timestep = 0
        self.goal_switch = False
        self.test_switch = False

        self.state = zeros(self.predictor.NUMBER_OF_STATES)
        self.sensitivity_mask = ones([2, self.predictor.NUMBER_OF_STATES])
        self.occurrencies = zeros(self.predictor.NUMBER_OF_STATES)
       
    #-------------------------------------------------------------------------
    #-------------------------------------------------------------------------
    #-------------------------------------------------------------------------
    def reset(self) :

        self.learner.reset()
        self.controller.reset()

        self.occurrencies = zeros(self.predictor.NUMBER_OF_STATES)
          
    #-------------------------------------------------------------------------
    #-------------------------------------------------------------------------
    #-------------------------------------------------------------------------
    def learn(self) :

        if self.trial_timestep > self.TRIAL_START :
            if self.goal_switch == True :
                self.learner.learn()
    
    #-------------------------------------------------------------------------
    #-------------------------------------------------------------------------
    #-------------------------------------------------------------------------
    def update_goal(self) :
        
        self.curr_sens = self.sensitivity_mask[int(self.goal_switch),:]
        self.state *= self.curr_sens 

        if self.goal_switch == False :
            if sum(self.state)>0 : 
                print "{:04d}: ACTIVATE".format(self.trial_timestep)
                self.goal_switch = True
                self.learner.reset_noise()
                self.learner.update_target(self.state)
                self.sensitivity_mask[int(self.goal_switch),:] = self.state
            else :
                print "{:04d}: OFF".format(self.trial_timestep)
                if rand() < 0.05:
                    print "{:04d}: OFF + NOISE".format(self.trial_timestep)
                    self.learner.reset_noise()
        
        elif self.goal_switch == True :
            
            curr_mem = dot(self.curr_sens, self.predictor.memory)
            print "{:04d}: ON - MEM = {:03.2f} - INDEX = {:02d} ".format(
                    self.trial_timestep, curr_mem, 
                    find(self.curr_sens==1)[0]
                    )
            if curr_mem > self.GOAL_TH :
                print "{:04d}: DEACTIVATE".format(self.trial_timestep)
                self.goal_switch = False
            else:
                self.learner.reset_noise((1-curr_mem)*0.3)
                if sum(self.state>0) : 
                    self.occurrencies += self.state 
                    print "{:04d}: ON + TOUCH - INDEX = {:02d} ".format(
                            self.trial_timestep,
                            find(self.state==1)[0]
                            )
                    self.learner.update_target(self.state)
            
            if self.trial_timestep == (self.TRIAL-1) :
                print "                        OCCUR = {:02.0f}".format(
                    sum( self.occurrencies)
                    )

        if self.trial_timestep == (self.TRIAL-1) :
            self.predictor.update(sum(self.occurrencies)/self.TRIAL, 
                    self.curr_sens )
         
    #-------------------------------------------------------------------------
    #-------------------------------------------------------------------------
    #-------------------------------------------------------------------------
    def step(self,t) :

        self.trial_counter = floor(t/self.TRIAL)
        self.trial_timestep = t%self.TRIAL

        if self.trial_counter%20 >= 10 :
            self.test_switch = True
        else :
            self.test_switch = False

        if self.trial_timestep == 0 :
            self.reset()
   
        # spreading
        self.learner.activation_step(t, self.controller.body_sensors)  
        
        # update positions
        self.controller.set_arms_positions(
                self.learner.noised,
                self.learner.readouts,
                self.learner.current_target_readouts )
        actor_pos = self.controller.get_hands_positions(Controller.ACTUAL)
        
        # update noise 
        self.learner.update_noise()
        
        # update sensors
        self.state = self.controller.get_sensors()
       
        if self.test_switch == False :
            
            self.update_goal()
            
            # learning
            self.learn()

if __name__ == "__main__" :

    simulator = Simulator()
    for t in xrange(100) :
        simulator.step(t)

