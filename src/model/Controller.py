#!/usr/bin/env python
from __future__ import division
import sys
sys.path.append("../")
import numpy as np
import time

from utils.gauss_utils import TwoDimensionalGaussianMaker as GM
import utils.kinematics as KM

#    
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------

class PerceptionManager(object) :
    def __init__(self, pixels=[20,20], lims=[[0,1],[0,1]], 
            epsilon=0.2, touch_th = 0.5, touch_sensors = 0, 
            touch_sigma=0.2, touch_len=0.5):

        self.pixels = np.array(pixels)
        self.lims = np.vstack(lims) 
        self.epsilon = epsilon
        self.size = (self.lims[:,1]-self.lims[:,0])
        self.ycenter = self.lims[1,0]+ self.size[1]/2
        self.bins = self.size/self.pixels
        self.chain = KM.Polychain()
        self.init_chains = []
        self.touch_sigma =  touch_sigma 
        self.sigma =  self.size*0.1
        self.sigma[0] *=  0.2
        self.gm = GM(lims = np.hstack([self.lims,self.pixels.reshape(2,1)]))
        self.touch_sensors = touch_sensors 
        self.touch_th = touch_th
        self.image_resolution = 10 

        self.chain.set_chain(np.vstack(([-1,0],[1,0])))
        self.sensors_prev = self.chain.get_dense_chain(self.touch_sensors)        
        self.sensors = self.chain.get_dense_chain(self.touch_sensors)        
    
    def get_image(self, body_tokens ):

        image = np.zeros(self.pixels)
        
        for c in body_tokens:
            self.chain.set_chain(c)
            dense_chain = self.chain.get_dense_chain(self.image_resolution)
            for point in dense_chain:
                p = np.floor(((point -self.lims[:,0])/self.size)*self.pixels).astype("int")
                if np.all(p<self.pixels) and np.all(p>0) :
                    image[p[0],p[1]] += 1
            image /= float(len(point))
        image /= float(len(body_tokens))
        
        return image.T

    def get_proprioception(self, angles_tokens):

        image = np.zeros(self.pixels).astype("float")
        all_angles = np.hstack(angles_tokens)
        lims = np.hstack([self.lims, len(all_angles)*np.ones([2,1]) ])
        abs_body_tokens = np.vstack([ np.linspace(*lims[0]), 
            self.ycenter*np.ones(len(all_angles))]).T
        for point, angle in zip(abs_body_tokens, all_angles):
            if abs(angle) > 1e-5 :
                sigma = abs(angle)*self.sigma 
                g = self.gm(point, sigma)[0]
                image += g.reshape(*self.pixels)*angle
        
        return image.T
    
    def calc_collision(self, body_tokens):
        
        bts = np.vstack(body_tokens)
        self.chain.set_chain( bts ) 
        colliding = self.chain.autocollision(is_set_collinear=True)
        
        return colliding

    def get_touch(self, body_tokens):
 
        image = np.zeros(self.pixels)
        
        bts = np.vstack(body_tokens)
        self.chain.set_chain( bts )
  
        self.sensors_prev = self.sensors   
        self.sensors = self.chain.get_dense_chain(self.touch_sensors)   
        sensors_n = len(self.sensors)
        touches = np.zeros(sensors_n)
       
        type(self.sensors)
        
        for x,sensor  in zip( [0,sensors_n-1], 
                [ self.sensors[0], self.sensors[-1] ] ):
            for y,point in  zip(range(sensors_n), self.sensors):        
                if x != y and abs(x-y)>2:
                    touches[y] += \
                    np.exp(-((np.linalg.norm(point - sensor))**2)/\
                            (2*self.touch_sigma**2)  )
        

        lims = np.hstack([self.lims[0], sensors_n+2])
        abs_body_tokens = np.vstack([ np.linspace(*lims), 
            self.ycenter*np.ones(sensors_n+2)]).T

        for touch,point in  zip(touches, abs_body_tokens[1:-1]) :
            g = self.gm(point, 1e-5+touch*self.sigma )[0]
            image += g.reshape(*self.pixels)*(touch>self.touch_th)
         
        return image.T, touches

#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------

class KinematicActuator(object) :

    #-------------------------------------------------------------------------
    #-------------------------------------------------------------------------
    #-------------------------------------------------------------------------
    def __init__(self) :
        self.NUMBER_OF_JOINTS = 3
        self.L_ORIGIN = [-1.5, 0.0]
        self.R_ORIGIN = [1.5, 0.0]
        self.arm_l = KM.Arm(
            number_of_joint = self.NUMBER_OF_JOINTS,    # 3 joints
            origin = self.L_ORIGIN, # origin at (1.0 , 0.5)
            joint_lims = [
                [0, np.pi*0.9],    # first joint limits
                [0, np.pi*0.5],    # second joint limits
                [0, np.pi*0.5]     # third joint limits
                ],
            mirror=True
        )
        self.arm_r = KM.Arm(
            number_of_joint = self.NUMBER_OF_JOINTS,    # 3 joints
            origin = self.R_ORIGIN, # origin at (1.0 , 0.5)
            joint_lims = [
                [0, np.pi*0.9],    # first joint limits
                [0, np.pi*0.5],    # second joint limits
                [0, np.pi*0.5],     # third joint limits
                ]
            )
        self.angles_l = np.zeros(self.NUMBER_OF_JOINTS)
        self.angles_r = np.zeros(self.NUMBER_OF_JOINTS)
        self.position_l, _ = self.arm_l.get_joint_positions(self.angles_l)
        self.position_r, _ = self.arm_r.get_joint_positions(self.angles_r)


    #-------------------------------------------------------------------------
    #-------------------------------------------------------------------------
    #-------------------------------------------------------------------------
    def set_angles(self, angles_l, angles_r) :
        self.angles_l = angles_l
        self.angles_r = angles_r
        self.position_l,_  = self.arm_l.get_joint_positions(self.angles_l)
        self.position_r,_  = self.arm_r.get_joint_positions(self.angles_r)

    #-------------------------------------------------------------------------
    #-------------------------------------------------------------------------
    #-------------------------------------------------------------------------
    def reset(self) :
        self.set_angles(self.angles_l*0.0,self.angles_r*0.0)



#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------


class SensorimotorController(object) :

    def __init__(self, pixels, lims, touch_th, **kargs):

        self.pixels = pixels 
        self.lims = lims

        self.actuator = KinematicActuator()
        self.theoric_actuator = KinematicActuator()
        self.target_actuator = KinematicActuator()

        n_joints = self.actuator.NUMBER_OF_JOINTS
        self.larm_angles = np.zeros(n_joints)
        self.rarm_angles = np.zeros(n_joints)
        self.larm_delta_angles = np.zeros(n_joints)
        self.rarm_delta_angles = np.zeros(n_joints)
        self.larm_angles_theoric = self.larm_angles
        self.rarm_angles_theoric = self.rarm_angles
        self.larm_angles_target = self.larm_angles
        self.rarm_angles_target = self.rarm_angles

        self.pos = np.zeros(self.pixels)
        self.pos_delta = np.zeros(self.pixels)
        self.pos_old = np.zeros(self.pixels)

        self.prop = np.zeros(self.pixels)
        self.prop_delta = np.zeros(self.pixels)
        self.prop_old  = np.zeros(self.pixels)

        self.touch = np.zeros(self.pixels)
        self.touch_delta = np.zeros(self.pixels)
        self.touch_old  = np.zeros(self.pixels)

        self.actuator.set_angles(np.array([0., 0, 0]),
                             np.array([0., 0, 0]))
        self.theoric_actuator.set_angles(np.array([0., 0, 0]),
                             np.array([0., 0, 0]))   
        self.target_actuator.set_angles(np.array([0., 0, 0]),
                             np.array([0., 0, 0])) 
        
        self.init_body_tokens = (self.actuator.position_l[::-1],        
                                 self.actuator.position_r)
        
        self.perc = PerceptionManager(  epsilon=0.1, 
                lims=lims, pixels=pixels, **kargs )

        self.touches = np.zeros(len(self.perc.sensors))
        self.curr_body_tokens = self.init_body_tokens

    def step_kinematic(self, larm_angles, rarm_angles, 
            larm_angles_theoric, rarm_angles_theoric, 
            larm_angles_target, rarm_angles_target, active=True ):

        self.larm_delta_angles_prev  = self.larm_delta_angles 
        self.rarm_delta_angles_prev  = self.rarm_delta_angles 
        self.larm_angles_prev  = self.larm_angles 
        self.rarm_angles_prev  = self.rarm_angles 
        self.larm_angles_theoric_prev  = self.larm_angles_theoric 
        self.rarm_angles_theoric_prev  = self.rarm_angles_theoric 
        self.larm_angles_target_prev  = self.larm_angles_target 
        self.rarm_angles_target_prev  = self.rarm_angles_target 
        self.pos_old = self.pos
        self.prop_old = self.prop
        self.touch_old = self.touch 

        self.larm_delta_angles = larm_angles - self.larm_angles
        self.rarm_delta_angles = rarm_angles - self.rarm_angles
        self.larm_angles = larm_angles[::-1]
        self.rarm_angles = rarm_angles
        self.larm_angles_theoric = larm_angles_theoric[::-1]
        self.rarm_angles_theoric = rarm_angles_theoric
        self.larm_angles_target = larm_angles_target[::-1]
        self.rarm_angles_target = rarm_angles_target

        self.actuator.set_angles(self.larm_angles, self.rarm_angles)
        self.theoric_actuator.set_angles(self.larm_angles_theoric, self.rarm_angles_theoric)
        self.target_actuator.set_angles(self.larm_angles_target, self.rarm_angles_target)
        
        self.larm_angles, self.rarm_angles = (self.actuator.angles_l, 
                self.actuator.angles_r)

        self.curr_body_tokens = (self.actuator.position_l[::-1], 
            self.actuator.position_r) 
            
        autocollision = self.perc.calc_collision(body_tokens=self.curr_body_tokens)

        if not autocollision :

            # VISUAL POSITION
            self.pos = self.perc.get_image(body_tokens=self.curr_body_tokens)

            # PROPRIOCEPTION
            angles_tokens = (self.larm_angles, self.rarm_angles)
            self.prop = self.perc.get_proprioception(
                    angles_tokens=angles_tokens)
            #TOUCH
            self.touch, self.touches = self.perc.get_touch(body_tokens=self.curr_body_tokens)

        else :
            self.larm_delta_angles  = self.larm_delta_angles_prev
            self.rarm_delta_angles  = self.rarm_delta_angles_prev
            self.larm_angles  = self.larm_angles_prev
            self.rarm_angles  = self.rarm_angles_prev
            self.larm_angles_theoric  = self.larm_angles_theoric_prev
            self.rarm_angles_theoric  = self.rarm_angles_theoric_prev
            self.larm_angles_target  = self.larm_angles_target_prev
            self.rarm_angles_target  = self.rarm_angles_target_prev

            self.actuator.set_angles(self.larm_angles, self.rarm_angles)
            self.theoric_actuator.set_angles(self.larm_angles_theoric, self.rarm_angles_theoric)
            self.target_actuator.set_angles(self.larm_angles_target, self.rarm_angles_target)

            self.larm_angles, self.rarm_angles = (self.actuator.angles_l, 
                    self.actuator.angles_r)

      
        delta_angles_tokens = (self.larm_delta_angles,
            self.rarm_delta_angles) 


        # deltas
        self.pos_delta = self.perc.get_image(
                body_tokens=self.curr_body_tokens)
        self.prop_delta = self.perc.get_proprioception(
                angles_tokens=delta_angles_tokens)
        self.touch_delta = self.touch - self.touch_old
            
        return  active and autocollision 

    def reset(self):
        self.pos_old *=0 
        self.pos *=0 
        self.prop_old *=0 
        self.prop *=0 
        self.touch_old *=0 
        self.touch *=0 

