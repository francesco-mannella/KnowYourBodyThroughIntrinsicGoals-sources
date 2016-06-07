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

class PerceptionManager:
    def __init__(self, pixels=[20,20], lims=[[0,1],[0,1]], 
            epsilon=0.2, touch_th = 0.5, touch_sensors = 0):

        self.pixels = np.array(pixels)
        self.lims = np.vstack(lims) 
        self.epsilon = epsilon
        self.size = (self.lims[:,1]-self.lims[:,0])
        self.ycenter = self.lims[1,0]+ self.size[1]/2
        self.bins = self.size/self.pixels
        self.chain = KM.Polychain()
        self.init_chains = []
        self.touch_sigma =  .8 
        self.sigma =  self.size*0.1
        self.sigma[0] *=  0.2
        self.gm = GM(lims = np.hstack([self.lims,self.pixels.reshape(2,1)]))
        self.touch_sensors = touch_sensors 
        self.touch_th = touch_th
        self.touch_len = .1
        self.im_res = 10 

        self.chain.set_chain(np.vstack(([-1,0],[1,0])))
        self.sensors = self.chain.get_dense_chain(self.touch_sensors)       
       
      
    
    def get_image(self, body_tokens ):

        image = np.zeros(self.pixels)
        for c in body_tokens:
            self.chain.set_chain(c)
            dense_chain = self.chain.get_dense_chain(self.im_res)
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

    def get_touch(self, body_tokens):
 
        image = np.zeros(self.pixels)
        
        bts = np.vstack(body_tokens)
        self.chain.set_chain( bts )
        self.sensors = self.chain.get_dense_chain(self.touch_sensors)   
        length = self.touch_len 
        sensors_n = len(self.sensors)
        touches = np.zeros(sensors_n)
        
        for x,sensor  in zip(range(sensors_n), self.sensors):
            for y,point in  zip(range(sensors_n), self.sensors):
                x_dist = x/float(sensors_n-1) 
                y_dist = y/float(sensors_n-1)
                
                if abs(x_dist - y_dist) > length:
                    touches[x] += \
                    np.exp(-((np.linalg.norm(point - sensor))**2)/\
                            (2*self.touch_sigma**2)  )
       
        lims = np.hstack([self.lims[0], sensors_n+2])
        abs_body_tokens = np.vstack([ np.linspace(*lims), 
            self.ycenter*np.ones(sensors_n+2)]).T

        for touch,point in  zip(touches, abs_body_tokens[1:-1]) :
            g = self.gm(point, 1e-5+touch*self.sigma )[0]
            image += g.reshape(*self.pixels)*(touch>self.touch_th)
         
        return image.T

#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------

class KinematicActuator :

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


class SensorimotorController:

    def __init__(self, pixels, lims, touch_th, fovea_radius = 10, 
            touch_sensors = 0):

        self.pixels = pixels 
        self.lims = lims
        self.fovea_radius = fovea_radius

        self.actuator = KinematicActuator()
        self.theoric_actuator = KinematicActuator()
        self.target_actuator = KinematicActuator()

        n_joints = self.actuator.NUMBER_OF_JOINTS
        self.larm_angles = np.zeros(n_joints)
        self.rarm_angles = np.zeros(n_joints)
        self.larm_delta_angles = np.zeros(n_joints)
        self.rarm_delta_angles = np.zeros(n_joints)

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
        
        self.perc = PerceptionManager(pixels=self.pixels,
                lims=self.lims,  epsilon=0.1, 
                touch_th=touch_th, touch_sensors=touch_sensors)

    def step(self, larm_delta_angles, rarm_delta_angles):

        self.larm_delta_angles = larm_delta_angles[::-1]
        self.rarm_delta_angles = rarm_delta_angles
        self.larm_angles += larm_delta_angles
        self.rarm_angles += rarm_delta_angles

        self.actuator.set_angles(self.larm_angles, self.rarm_angles)
        self.larm_angles, self.rarm_angles = (self.actuator.angles_l, 
                self.actuator.angles_r)

        # VISUAL POSITION
        curr_body_tokens = (self.actuator.position_l,
                self.actuator.position_r)
        self.pos = self.perc.get_image(body_tokens=curr_body_tokens)
        self.pos_delta = self.pos - self.pos_old

        # PROPRIOCEPTION
        body_tokens = self.init_body_tokens
        angles_tokens = (self.larm_angles, self.rarm_angles )
        self.prop = self.perc.get_proprioception(
                angles_tokens=angles_tokens)
        delta_angles_tokens = (self.larm_delta_angles,
                self.rarm_delta_angles)
        self.prop_delta = self.perc.get_proprioception(
                body_tokens=body_tokens,
                angles_tokens=delta_angles_tokens)


        #TOUCH
        self.touch = self.perc.get_touch(
                body_tokens=curr_body_tokens )
        self.touch_delta = self.touch - self.touch_old

        self.pos_old = self.pos
        self.prop_old = self.prop
        self.touch_old = self.touch


    def step_kinematic(self, larm_angles, rarm_angles, 
            larm_angles_theoric, rarm_angles_theoric, 
            larm_angles_target, rarm_angles_target, eye_pos):

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

        # VISUAL POSITION
        curr_body_tokens = (self.actuator.position_l[::-1], 
                self.actuator.position_r)

        self.pos = self.perc.get_image(body_tokens=curr_body_tokens)
        self.pos_delta = self.pos - self.pos_old

        hand_pos = self.actuator.position_l[-1]
        g = self.perc.gm(hand_pos, self.fovea_radius)[0]
        hand_mask = g.reshape(*self.pixels).T
        g = self.perc.gm(eye_pos, self.fovea_radius)[0]
        fovea_mask = g.reshape(*self.pixels).T

        self.pos_delta *= hand_mask
        self.pos_delta *= fovea_mask

        # PROPRIOCEPTION
        body_tokens = self.init_body_tokens
        angles_tokens = (self.larm_angles, self.rarm_angles)
        self.prop = self.perc.get_proprioception(
                angles_tokens=angles_tokens)
        delta_angles_tokens = (self.larm_delta_angles,
                self.rarm_delta_angles)
        self.prop_delta = self.perc.get_proprioception(
                angles_tokens=delta_angles_tokens)
 
        #TOUCH
        self.touch = self.perc.get_touch(
                body_tokens=curr_body_tokens)
        self.touch_delta = self.touch - self.touch_old

        self.pos_old = self.pos
        self.prop_old = self.prop
        self.touch_old = self.touch

    def reset(self):
        self.pos_old *=0 
        self.pos *=0 
        self.prop_old *=0 
        self.prop *=0 
        self.touch_old *=0 
        self.touch *=0 


