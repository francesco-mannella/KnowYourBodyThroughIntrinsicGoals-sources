#!/usr/bin/env python
from __future__ import division
import sys
sys.path.append("../")
import numpy as np
import time

from utils.gauss_utils import MultidimensionalGaussianMaker as GM
import utils.kinematics as KM

#    
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
class PerceptionManager:

    def __init__(self, pixels=[20,20], lims=[[0,1],[0,1]], epsilon=0.2, touch_th = 0.9):

        self.pixels = np.array(pixels)
        self.lims = np.vstack(lims) 
        self.epsilon = epsilon
        self.size = (self.lims[:,1]-self.lims[:,0])
        self.bins = self.size/self.pixels
        self.chain = KM.Polychain()
        self.sigma = self.bins*0.4
        self.gm = GM(lims = np.hstack([self.lims,self.pixels.reshape(2,1)]))
        self.touch_th = touch_th

    def get_image(self, body_tokens):

        image = np.zeros(self.pixels)
        for c in body_tokens:
            self.chain.set_chain(c)
            dense_chain = self.chain.get_dense_chain(5)
            for point in dense_chain:
                p = np.floor(((point -self.lims[:,0])/self.size)*self.pixels)
                if np.all(p<self.pixels) and np.all(p>0) :
                    image[p[0],p[1]] += 1
            image /= float(len(point))
        image /= float(len(body_tokens))

        return image.T

    def get_proprioception(self, body_tokens, angles_tokens):

        image = np.zeros(self.pixels).astype("float")
        for body_token, angles in zip(body_tokens, angles_tokens):
            for point, angle in zip(body_token, angles):
                if abs(angle) > 1e-5 :
                    g,_ = self.gm.get_gaussian(point, np.ones(2) * (self.sigma**2 + abs(angle) ))
                    image += g.reshape(*self.pixels)*angle

        return image.T

    def get_touch(self, body_tokens, body_tokens_sensors):

        sensors_points = []
        for point,is_sensor in zip(body_tokens, body_tokens_sensors):
            if is_sensor != 0 :
                sensors_points.append(point)

        image = np.zeros(self.pixels)
        for c in body_tokens:
            self.chain.set_chain(c)
            dense_chain = self.chain.get_dense_chain(2)
            for point in dense_chain:
                for sensor in  sensors_points :
                    touch = np.exp(-((np.linalg.norm(point) -
                                      np.linalg.norm(sensor))**2) )
                    g,_ = self.gm.get_gaussian(point, 0.01*np.ones(2) )
                    image += g.reshape(*self.pixels)*(touch>self.touch_th)

            image /= float(len(point))
        image /= float(len(body_tokens))

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
        self.ORIGIN = [0.0, 0.0]
        self.arm_l = KM.Arm(
            number_of_joint = self.NUMBER_OF_JOINTS,    # 3 joints
            origin = self.ORIGIN, # origin at (1.0 , 0.5)
            joint_lims = [
                [0, np.pi*0.5],    # first joint limits
                [0, np.pi*0.5],    # second joint limits
                [0, np.pi*0.5]     # third joint limits
                ],
            mirror=True
        )
        self.arm_r = KM.Arm(
            number_of_joint = self.NUMBER_OF_JOINTS,    # 3 joints
            origin = self.ORIGIN, # origin at (1.0 , 0.5)
            joint_lims = [
                [0, np.pi*0.5],    # first joint limits
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

    def __init__(self):

        self.pixels = [20, 20]
        self.lims =[[-3.5, 3.5],
                    [-0.5, 3.]]

        self.actuator = KinematicActuator()
        self.perc = PerceptionManager(pixels=self.pixels,
            lims=self.lims, epsilon=0.1)

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

        self.actuator.set_angles(np.array([np.pi / 6., 0, 0]),
                             np.array([np.pi / 6., 0, 0]))
        self.init_body_tokens = [self.actuator.position_l,
                                 self.actuator.position_r]
        self.touch_mask = [[0,0,0,1],[0,0,0,1]]

    def step(self, larm_delta_angles, rarm_delta_angles):

        self.larm_delta_angles = larm_delta_angles
        self.rarm_delta_angles = rarm_delta_angles
        self.larm_angles += larm_delta_angles
        self.rarm_angles += rarm_delta_angles

        self.actuator.set_angles(self.larm_angles, self.rarm_angles)
        self.larm_angles, self.rarm_angles = (self.actuator.angles_l, self.actuator.angles_r)

        # VISUAL POSITION
        curr_body_tokens = (self.actuator.position_l, self.actuator.position_r)
        self.pos = self.perc.get_image(body_tokens=curr_body_tokens)
        self.pos_delta = self.pos - self.pos_old


        # PROPRIOCEPTION
        body_tokens = self.init_body_tokens
        angles_tokens = np.vstack([self.larm_angles, self.rarm_angles ])
        self.prop = self.perc.get_proprioception(body_tokens= body_tokens,
                                            angles_tokens=angles_tokens * 0.1)
        angles_tokens = np.vstack([self.larm_delta_angles, self.rarm_delta_angles ])
        self.prop_delta = self.perc.get_proprioception(body_tokens=body_tokens,
                                                  angles_tokens=angles_tokens * 0.1)

        #TOUCH
        self.touch = self.perc.get_touch(body_tokens=curr_body_tokens,
                                         body_tokens_sensors=self.touch_mask)
        self.touch_delta = self.touch - self.touch_old

        self.pos_old = self.pos
        self.prop_old = self.prop
        self.touch_old = self.touch


    def step_kinematic(self, larm_angles, rarm_angles):

        self.larm_delta_angles = larm_angles - self.larm_angles
        self.rarm_delta_angles = rarm_angles - self.rarm_angles
        self.larm_angles = larm_angles
        self.rarm_angles = rarm_angles


        self.actuator.set_angles(self.larm_angles, self.rarm_angles)
        self.larm_angles, self.rarm_angles = (self.actuator.angles_l, self.actuator.angles_r)

        # VISUAL POSITION
        curr_body_tokens = (self.actuator.position_l, self.actuator.position_r)
        self.pos = self.perc.get_image(body_tokens=curr_body_tokens)
        self.pos_delta = self.pos - self.pos_old


        # PROPRIOCEPTION
        body_tokens = self.init_body_tokens
        angles_tokens = np.vstack([self.larm_angles, self.rarm_angles ])
        self.prop = self.perc.get_proprioception(body_tokens= body_tokens,
                                            angles_tokens=angles_tokens * 0.1)
        angles_tokens = np.vstack([self.larm_delta_angles, self.rarm_delta_angles ])
        self.prop_delta = self.perc.get_proprioception(body_tokens=body_tokens,
                                                  angles_tokens=angles_tokens * 0.1)

        #TOUCH
        self.touch = self.perc.get_touch(body_tokens=curr_body_tokens,
                                         body_tokens_sensors=self.touch_mask)
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



#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------

if __name__ == "__main__" :

    import matplotlib.pyplot as plt
    plt.ion()
    plt.close("all")

    #-------------------------------------------------------------------------
    #-------------------------------------------------------------------------
    #-------------------------------------------------------------------------

    controller = SensorimotorController()

    #-------------------------------------------------------------------------
    #-------------------------------------------------------------------------
    #-------------------------------------------------------------------------

    import matplotlib.gridspec as gridspec
    gs = gridspec.GridSpec(5, 2)

    fig = plt.figure(figsize=(8,12))


    ax = fig.add_subplot(gs[:2,:], aspect="auto")
    linel, = ax.plot(*np.zeros([2,4]))
    pointsl = ax.scatter(*np.zeros([2,4]))
    liner, = ax.plot(*np.zeros([2,4]))
    pointsr = ax.scatter(*np.zeros([2,4]))
    ax.set_xlim(controller.lims[0])
    ax.set_ylim(controller.lims[1])

    axes = []
    imgs = []

    image_default = np.zeros([20, 20])

    for x in xrange(3):
        for y in xrange(2):
            ax = fig.add_subplot(gs[2+x,y])
            img = ax.imshow(image_default, interpolation='none',
                   vmin=-1, vmax=1, cmap=plt.cm.binary)
            axes.append(ax)
            imgs.append(img)

    stime = 100000
    period = 10.0

    for t in  xrange(stime):

        #---------------------------------------------------------------------
        #---------------------------------------------------------------------
        #---------------------------------------------------------------------

        if np.mean(controller.actuator.angles_l) < np.pi*0.2 :
            ldelta = np.ones(controller.actuator.NUMBER_OF_JOINTS)*0.3+np.sin(t/period)*0.2
        else :
            ldelta = -np.ones(controller.actuator.NUMBER_OF_JOINTS)*0.3+np.sin(t/period)*0.2

        if np.mean(controller.actuator.angles_r) < np.pi*0.2 :
            rdelta = np.ones(controller.actuator.NUMBER_OF_JOINTS)*0.2+np.sin(t/period)*0.2
        else :
            rdelta = -np.ones(controller.actuator.NUMBER_OF_JOINTS)*0.2+np.sin(t/period)*0.2

        controller.step(larm_delta_angles=ldelta, rarm_delta_angles=rdelta)

        data =[controller.pos,
               controller.pos_delta,
               controller.prop*50.,
               controller.prop_delta*50.,
               controller.touch*50.,
               controller.touch_delta*50.
               ]

        #---------------------------------------------------------------------
        #---------------------------------------------------------------------
        #---------------------------------------------------------------------

        linel.set_data(*controller.ctrl.position_l.T)
        pointsl.set_offsets(controller.ctrl.position_l)
        liner.set_data(*controller.ctrl.position_r.T)
        pointsr.set_offsets(controller.ctrl.position_r)
        for x in xrange(6):
            d = data[x]
            d = d[::-1]
            imgs[x].set_array(d)
        fig.canvas.draw()

        plt.pause(0.00001)



    raw_input() 
