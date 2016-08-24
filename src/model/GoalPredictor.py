#!/usr/bin/env python

import sys

sys.path.append("../")

import numpy as np
import numpy.random as rnd
from nets.esn import ESN

def match(abstaction_layer, selection_layer) :
    '''
    :param abstaction_layer
    :param selection_layer
    '''
    al = np.array(abstaction_layer).astype("float")
    gl = np.array(selection_layer).astype("float")

    return np.any( np.logical_and(al>0,gl>0) )

class GoalPredictor :

    def __init__(self, n_goal_units, eta) :
        '''
        :param n_goal_units: number of units in the goal layer
        :param eta: learning rate
        '''
        
        self.N_GOAL_UNITS = n_goal_units
        self.ETA = eta

        self.inp_layer = np.zeros(self.N_GOAL_UNITS)
        self.w = np.zeros(self.N_GOAL_UNITS)
        self.goal_win = np.zeros(self.N_GOAL_UNITS)

        self.prediction_error = 0.0
        self.out = 0.0
        

    def step(self, goal_win):

        self.goal_win = goal_win
        self.out = np.dot(self.w,self.goal_win)

    def learn(self, match):
        
        self.w += self.ETA*self.goal_win*(match - self.out)
        self.prediction_error = np.maximum(0.0, match - self.out)

