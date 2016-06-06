#!/usr/bin/env python

import sys

sys.path.append("../")

import numpy as np
import numpy.random as rnd
from nets.esn import ESN

def softmax(x, t=0.1):
    '''
    :param x: array of values
    :param t: temperature
    :return: softmax of values
    '''
    e = np.exp(x/t)
    return e/np.sum(e)

def my_argwhere(x) :
    
    res = np.nonzero(x)[0]

    return res

def oscillator(x, scale, p) :
    
    x = np.array(x)
    p = np.array(p)
    pam = p[:(p.size/2)]
    pph = p[(p.size/2):]
    x = np.outer(x, np.ones(pam.shape))

    return 0.5*np.pi*np.cos(pam*np.pi*(x/scale-pph))


class GoalSelector :

    def __init__(self, dt, tau, alpha, epsilon, eta, n_input,
            n_goal_units, n_echo_units, n_rout_units,
            im_decay, match_decay, noise, sm_temp, g2e_spars,
            goal_window, goal_learn_start, reset_window, echo_ampl=1000):
        '''
        :param dt: integration time of the ESN
        :param tau: decay of the ESN
        :param alpha: infinitesimal rotation coefficent of the ESN
        :param epsilon:  ESN distance from the unit spectral radius
        :param eta: learning rate
        :param n_input: number of inputs
        :param n_goal_units: number of units in the goal layer
        :param n_echo_units: number of units in the ESN
        :param n_rout_units: number of actuators
        :param im_decay: decay of the intrinsic trace
        :param match_decay: decay of the matching trace
        :param noise: standard deviation of white noise in the actuators
        :param sm_temp: temperature of the softmax
        :param g2e_spars: sparseness of weights form goals to esn
        :param goal_window: max duration of a goal selection
        :param goal_learn_start: start of learning during trial 
        :param reset_window: duration of reset
        :param echo_ampl: amplitude of the input to the echo-state
        '''

        self.DT = dt
        self.TAU = tau
        self.ALPHA = alpha
        self.EPSILON = epsilon
        self.ETA = eta
        self.IM_DECAY = im_decay
        self.MATCH_DECAY = match_decay
        self.NOISE = noise
        self.SM_TEMP = sm_temp
        self.GOAL_WINDOW = goal_window
        self.GOAL_LEARN_START = goal_learn_start
        self.RESET_WINDOW = reset_window

        self.N_INPUT = n_input
        self.N_GOAL_UNITS = n_goal_units
        self.N_ECHO_UNITS = n_echo_units
        self.N_ROUT_UNITS = n_rout_units
        self.GOAL2ECHO_SPARSENESS = g2e_spars
        self.ECHO_AMPL = echo_ampl

        self.goalvec = np.zeros(self.N_GOAL_UNITS)
        self.goal_win = np.zeros(self.N_GOAL_UNITS)
        self.goal_window_counter = 0
        self.reset_window_counter = 0

        self.echonet = ESN(
                N       = self.N_ECHO_UNITS,
                stime   = self.GOAL_WINDOW,
                dt      = self.DT,
                tau     = self.TAU,
                alpha   = self.ALPHA,
                beta    = 1-self.ALPHA,
                epsilon = self.EPSILON
                )

        # input -> ESN
        self.INP2ECHO_W = np.zeros([self.N_ECHO_UNITS, 
            self.N_INPUT+ self.N_GOAL_UNITS])
        
        self.INP2ECHO_W[(self.N_ECHO_UNITS/2):, :] = \
                np.random.randn(self.N_ECHO_UNITS/2, 
                        self.N_INPUT+ self.N_GOAL_UNITS)
        
        self.INP2ECHO_W[(self.N_ECHO_UNITS/2):, :] *= \
                (np.random.rand((self.N_ECHO_UNITS/2),
                    self.N_INPUT+ self.N_GOAL_UNITS)<self.GOAL2ECHO_SPARSENESS)
      
        # goal_layer -> ESN
        self.GOAL2ECHO_W = np.zeros([self.N_ECHO_UNITS, 
            self.N_GOAL_UNITS])
       
        self.GOAL2ECHO_W[:(self.N_ECHO_UNITS/2), :] = \
                -np.random.rand(self.N_ECHO_UNITS/2, 
                        self.N_GOAL_UNITS )

        self.GOAL2ECHO_W[:(self.N_ECHO_UNITS/2), :] *= \
                (np.random.rand((self.N_ECHO_UNITS/2),
                    self.N_GOAL_UNITS)<self.GOAL2ECHO_SPARSENESS)
 
        self.echo2out_w = 0.1*np.random.randn(self.N_ROUT_UNITS,
            self.N_ECHO_UNITS)

        self.read_out = np.zeros(self.N_ROUT_UNITS)
        self.out = np.zeros(self.N_ROUT_UNITS)
        self.tout = np.zeros(self.N_ROUT_UNITS)
        self.gout = np.zeros(self.N_ROUT_UNITS)
        self.target_position = dict()
        self.target_counter = dict()
        self.match_mean = np.zeros(self.N_GOAL_UNITS)
        self.curr_noise = 0.0

        self.goal_selected = False
        self.random_oscil = np.random.rand(2*self.N_ROUT_UNITS)
        self.t = 0

    def goal_index(self):
        
        idx = sum(2**(np.arange(self.N_GOAL_UNITS) )* self.goal_win)

        return idx
    
    def from_goal_index(self, idx):      
        return  np.log2(idx)


    def goal_selection(self, im_value, goal_mask = None):
        '''
        :param im_value: current intrinsic motivational value
        :param goal_mask: which goals can be selected
        '''
        
        # in case we do not have a mask create one 
        # ans select all goals as possible
        if goal_mask is None:
            goal_mask = np.ones(self.N_GOAL_UNITS)

        # if no goal has been selected
        if self.goal_selected == False :
            
            # the index of the current highest goal
            win_indx = np.argmax(self.goal_win)
            # update the movin' average for that goal
            self.goalvec[win_indx] += self.IM_DECAY*(
                    -self.goalvec[win_indx]  +im_value)

            # get indices of the currently avaliable goals
            curr_goal_idcs = my_argwhere(goal_mask>0)
            # get values of the averages of the currently 
            # avaliable goals 
            curr_goals = self.goalvec[curr_goal_idcs]
            # compute softmax between the currently avaliable goals
            sm = softmax(curr_goals, self.SM_TEMP )
            # cumulate probabilities between them
            cum_prob = np.hstack((0,np.cumsum(sm)))
            # flip the coin 
            coin = np.random.rand()
            # the winner between avaliable goals based on the flipped coin
            cur_goal_win = np.logical_and(cum_prob[:-1] < coin, cum_prob[1:] >= coin)
            # index of the winner in the vector of all goals
            goal_win_idx = curr_goal_idcs[my_argwhere(cur_goal_win==True)]
            # reset goal_win to all False values
            self.goal_win *= False
            # set the winner to True
            self.goal_win[goal_win_idx] = True 

            self.t = 0
            self.random_oscil = np.random.rand(2*self.N_ROUT_UNITS)

            self.goal_selected = True
            
            goalwin_idx = self.goal_index()
            try:
                target = self.target_position[goalwin_idx]
                self.gout = target 
            except KeyError : 
                self.gout = np.zeros(self.N_ROUT_UNITS) 



        

    def update_target(self):
        goalwin_idx = self.goal_index()

        try :
            self.target_counter[goalwin_idx] += 1
        except KeyError:
            self.target_counter[goalwin_idx] = 1
        
        try :
            pos_mean =  self.target_position[goalwin_idx]
            pos =  self.out
            n =  self.target_counter[goalwin_idx] 
            pos_mean += (1.0 - self.match_mean[self.goal_win>0])* (-pos_mean + pos)
            self.target_position[goalwin_idx] = pos_mean
        except KeyError: 
            pos =  self.out
            self.target_position[goalwin_idx] = pos


    def reset(self, match):
            self.match_mean += self.MATCH_DECAY*(
                    -self.match_mean + match)*self.goal_win
            self.goal_win *= 0
            self.goal_window_counter = 0
            self.reset_window_counter = 0
            self.echonet.reset()
            self.echonet.reset_data()
    
    def step(self, inp):
        '''
        :param im_value: current intrinsic motivational value
        :param match_value: current reward value
        :param inp: external input
        '''

        goal2echo_inp = np.dot(
                self.GOAL2ECHO_W,
                self.goal_win*self.goal_selected)

        inp2echo_inp = np.dot(
                self.INP2ECHO_W,
                np.hstack((
                    inp,
                    self.goal_win*self.goal_selected
                    )) )

        echo_inp = inp2echo_inp + goal2echo_inp
        self.echonet.step(self.ECHO_AMPL*echo_inp) 
        self.echonet.store(self.goal_window_counter)

        self.inp = self.echonet.out
        self.read_out = np.dot(self.echo2out_w, self.echonet.out)
        curr_match = np.squeeze(self.match_mean[self.goal_win>0])
        
        if np.all(self.goal_win==0):
            curr_match = 0.0
 
        added_signal = self.NOISE*oscillator(self.t, 10, self.random_oscil)[0]
        self.out = self.read_out + (1.0 - curr_match)*added_signal 
        self.tout = self.read_out 
         
        self.t += 1

    def learn(self):

        goalwin_idx = self.goal_index()

        #------------------------------------------------
        try:
            target = self.target_position[goalwin_idx]
            x = self.inp
            y = self.read_out
            eta = self.ETA
            w = self.echo2out_w
            w += eta*np.outer(target-y,x)
        except KeyError : pass
        #------------------------------------------------
        

