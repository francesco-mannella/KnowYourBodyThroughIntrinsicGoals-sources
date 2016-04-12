import sys
sys.path.append("../")

import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt

import GoalSelector
import GoalPredictor
import GoalMaker
import Controller
import utils.kinematics as KM


class Robot :

    def __init__(self) :
    
        self.controller = Controller.SensorimotorController()

        self.gs = GoalSelector.GoalSelector(
                dt = 0.001,
                tau = 0.08,
                alpha = 0.1,
                epsilon = 1.0e-10,
                eta = 0.01,
                n_input = self.controller.pixels[0]*self.controller.pixels[0],
                n_goal_units = 10,
                n_echo_units = 100,
                n_rout_units = self.controller.actuator.NUMBER_OF_JOINTS*2,
                im_decay = 0.2,
                noise = .5,
                sm_temp = 0.2,
                g2e_spars = 0.01,
                goal_window = 10,
                reset_window = 4
                )

        self.gp = GoalPredictor.GoalPredictor(
                n_goal_units = 2,
                eta = 0.01
                )

        inp_dim = self.controller.pixels[0]*self.controller.pixels[1]
        self.gm = GoalMaker.GoalMaker(
                n_input_layers=[inp_dim, inp_dim, inp_dim],
                n_singlemod_layers= [64, 64, 64],
                n_hidden_layers=[16, 16],
                n_out=16,
                n_goalrep= 10,
                singlemod_lrs = [0.3, 0.3, 0.3],
                hidden_lrs=[0.1, 0.1],
                output_lr=0.1,
                goalrep_lr=0.9,
                goal_th=0.1
            )


        self.stime = 10000

        self.trial_window = self.gs.RESET_WINDOW + self.gs.GOAL_WINDOW
        
        self.goal_mask = np.zeros(self.gs.N_GOAL_UNITS).astype("bool")
        
        self.match_value = False

        self.intrinsic_motivation_value = 0.0
    
        self.static_inp = np.zeros(self.gs.N_INPUT)
 
    def step(self) :
   
        if self.gs.reset_window_counter >= self.gs.RESET_WINDOW:

            # update the subset of goals to be selected
            self.goal_mask = np.logical_or(self.goal_mask, (self.gm.goalrep_layer > 0) )

            # Selection
            if any(self.goal_mask==True):
                self.gs.goal_selection(
                        self.intrinsic_motivation_value, 
                        self.goal_mask )
            else:
                self.gs.goal_selection(self.intrinsic_motivation_value)

            # Prediction
            if self.gs.goal_window_counter == 0:
                self.gp.step(self.gs.goal_win) 

            self.static_inp *= 0.0

        else:
            if self.gs.reset_window_counter == 0:
                self.static_inp = np.random.rand(*self.static_inp.shape)
        
        # Movement

        self.gs.step( self.static_inp )

        self.controller.step_kinematic(
                larm_angles=np.pi*self.gs.out[:(self.gs.N_ROUT_UNITS/2)],
                rarm_angles=np.pi*self.gs.out[(self.gs.N_ROUT_UNITS/2):]
                )

        if self.gs.reset_window_counter >= self.gs.RESET_WINDOW:

            self.gm.step([
                self.controller.pos_delta.ravel()*500.0,
                self.controller.prop_delta.ravel()*5000.0,
                self.controller.touch_delta.ravel()*5000.0])

            self.gm.learn()


        if self.gs.reset_window_counter >= self.gs.RESET_WINDOW:
            
            # Train experts
            
            self.gs.learn(match_value = self.match_value)
            
            # update counters
            
            self.gs.goal_window_counter += 1
            
            # End of trial
            
            self.match_value = GoalPredictor.match(
                    self.gm.goalrep_layer, 
                    self.gs.goal_win
                    )
            
            if self.match_value ==1 or self.gs.goal_window_counter >= self.gs.GOAL_WINDOW:
                
                if self.match_value == 1:
                    self.gs.update_target()
                
                self.intrinsic_motivation_value = self.gp.prediction_error 
                
                self.gs.goal_selected = False
                self.gs.reset(match = self.match_value)
                self.controller.reset()
                
        else:
            
            self.gs.reset_window_counter += 1      
