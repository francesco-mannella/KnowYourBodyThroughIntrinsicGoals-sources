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
    
        self.controller = Controller.SensorimotorController(       
                pixels = [20, 20],
                lims = [[-5, 5], [-2, 4.]],
<<<<<<< HEAD
                touch_th = 0.8, 
                touch_sensors = 0  )
=======
                touch_th = 0.8,
                fovea_radius=.8)
>>>>>>> b6a3eeb9bf2f021198862582fb1a23a57ec655cb

        self.GOAL_NUMBER = 9

        self.gs = GoalSelector.GoalSelector(
                dt = 0.001,
                tau = 0.008,
                alpha = 0.1,
                epsilon = 1.0e-10,
                eta = 0.04,
                n_input = self.controller.pixels[0]*self.controller.pixels[0],
                n_goal_units = self.GOAL_NUMBER,
                n_echo_units = 100,
                n_rout_units = self.controller.actuator.NUMBER_OF_JOINTS*2,
<<<<<<< HEAD
                im_decay = 0.5,
=======
                im_decay = 0.05,
>>>>>>> b6a3eeb9bf2f021198862582fb1a23a57ec655cb
                match_decay = 0.5,
                noise = .5,
                sm_temp = 0.2,
                g2e_spars = 0.2,
                echo_ampl = 5.0,
                goal_window = 100,
<<<<<<< HEAD
                goal_learn_start = 20,
                reset_window = 10
=======
                reset_window = 10,
                eye_pos = self.controller.eye_pos
>>>>>>> b6a3eeb9bf2f021198862582fb1a23a57ec655cb
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
                n_goalrep= self.GOAL_NUMBER,
<<<<<<< HEAD
                singlemod_lrs = [0.05, 0.05, 0.05],
                hidden_lrs=[0.001, 0.001],
                output_lr=0.001,
                goalrep_lr=0.001,
=======
                singlemod_lrs = [0.1, 0.1, 0.1],
                hidden_lrs=[0.005, 0.005],
                output_lr=0.001,
                goalrep_lr=0.08,
>>>>>>> b6a3eeb9bf2f021198862582fb1a23a57ec655cb
                goal_th=0.1
            )


        self.stime = 10000

        self.trial_window = self.gs.RESET_WINDOW + self.gs.GOAL_WINDOW
        
        self.goal_mask = np.zeros(self.gs.N_GOAL_UNITS).astype("bool")
        
        self.match_value = False

        self.intrinsic_motivation_value = 0.0
    
        self.static_inp = np.zeros(self.gs.N_INPUT)
   
        self.eye_pos = np.zeros(2)

    def get_selection_arrays(self) :

        sel = self.gs.goal_selected
        
        gmask = self.goal_mask.astype("float")
        gv = self.gs.goalvec
        gw = self.gs.goal_win
        gr = self.gm.goalrep_layer
        
        goal_keys = self.gs.target_position.keys()
        acquired_targets = []
        if len(goal_keys) != 0:
            acquired_targets = self.gs.get_goal_from_index(self.gs.target_position.keys()) 
            acquired_eye_poss = self.gs.get_eye_pos_from_index(self.gs.target_position.keys())
       
            g = np.vstack(( acquired_targets, np.vstack(acquired_eye_poss).T )).T

            for r in g:
                print "Robot:100  goal: {}  x: {}  y: {}".format(*r)
            print

        targets = np.array([ 1.0*(target in acquired_targets) 
            for target in np.arange(self.gs.N_GOAL_UNITS) ])
        esn_data = self.gs.echonet.data[self.gs.echonet.out_lab]

        return gmask, gv, gw, gr, targets, esn_data
        

    def get_sensory_arrays(self) :

        i1, i2, i3 = self.gm.input_layers
        sm1, sm2, sm3 = self.gm.singlemod_layers
        h1, h2 = self.gm.hidden_layers
        wsm1, wsm2, wsm3 = (som.inp2out_w for som in self.gm.singlemod_soms)
        wh1, wh2 = (som.inp2out_w for som in self.gm.hidden_soms)
        wo1 = self.gm.out_som.inp2out_w
        o1 = self.gm.output_layer
        wgr1 = self.gm.goalrep_som.inp2out_w
        gr1 = self.gm.goalrep_layer

        return (i1, i2, i3, sm1, sm2, sm3, h1, 
                h2, o1, wsm1, wsm2, wsm3, wh1,
                wh2, wo1, wgr1, gr1)

    def get_arm_positions(self) :
        
        sel = self.gs.goal_selected
        
<<<<<<< HEAD
=======
        # real = np.pi*self.gs.out
        # self.controller.actuator.set_angles(
        #         real[:(self.gs.N_ROUT_UNITS/2)],
        #         real[(self.gs.N_ROUT_UNITS/2):]
        #         )
>>>>>>> b6a3eeb9bf2f021198862582fb1a23a57ec655cb
        real_l_pos = self.controller.actuator.position_l
        real_r_pos = self.controller.actuator.position_r
        
        real_l_pos *= sel
        real_r_pos *= sel

<<<<<<< HEAD
        goalwin_idx =  self.gs.goal_index()
        target_l_pos = self.controller.target_actuator.position_l
        target_r_pos = self.controller.target_actuator.position_r
        if  not self.gs.target_position.has_key(goalwin_idx) :
            target_l_pos *= 0
            target_r_pos *= 0

        target_l_pos *= sel
        target_r_pos *= sel

=======
   
        goalwin_idx = self.gs.goal_index()
        if goalwin_idx is not None and self.gs.target_position.has_key(goalwin_idx):
                target = np.pi*self.gs.target_position[goalwin_idx]
                self.controller.target_actuator.set_angles(
                        target[:(self.gs.N_ROUT_UNITS/2)],
                        target[(self.gs.N_ROUT_UNITS/2):],
                        )
                target_l_pos = self.controller.target_actuator.position_l
                target_r_pos = self.controller.target_actuator.position_r
        else :
            target_l_pos = self.controller.target_actuator.position_l*0
            target_r_pos = self.controller.target_actuator.position_r*0
        target_l_pos *= sel
        target_r_pos *= sel

        theoric = np.pi*self.gs.tout
        self.controller.theoric_actuator.set_angles(
                theoric[:(self.gs.N_ROUT_UNITS/2)],
                theoric[(self.gs.N_ROUT_UNITS/2):]
                )
>>>>>>> b6a3eeb9bf2f021198862582fb1a23a57ec655cb
        theor_l_pos = self.controller.theoric_actuator.position_l
        theor_r_pos = self.controller.theoric_actuator.position_r

        sensors = self.controller.perc.sensors * sel

        return (real_l_pos, real_r_pos, target_l_pos,
<<<<<<< HEAD
                target_r_pos, theor_l_pos, theor_r_pos, sensors)
=======
                target_r_pos, theor_l_pos, theor_r_pos, 
                sensors, self.controller.eye_pos, self.controller.fovea_radius )
>>>>>>> b6a3eeb9bf2f021198862582fb1a23a57ec655cb

    def step(self) :
   
        if self.gs.reset_window_counter >= self.gs.RESET_WINDOW:

            # update the subset of goals to be selected
            self.goal_mask = np.logical_or(self.goal_mask, (self.gm.goalrep_layer > 0) )

            # Selection
            if any(self.goal_mask==True):
                self.gs.goal_selection(
                        self.intrinsic_motivation_value, 
                        self.goal_mask,
                        eye_pos=self.eye_pos)
            else:
                self.gs.goal_selection(self.intrinsic_motivation_value)

            # Prediction
            if self.gs.goal_window_counter == 0:
                self.gp.step(self.gs.goal_win) 

            self.static_inp *= 0.0

        else:
            if self.gs.reset_window_counter == 0:
                self.static_inp = np.random.rand(*self.static_inp.shape)
                self.eye_pos = 0.1 + 0.8*(np.random.rand(2)*4).round(0)/4
        # Movement

        self.gs.step( self.static_inp )

        self.controller.step_kinematic(
                larm_angles=np.pi*self.gs.out[:(self.gs.N_ROUT_UNITS/2)],
                rarm_angles=np.pi*self.gs.out[(self.gs.N_ROUT_UNITS/2):],
<<<<<<< HEAD
                larm_angles_theoric=np.pi*self.gs.tout[:(self.gs.N_ROUT_UNITS/2)],
                rarm_angles_theoric=np.pi*self.gs.tout[(self.gs.N_ROUT_UNITS/2):],
                larm_angles_target=np.pi*self.gs.gout[:(self.gs.N_ROUT_UNITS/2)],
                rarm_angles_target=np.pi*self.gs.gout[(self.gs.N_ROUT_UNITS/2):],

=======
                eye_pos=self.eye_pos
>>>>>>> b6a3eeb9bf2f021198862582fb1a23a57ec655cb
                )

        if self.gs.reset_window_counter >= self.gs.RESET_WINDOW:

            self.gm.step([
<<<<<<< HEAD
                self.controller.pos_delta.ravel()*500.0,
                self.controller.prop_delta.ravel()*.5,
                self.controller.touch_delta.ravel()*5000.0])
=======
                self.controller.pos_delta.ravel()*5000.0,
                self.controller.prop_delta.ravel()*5000.0,
                self.controller.touch_delta.ravel()*5000.0] )
>>>>>>> b6a3eeb9bf2f021198862582fb1a23a57ec655cb

            self.gm.learn()


        if self.gs.reset_window_counter >= self.gs.RESET_WINDOW:
            
            # Train experts
            if  self.gs.goal_window_counter > self.gs.GOAL_LEARN_START :
                self.gs.learn()
            
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
