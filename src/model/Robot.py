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
import sys
sys.path.append("../")
import numpy as np

import GoalSelector
import GoalPredictor
import GoalMaker
import Controller
import utils.kinematics as KM


class Robot(object) :

    def __init__(self) :
    
        self.controller = Controller.SensorimotorController(       
                pixels = [20, 20],
                lims = [[-5, 5], [-2, 4.]],
                touch_th = 0.8, 
                touch_sensors = 18,
                touch_sigma=0.25, 
                touch_len=0.01
                )

        self.GOAL_NUMBER = 49

        self.gs = GoalSelector.GoalSelector(
                dt = 0.001,
                tau = 0.015,
                alpha = 0.4,
                epsilon = 1.0e-10,
                eta = 0.04,
                n_input = self.controller.pixels[0]*self.controller.pixels[0],
                n_goal_units = self.GOAL_NUMBER,
                n_echo_units = 100,
                n_rout_units = self.controller.actuator.NUMBER_OF_JOINTS*2,
                im_decay = 0.9,
                match_decay = 0.5,
                noise = .5,
                sm_temp = 0.2,
                g2e_spars = 0.2,
                echo_ampl = 5.0,
                goal_window = 100,
                goal_learn_start = 20,
                reset_window = 10
                )

        self.gp = GoalPredictor.GoalPredictor(
                n_goal_units = self.GOAL_NUMBER,
                eta = 0.001
                )

       
        inp_dim = self.controller.pixels[0]*self.controller.pixels[1]
        
        self.stime = 10000
        
        self.gm = GoalMaker.GoalMaker(
                n_input_layers=[inp_dim, inp_dim, inp_dim],
                n_singlemod_layers= [64, 64, 64],
                n_hidden_layers=[16, 16],
                n_out=16,
                n_goalrep= self.GOAL_NUMBER,
                singlemod_lrs = [0.05, 0.01, 0.1],
                hidden_lrs=[0.001, 0.001],
                output_lr=0.001,
                goalrep_lr=0.8,
                goal_th=0.1,
                stime=self.stime,
                single_kohonen = True
            )



        self.trial_window = self.gs.RESET_WINDOW + self.gs.GOAL_WINDOW
        
        self.goal_mask = np.zeros(self.gs.N_GOAL_UNITS).astype("bool")
        
        self.match_value = False

        self.intrinsic_motivation_value = 0.0
    
        self.static_inp = np.zeros(self.gs.N_INPUT)
        
        self.collision = False

        self.init_streams()

        
        self.timestep = 0

    def init_streams(self):

        self.log_sensors = None
        self.log_position = None 
        self.log_predictions = None 
        self.log_targets = None 
        self.log_weights = None 

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
        matched_targets = np.array([ 1.0*(target in acquired_targets) 
            for target in np.arange(self.gs.N_GOAL_UNITS) ])
        targets = self.gp.w
        esn_data = self.gs.echonet.data[self.gs.echonet.out_lab]
 
        return gmask, gv, gw, gr,matched_targets, targets, self.gs.target_position, \
                esn_data 
        

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
        
        real_l_pos = self.controller.actuator.position_l
        real_r_pos = self.controller.actuator.position_r
        
        real_l_pos *= sel
        real_r_pos *= sel

        goalwin_idx =  self.gs.goal_index()
        target_l_pos = self.controller.target_actuator.position_l
        target_r_pos = self.controller.target_actuator.position_r
        if  not goalwin_idx in self.gs.target_position :
            target_l_pos *= 0
            target_r_pos *= 0

        target_l_pos *= sel
        target_r_pos *= sel

        theor_l_pos = self.controller.theoric_actuator.position_l
        theor_r_pos = self.controller.theoric_actuator.position_r

        sensors = self.controller.perc.sensors * sel

        return (real_l_pos, real_r_pos, target_l_pos,
                target_r_pos, theor_l_pos, theor_r_pos, sensors )

    def get_weights_metrics(self):

        res = dict()
        w = None
        
        # add norm of the kohonen weights
        if self.gm.SINGLE_KOHONEN :
            w = self.gm.goalrep_som.inp2out_w.ravel()
        else :
            w = np.hstack((
                np.hstack([som.inp2out_w.ravel for som in self.gm.singlemod_soms ]),
                np.hstack([som.inp2out_w.ravel for som in self.gm.hidden_soms ]),
                self.gm.out_som.inp2out_w.ravel(),
                self.gm.goalrep_som.inp2out_w.ravel() 
                ))
        res["kohonen_weights"] = np.linalg.norm(w)

        # add norm of the ESN weights
        w = self.gs.echo2out_w
        res["echo_weights"] = np.linalg.norm(w)

        return res  


    def step(self) :
  

        self.timestep += 1 

        if self.gs.reset_window_counter >= self.gs.RESET_WINDOW:
        

            # update the subset of goals to be selected
            self.goal_mask = np.logical_or(self.goal_mask, (self.gm.goalrep_layer > 0) )
            
            # Selection
            if any(self.goal_mask==True):
                self.gs.goal_selection(
                        self.goal_mask)
            else:
                self.gs.goal_selection(self.intrinsic_motivation_value)

            # Prediction
            if self.gs.goal_window_counter == 0:
                self.gp.step(self.gs.goal_win) 


            self.gm_input = [
                self.controller.pos_delta.ravel()*500.0 *0.0,
                self.controller.prop_delta.ravel()*500.0*0.0,
                self.controller.touch_delta.ravel()*5000.0]
            self.static_inp *= 0
            for inp in self.gm_input :
                self.static_inp += 0.0001*np.array(inp)

        else:
            if self.gs.reset_window_counter == 0:
                self.static_inp = np.random.rand(*self.static_inp.shape)
        
        # Movement
        self.gs.step( self.static_inp )

        larm_angles=np.pi*self.gs.out[:(self.gs.N_ROUT_UNITS//2)]
        rarm_angles=np.pi*self.gs.out[(self.gs.N_ROUT_UNITS//2):]
        larm_angles_theoric=np.pi*self.gs.tout[:(self.gs.N_ROUT_UNITS//2)]
        rarm_angles_theoric=np.pi*self.gs.tout[(self.gs.N_ROUT_UNITS//2):]
        larm_angles_target=np.pi*self.gs.gout[:(self.gs.N_ROUT_UNITS//2)]
        rarm_angles_target=np.pi*self.gs.gout[(self.gs.N_ROUT_UNITS//2):]

        collision = self.controller.step_kinematic(
                larm_angles=larm_angles,
                rarm_angles=rarm_angles,
                larm_angles_theoric=larm_angles_theoric,
                rarm_angles_theoric=rarm_angles_theoric,
                larm_angles_target=larm_angles_target,
                rarm_angles_target=rarm_angles_target,
                active=(self.gs.reset_window_counter >= self.gs.RESET_WINDOW)
                )
        
        self.collision = collision
        if self.collision :
            self.gs.out[:(self.gs.N_ROUT_UNITS//2)] = self.controller.larm_angles/np.pi
            self.gs.out[(self.gs.N_ROUT_UNITS//2):] = self.controller.rarm_angles/np.pi
   

        if self.gs.reset_window_counter >= self.gs.RESET_WINDOW:

            self.gm.step( self.gm_input )

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

                    if self.log_sensors is not None :

                        # save sensor info on file 

                        # create log line
                        log_string = ""
                        # add timing
                        log_string += "{:8d} ".format(self.timestep)
                        # add touch info
                        for touch in  self.controller.touches :
                            log_string += "{:6.4f} ".format(touch)
                        # add goal index
                        log_string += "{:6d} {:d}".format(np.argmax(self.gs.goal_win), self.timestep) 
                        # save to file
                        self.log_sensors.write( log_string + "\n")
                        self.log_sensors.flush()
                
                    if self.log_position is not None :
                        
                        # save position info on file 

                        # create log line
                        log_string = ""
                        # add timing
                        log_string += "{:8d} ".format(self.timestep)
                        # add position info
                        curr_position =  np.vstack(self.controller.curr_body_tokens).ravel() 
                        for pos in  curr_position:
                            log_string += "{:6.4f} ".format(pos)
                        # add goal index
                        log_string += "{:6d} {:d}".format(np.argmax(self.gs.goal_win), self.timestep) 
                        # save to file
                        self.log_position.write( log_string + "\n")
                        self.log_position.flush()
                     
                    if self.log_predictions is not None :
                        
                        # save predictions info on file 

                        # create log line
                        log_string = ""
                        # add timing
                        log_string += "{:8d} ".format(self.timestep)
                        # add predictions info
                        curr_predictions = self.gp.w
                        for pre in  curr_predictions:
                            log_string += "{:6.4f} ".format(pre)
                        # add goal index
                        log_string += "{:6d} {:d}".format(np.argmax(self.gs.goal_win), self.timestep) 
                        # save to file
                        self.log_predictions.write( log_string + "\n")
                        self.log_predictions.flush()
                                     
                    if self.log_targets is not None :
                        
                        # save targets info on file 

                        # create log line
                        log_string = ""
                        # add timing
                        log_string += "{:8d} ".format(self.timestep)
                        # add targets info
                        keys = sorted(self.gs.target_position.keys())
                        all_goals = np.NaN*np.ones( [self.gs.N_GOAL_UNITS, self.gs.N_ROUT_UNITS] )
                        
                        for key in sorted(self.gs.target_position.keys()):
                            all_goals[key,:] = self.gs.target_position[key] 

                        for angle in all_goals.ravel():
                                log_string += "{:6.4f} ".format(angle)

                        # add goal index
                        log_string += "{:6d} {:d}".format(np.argmax(self.gs.goal_win), self.timestep) 
                        # save to file
                        self.log_targets.write( log_string + "\n")
                        self.log_targets.flush()

                # learn

                self.gp.learn(self.match_value) 

                if self.match_value == 1:
                    self.gs.update_target()
              
                # save weights metrics
                if self.log_weights is not None :
                    # create log line
                    log_string = ""
                    # add timing
                    log_string += "{:8d} ".format(self.timestep)
                    wm = self.get_weights_metrics()
                    log_string += "{:6.4} ".format( wm["kohonen_weights"] )
                    log_string += "{:6.4} ".format( wm["echo_weights"] )

                    self.log_weights.write( log_string + "\n")
                    self.log_weights.flush()

                # update variables

                self.intrinsic_motivation_value = self.gp.prediction_error 
                self.gs.goal_update(self.intrinsic_motivation_value)

                self.gs.goal_selected = False
                self.gs.reset(match = self.match_value)
                self.controller.reset()
                
        else:
            
            self.gs.reset_window_counter += 1      
