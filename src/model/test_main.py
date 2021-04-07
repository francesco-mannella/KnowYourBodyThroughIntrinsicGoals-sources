import sys
sys.path.append("../")
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

import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt

import GoalSelector
import GoalPredictor
import GoalMaker
import Controller
import utils.kinematics as KM

if __name__ == "__main__":


    controller = Controller.SensorimotorController()

    gs = GoalSelector.GoalSelector(
            dt = 0.001,
            tau = 0.08,
            alpha = 0.1,
            epsilon = 1.0e-10,
            eta = 0.01,
            n_input = controller.pixels[0]*controller.pixels[0],
            n_goal_units = 10,
            n_echo_units = 100,
            n_rout_units = controller.actuator.NUMBER_OF_JOINTS*2,
            im_decay = 0.2,
            noise = .5,
            sm_temp = 0.2,
            g2e_spars = 0.01,
            goal_window = 10,
            reset_window = 4
            )

    gp = GoalPredictor.GoalPredictor(
            n_goal_units = 2,
            eta = 0.01
            )

    inp_dim = controller.pixels[0]*controller.pixels[1]
    gm = GoalMaker.GoalMaker(
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


    stime = 10000

    trial_window = gs.RESET_WINDOW + gs.GOAL_WINDOW


    #----------------------------------------------------------
    # Plot

    plt.ion()

    fig = plt.figure()
    ax = fig.add_subplot(111,aspect="equal")
    larm, = ax.plot(*controller.actuator.position_l.T,
            lw=10, color="blue",zorder=2)
    rarm, = ax.plot(*controller.actuator.position_r.T, 
            lw=10, color="blue",zorder=2)
    larm_target, = ax.plot(*(controller.actuator.position_l.T*1e10),
            lw=5, color="black",zorder=1)
    rarm_target, = ax.plot(*(controller.actuator.position_r.T*1e10), 
            lw=5, color="black",zorder=1)
    larm_theor, = ax.plot(*controller.actuator.position_l.T, lw=4, 
            color="green",zorder=3, alpha=.5)
    rarm_theor, = ax.plot(*controller.actuator.position_r.T, lw=4, 
            color="green",zorder=3, alpha=.5)
    
    tl_lt = 2.0
    tl_start = -2.8
    tl_height = 2.5

    ts = tl_lt/float(trial_window)
    rwl = ts*gs.RESET_WINDOW
    gwl = ts*gs.GOAL_WINDOW

    tl_reset, =  ax.plot(
            [tl_start, tl_start+rwl], 
            [tl_height, tl_height],
            lw=3,  color="red",zorder=4)
    tl_goal, =  ax.plot(
            [tl_start+rwl, tl_start+tl_lt], 
            [tl_height, tl_height], 
            lw=3,  color="blue",zorder=4)
    tl_point, =  ax.plot(
            [tl_start], 
            [tl_height], 
            lw=3, marker="o", color="green",
            markersize=10, zorder=3)

    gs_lt = 1.0
    gs_start = -2.8
    gs_height = 2.0
    gs_ts = gs_lt/gs.N_GOAL_UNITS

    gs_zero, =  ax.plot(
            [gs_start, gs_start+gs_lt], 
            [gs_height,gs_height ], 
            color=[.8,.8,.8],
            lw=5, zorder=3)
    
    gs_one, =  ax.plot(
            [gs_start+(gs_ts)+9999, gs_start+(gs_ts*(1+1))+9999], 
            [gs_height, gs_height ], 
            color=[.2,.2,.2],
            lw=5, zorder=3) 

    gm_lt = 1.0
    gm_start = -2.8
    gm_height = 1.75
    gm_ts = gm_lt/gs.N_GOAL_UNITS

    gm_zero, =  ax.plot(
            [gm_start, gm_start+gm_lt], 
            [gm_height,gm_height ], 
            color=[.8,.6,.6],
            lw=5, zorder=3)
    
    gm_one, =  ax.plot(
            [gm_start+(gm_ts)+9999, gm_start+(gm_ts*(1+1))+9999], 
            [gm_height, gm_height ], 
            color=[.2,.2,.2],
            lw=5, zorder=3) 

    gmm_lt = 1.0
    gmm_start = -2.8
    gmm_height = 1.5
    gmm_ts = gmm_lt/gs.N_GOAL_UNITS

    gmm_zero, =  ax.plot(
            [gmm_start, gmm_start+gmm_lt],
            [gmm_height,gmm_height ],
            color=[.6,.8,.6],
            lw=5, zorder=3)

    gmm_one, =  ax.plot(
            [gmm_start+(gmm_ts)+9999, gmm_start+(gmm_ts*(1+1))+9999],
            [gmm_height, gmm_height ],
            color=[.2,.2,.2],
            lw=5, zorder=3)

    rew, = ax.plot(*[-99.0,-99.0], 
            marker="o", color="red", ms=20 )
    skin, = ax.plot(*[-99.0,-99.0], 
            marker="o", color="blue", ms=5 )
    ax.set_xlim([-3,3])
    ax.set_ylim([-0.5,3])

    #----------------------------------------------------------
    
    chain = KM.Polychain()

    target = (0.98, 0.78)

    touch = False
    im_value = 0.0
    match_value = False

    goal_mask = np.zeros(gs.N_GOAL_UNITS).astype("bool")

    for t in xrange(stime):


        if gs.reset_window_counter>=gs.RESET_WINDOW:

            # update the subset of goals to be selected
            goal_mask = np.logical_or(goal_mask, (gm.goalrep_layer > 0) )

            # Selection
            if any(goal_mask==True):
                gs.goal_selection(im_value, goal_mask )
            else:
                gs.goal_selection(im_value)

            # Prediction
            if gs.goal_window_counter == 0:
                gp.step(gs.goal_win) 

            static_inp *= 0.0

        else:
            if gs.reset_window_counter == 0:
                static_inp = np.random.rand(*static_inp.shape)

        # Movement

        gs.step( static_inp )

        controller.step_kinematic(
                larm_angles=np.pi*gs.out[:(gs.N_ROUT_UNITS/2)],
                rarm_angles=np.pi*gs.out[(gs.N_ROUT_UNITS/2):]
                )

        if gs.reset_window_counter>=gs.RESET_WINDOW:

            gm.step([
                controller.pos_delta.ravel()*500.0,
                controller.prop_delta.ravel()*5000.0,
                controller.touch_delta.ravel()*5000.0])

            gm.learn()

        if gs.reset_window_counter>=gs.RESET_WINDOW:
            
            # # Detect touch
            # 
            # body = np.vstack([
            #     controller.actuator.position_l[::-1],
            #     controller.actuator.position_r[1:]])
            # chain.set_chain(body)
            # 
            # finger = chain.get_point(0.0)
            # skin_region = chain.get_point(target[np.arange(len(target))[gs.goal_win][0]])
            # 
            # touch = np.linalg.norm(finger - skin_region) < 0.5
             
            # Train experts
            
            gs.learn(match_value=match_value)
            
            # update counters
            
            gs.goal_window_counter += 1
            
            # End of trial
            
            match_value = GoalPredictor.match(
                    gm.goalrep_layer, 
                    gs.goal_win
                    )
            
            if match_value ==1 or gs.goal_window_counter >= gs.GOAL_WINDOW:
                
                if match_value == 1:
                    gs.update_target()
                
                im_value = gp.prediction_error 
                
                gs.goal_selected = False
                gs.reset(match = match_value)
                controller.reset()
                
        else:
            
            skin_region = [1e10,1e10]
            gs.reset_window_counter += 1

        #----------------------------------------------------------
        # Plot
        idcs = np.squeeze(np.argwhere(gs.goal_win==True))
        if idcs.size==1  :
            gs_one.set_data(
                    [gs_start+gs_ts*(idcs),
                        gs_start+gs_ts*(idcs+1)],
                        [gs_height, gs_height ] )

        idcs = np.nonzero(gm.goalrep_layer > 0)[0]
        if idcs.size>=1  :
            gm_one.set_data(
                    [gm_start+gm_ts*(idcs),
                        gm_start+gm_ts*(idcs+1)],
                    [gm_height, gm_height ] )
        else:
            gm_one.set_data(
                    [gm_start+(gm_ts)+9999,
                        gm_start+(gm_ts*(1+1))+9999], 
                    [gm_height, gm_height ])

        idcs = np.nonzero(1*goal_mask)[0]
        if idcs.size>=1  :
            gmm_one.set_data(
                    [gmm_start+gmm_ts*(idcs),
                        gmm_start+gmm_ts*(idcs+1)],
                    [gmm_height, gmm_height ] )
        else:
            gmm_one.set_data(
                    [gm_start+(gm_ts)+9999,
                        gm_start+(gm_ts*(1+1))+9999], 
                    [gm_height, gm_height ])

      
    
        # if touch == True:
        #     rew.set_data(*skin_region)

        if gs.reset_window_counter>=gs.RESET_WINDOW:
            tl_point.set_data(tl_start+rwl + ts*gs.goal_window_counter, tl_height )
            #skin.set_data(*skin_region)
            #skin.set_markersize(10*gs.curr_noise)

            larm.set_data(*controller.actuator.position_l.T)
            rarm.set_data(*controller.actuator.position_r.T)
            try:
                goalwin_idx = gs.goal_index()
                target_angles = gs.target_position[goalwin_idx]
                controller.actuator.set_angles(
                        np.pi*target_angles[:(gs.N_ROUT_UNITS/2)],
                        np.pi*target_angles[(gs.N_ROUT_UNITS/2):],
                        )
                larm_target.set_data(*controller.actuator.position_l.T)
                rarm_target.set_data(*controller.actuator.position_r.T)
            except KeyError: pass
        else:
            tl_point.set_data(tl_start + ts*gs.reset_window_counter, tl_height )
            #skin.set_data(*((np.array(skin_region)+1)*9999))
            #skin.set_markersize(10*gs.curr_noise)
            larm.set_data(*((controller.actuator.position_l+1)*9999).T)
            rarm.set_data(*((controller.actuator.position_r+1)*9999).T)
            larm_target.set_data(*((controller.actuator.position_l+1)*9999).T)
            rarm_target.set_data(*((controller.actuator.position_r+1)*9999).T)
        
        controller.actuator.set_angles(
                np.pi*gs.tout[:(gs.N_ROUT_UNITS/2)],
                np.pi*gs.tout[(gs.N_ROUT_UNITS/2):]
            )
        larm_theor.set_data(*controller.actuator.position_l.T)
        rarm_theor.set_data(*controller.actuator.position_r.T)

        plt.pause(.1)

        #----------------------------------------------------------

