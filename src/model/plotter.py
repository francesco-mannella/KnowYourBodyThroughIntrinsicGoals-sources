#!/usr/bin/python
# -*- coding: utf-8 -*-


import os
import sys
import copy
sys.path.append("../")
import numpy as np
from Robot import Robot

np.set_printoptions(edgeitems=3, linewidth=999,  precision=3,
        suppress=True, threshold=1000)

#################################################################
#################################################################

import pyqtgraph.Qt as Qt 
from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg
from pyqtgraph.ptime import time
pg.setConfigOptions(antialias=True)

#################################################################
#################################################################

def reshape_weights(w):
    reshaped_w = []
    reshaped_w_raw = []
    single_w_raws = int(np.sqrt(len(w[0])))
    single_w_cols = single_w_raws

    n_single_w = len(w)
    out_raws = np.sqrt(n_single_w)
    for single_w, i in zip(w, xrange(n_single_w)):
        reshaped_w_raw.append(
            single_w.reshape(single_w_cols,
                single_w_raws, order="F"))
        if (i+1)%out_raws == 0:
            reshaped_w.append(np.hstack(reshaped_w_raw))
            reshaped_w_raw =[]
    reshaped_w = np.vstack(reshaped_w)

    return reshaped_w

def reshape_coupled_weights(w):
    '''
    Reshape the matrix of weight from two (nxn) input layers
    to one (mxm) output layers so that it becomes a block matrix
    made by  mxm  nx2n matrices
    '''

    reshaped_w = []
    reshaped_w_raw = []
    single_w_raws = int(np.sqrt(len(w[0])/2))
    single_w_cols = 2*single_w_raws

    n_single_w = len(w)
    out_raws = np.sqrt(n_single_w)
    for single_w, i in zip(w, xrange(n_single_w)):
        reshaped_w_raw.append(
            single_w.reshape(single_w_cols,
                single_w_raws, order="F"))
        if (i+1)%out_raws == 0:
            reshaped_w.append(np.hstack(reshaped_w_raw))
            reshaped_w_raw =[]
    reshaped_w = np.vstack(reshaped_w)

    return reshaped_w

#################################################################
#################################################################

class Plotter(QtGui.QWidget):

    def __init__(self, app, amaps, smaps, kin, robot):
        
        super(Plotter, self).__init__()
        
        self.amaps = amaps
        self.smaps = smaps
        self.kin = kin
        
        screen_resolution = app.desktop().screenGeometry()
        width, height = screen_resolution.width(), screen_resolution.height()
            
        self.LEFT = height*(2/20.)
        self.TOP = height*(2/20.)
        self.WIDTH = height*(6/20.)
        self.HEIGHT = height*(6/20.)
        
        self.setGeometry(self.LEFT, self.TOP, self.WIDTH, self.HEIGHT) 
	self.setWindowTitle("MainBoard")
        
        self.amapsButton = QtGui.QCheckBox('Abstraction Maps', self)
	self.amapsButton.clicked.connect(self.onAMapsButton)
	self.amapsButton.move(20,20)
        self.amaps_toggle = False;
        
        self.smapsButton = QtGui.QCheckBox('Selection Maps', self)
	self.smapsButton.clicked.connect(self.onSMapsButton)
	self.smapsButton.move(20,50)
        self.smaps_toggle = False;
        
        self.kinButton = QtGui.QCheckBox('Kinematics', self)
	self.kinButton.clicked.connect(self.onKinButton)
	self.kinButton.move(20,80)
        self.kin_toggle = False;

        self.simButton = QtGui.QPushButton('Run', self)
        self.simButton.clicked.connect(self.onSimButton)
        self.simButton.setAutoDefault(False)
        self.simButton.setFlat(False)
        self.simButton.move(20, 110)
        self.sim_toggle = False;

        self.slider = QtGui.QSlider(QtCore.Qt.Horizontal, self)
        self.slider.setFocusPolicy(QtCore.Qt.NoFocus)
        self.slider.setGeometry(20, 160, 120, 10)
        self.slider.valueChanged[int].connect(self.changeValue)
        self.label = QtGui.QLabel(self)
        self.label.setGeometry(150, 160, 60, 10)
        self.label.setText("speed")
        
        self.robot = robot
        
        self.max_duration = 1000
        self.min_duration = 0.0001
        self.interval = (self.max_duration  - self.min_duration)/100.0
        self.ts_duration = 1
        self.slider.setValue( int( (self.ts_duration - self.max_duration)/self.interval) )
        self.show()
        
    def onAMapsButton(self) :
            
        self.amaps_toggle = not self.amaps_toggle
        if self.amaps_toggle :
            self.amaps.show()
            self.amaps.start()
        else:
            self.amaps.stop()
            self.amaps.hide()
    
    def onSMapsButton(self) :
        self.smaps_toggle = not self.smaps_toggle
        if self.smaps_toggle :
            self.smaps.show()
            self.smaps.start()
        else:
            self.smaps.stop()
            self.smaps.hide()
    

    def onKinButton(self) :
            
        self.kin_toggle = not self.kin_toggle
        if self.kin_toggle :
            self.kin.start()
            self.kin.show()
        else:
            self.kin.hide()   
 
    def onSimButton(self) :
            
        self.sim_toggle = not self.sim_toggle
        if self.sim_toggle == True:
            self.time_id = self.startTimer(self.ts_duration)
            self.simButton.setFlat(True)
            self.simButton.setText("stop")
        else:
            self.killTimer(self.time_id)
            self.simButton.setFlat(False)
            self.simButton.setText("Run")


    def closeEvent(self, event):

        QtGui.QApplication.quit()
    
    def timerEvent(self, event):

        self.robot.step()

    def changeValue(self, value):
    

        self.ts_duration = value*self.interval
        if self.sim_toggle == True:
            try:
                self.killTimer(self.time_id)
                self.time_id = self.startTimer(self.ts_duration)
            except AttributeError:
                pass

            
#################################################################
#################################################################

class GoalAbstractionMaps(pg.GraphicsView):
    
    pos = np.array([0.0, 1.0])
    color = np.array([[255,255,255,255], [0,0,0,255]], 
            dtype=np.ubyte)
    cmap = pg.ColorMap(pos, color)
    lut = cmap.getLookupTable(0.0, 1.0, 256)
        
    def __init__(self, app, robot):
         
	super(GoalAbstractionMaps, self).__init__()
        self.robot = robot
        self.ts_duration = 1 
        self.main_app = app

	self.setWindowTitle("Maps")
        
        self.resetWindow()

        self.layout = pg.GraphicsLayout(
                border=pg.mkPen({'color':'300','width':2}))
        self.setCentralItem(self.layout) 
        self.views = []
        self.imgs = []

        
        if self.robot.gm.SINGLE_KOHONEN :
            self.append_views(items=1,row=0,col=0)
            self.append_views(items=1,row=1,col=0)
            self.append_views(items=1,row=2,col=0)
        else:
            self.append_views(items=1,row=0,col=0)
            self.append_views(items=1,row=0,col=1)
            self.append_views(items=1,row=0,col=2)
            self.append_views(items=1,row=1,col=0)
            self.append_views(items=1,row=1,col=1)
            self.append_views(items=1,row=1,col=2)
            self.append_views(items=1,row=2,col=0)
            self.append_views(items=1,row=2,col=1)
            self.append_views(items=1,row=2,col=2)
            self.append_views(items=1,row=3,col=0)
            self.append_views(items=1,row=3,col=1)
            self.append_views(items=1,row=4,col=0)
            self.append_views(items=1,row=4,col=1)
            self.append_views(items=1,row=5,col=0)
            self.append_views(items=1,row=6,col=0)
            self.append_views(items=1,row=7,col=0)
            self.append_views(items=1,row=7,col=1)
            self.append_views(items=1,row=7,col=2)
            self.append_views(items=1,row=8,col=0)


    def resetWindow(self):

        screen_resolution = self.main_app.desktop().screenGeometry()
        width, height = screen_resolution.width(), screen_resolution.height()
         
        self.LEFT = height*(8/20.)
        self.TOP = height*(1.2/20.)
        self.WIDTH = height*(6/20.)
        self.HEIGHT = height*(16/20.)
        
        self.setGeometry(self.LEFT, self.TOP, self.WIDTH, self.HEIGHT)



    def append_views(self, items=1, row=None, col=None, 
            rowspan=1, colspan=1) :
        for x in xrange(items) :
            (view, img) = self.add_img(self.layout, row=row, 
                    col=col, rowspan=rowspan, colspan=colspan)
            self.views.append(view)
            self.imgs.append(img)


    def add_img(self, win, row=None, col=None, 
            rowspan=1, colspan=1) :
        
        ## Add viewbox
        view = win.addViewBox(row=row, col=col, 
                rowspan=rowspan, colspan=colspan)
        view.enableAutoRange()
        
        ## Create image item
        img = pg.ImageItem(border='w',lut=self.lut)
        view.addItem(img)

        return (view, img)

    def timerEvent(self, event):
                
        (i1, i2, i3, sm1, sm2, sm3, h1, 
            h2, o1, wsm1, wsm2, wsm3, wh1, 
            wh2, wo1, wgr1, gr1) = self.robot.get_sensory_arrays()

        if np.hstack((np.isnan(wh1), np.isnan(wh2))).any() :
            self.stop()

        wsmm1 = reshape_weights(wsm1)
        wsmm2 = reshape_weights(wsm2)
        wsmm3 = reshape_weights(wsm3)

        raw_sm1 = int(np.sqrt(len(sm1)))
        sm1 = sm1.reshape(raw_sm1,raw_sm1, order="F")
        sm2 = sm2.reshape(raw_sm1,raw_sm1, order="F")
        sm3 = sm3.reshape(raw_sm1,raw_sm1, order="F")

        whh1 = reshape_coupled_weights(wh1)
        whh2 = reshape_coupled_weights(wh2)
        woo1 = reshape_coupled_weights(wo1)

        raw_h1 = int(np.sqrt(len(h1)))
        h1 = h1.reshape(raw_h1,raw_h1, order="F")
        h2 = h2.reshape(raw_h1,raw_h1, order="F")

        raw_inp = int(np.sqrt(len(i1)))
        i1 = i1.reshape(raw_inp,raw_inp, order="F")
        i2 = i2.reshape(raw_inp,raw_inp, order="F")
        i3 = i3.reshape(raw_inp,raw_inp, order="F")

        raw_out = int(np.sqrt(len(o1)))
        o1 = o1.reshape(raw_out, raw_out)
        
        sm_l = len(wsm1)

        h_l = len(h1)
        o_l = len(o1)
        
        if self.robot.gm.SINGLE_KOHONEN :
            wi3_gr =reshape_weights(wgr1)

        else:
            wsm1_gr = wgr1[:,(0*sm_l):(1*sm_l)]
            wsm2_gr = wgr1[:,(1*sm_l):(2*sm_l)]
            wsm3_gr = wgr1[:,(2*sm_l):(3*sm_l)]

            wsm_gr = np.hstack([
                reshape_weights(wsm1_gr),
                reshape_weights(wsm2_gr),
                reshape_weights(wsm3_gr)
            ])
            
            wh1_gr = wgr1[:,((3*sm_l) + 0*h_l):((3*sm_l) + 1*h_l)]
            wh2_gr = wgr1[:,((3*sm_l) + 1*h_l):((3*sm_l) + 2*h_l)]

            wh_gr = np.hstack([
                reshape_weights(wh1_gr),
                reshape_weights(wh2_gr)
            ])

            wo_gr = wgr1[:,(3*sm_l+3*h_l):((3*sm_l+3*h_l) + o_l)]
            reshape_weights(wo_gr)
        
        raw_gr1 = int(np.sqrt(len(gr1)))
        gr1 = gr1.reshape(raw_gr1,raw_gr1, order="F")



        if self.robot.gm.SINGLE_KOHONEN :
            self.imgs[0].setImage( i3 , levels=(-1,1) )
            self.imgs[1].setImage( wi3_gr , levels=(-1,1) )
            self.imgs[2].setImage( gr1 , levels=(0,1) )
        else:

            self.imgs[0].setImage( i1 , levels=(-1,1) )
            self.imgs[1].setImage( i2 , levels=(-1,1) )
            self.imgs[2].setImage( i3 , levels=(-1,1) )

            self.imgs[3].setImage( wsmm1 , levels=(-.2,.2) )
            self.imgs[4].setImage( wsmm2 , levels=(-.2,.2) )
            self.imgs[5].setImage( wsmm3 , levels=(-.2,.2) )

            self.imgs[6].setImage( sm1 , levels=(0,1) )
            self.imgs[7].setImage( sm2 , levels=(0,1) )
            self.imgs[8].setImage( sm3 , levels=(0,1) )

            self.imgs[9].setImage( whh1 , levels=(-1,1) )
            self.imgs[10].setImage( whh2 , levels=(-1,1) )

            self.imgs[11].setImage( h1 , levels=(0,1) )
            self.imgs[12].setImage( h2 , levels=(0,1) )

            self.imgs[13].setImage( woo1 , levels=(-1,1) )

            self.imgs[14].setImage( o1 , levels=(0,1) )

            self.imgs[15].setImage( wsm_gr , levels=(-1,1) )
            self.imgs[16].setImage( wh_gr , levels=(-1,1) )
            self.imgs[17].setImage( wo_gr , levels=(-1,1) )

            self.imgs[16].setImage( gr1, levels=(0,1) )
        
        self.update()

    def start(self) :
        self.resetWindow()
        self.time_id = self.startTimer(self.ts_duration)

    def stop(self) :
        self.killTimer(self.time_id)


#################################################################
#################################################################

class GoalSelectionMaps(pg.GraphicsView):
    
    pos = np.array([0.0, 1.0])
    color = np.array([[255,255,255,255], [0,0,0,255]], 
            dtype=np.ubyte)
    cmap = pg.ColorMap(pos, color)
    lut = cmap.getLookupTable(0.0, 1.0, 256)
        
    def __init__(self, app, robot):
         
	super(GoalSelectionMaps, self).__init__()
        self.main_app = app
        
        self.resetWindow()

	self.setWindowTitle("Selection and control")
               
        layout = pg.GraphicsLayout()
        self.setCentralItem(layout)  

 
        view = layout.addViewBox(row=0,col=0,colspan=1)
        self.plot1 = pg.ImageItem(border='w',lut=self.lut)
        view.addItem(self.plot1)
        view.setAspectLocked(lock=True, ratio=1)
        
        view = layout.addViewBox(row=0,col=1,colspan=1)
        self.plot3 = pg.ImageItem(border='w',lut=self.lut)
        view.addItem(self.plot3)
        view.setAspectLocked(lock=True, ratio=1)
        
   
        view = layout.addViewBox(row=0,col=2,colspan=1)   
        self.plot4 = pg.ImageItem(border='w',lut=self.lut)
        self.plot6 = pg.ScatterPlotItem()
        view.addItem(self.plot4)
        view.addItem(self.plot6)
        view.setAspectLocked(lock=True, ratio=1)       
        
        self.plot5view = layout.addViewBox(row=0,col=3, colspan=2)
        self.plot5 = pg.GraphItem()
        self.plot5view.addItem(self.plot5)
 
        self.plot2 = layout.addPlot(row=2,col=0, rowspan=3, colspan=5) 
        view = layout.addViewBox(row=4,col=2)   

        self.robot = robot
       
        self.curves = []  
        for g in range(self.robot.gs.N_ECHO_UNITS):
            r = np.random.rand()*255
            g = np.random.rand()*255
            b = np.random.rand()*255
            self.curves.append(self.plot2.plot(pen=(r,g,b)))

      
        self.ts_duration = 1

    def resetWindow(self):
        
        screen_resolution = self.main_app.desktop().screenGeometry()
        width, height = screen_resolution.width(), screen_resolution.height()
         
        self.LEFT = height*(14/20.)
        self.GAP = height*(1.5/20.)
        self.WIDTH = 10*self.GAP
        self.HEIGHT =  height*(16/20.) - 6.35*self.GAP
        self.TOP =  height*(1.2/20.)+ 6.35*self.GAP
        
        self.setGeometry(self.LEFT, self.TOP, self.WIDTH, self.HEIGHT)

    def timerEvent(self, event):
        
        (gmask, gv, gw, gr, matched,  targets, 
                target_position, 
                esn_data ) = self.robot.get_selection_arrays() 
    
        goals = len(gw)
        raw_gw = int(np.sqrt(goals))
        self.plot1.setImage( 0.6*gw.reshape(raw_gw, raw_gw) 
            +0.4*gr.reshape(raw_gw, raw_gw) , levels=(0,1) )
        self.plot3.setImage( (gmask).reshape(raw_gw, raw_gw), levels=(0,1) ) 
        
        targets = (targets).reshape(raw_gw, raw_gw)
        self.plot4.setImage( targets, levels=(0,1) ) 
        points = np.vstack(np.where(targets>0)) +0.5
        self.plot6.setData(*points)

        gr  = self.plot5
        lims = self.robot.controller.perc.lims
        rngs = lims[:,1]-lims[:,0]
        mins = lims[:,0]

        self.plot5view.setXRange(0,raw_gw)
        self.plot5view.setYRange(0,raw_gw)
        act = copy.deepcopy(self.robot.controller.actuator) 
        goal_idcs = np.arange(goals).reshape(raw_gw,raw_gw)
        
        ps = []
        for goal in target_position.keys() :
             row, col = np.squeeze(np.nonzero(goal_idcs==goal))
             trj_angles = target_position[goal]
             larm_angles = np.pi*trj_angles[:(self.robot.gs.N_ROUT_UNITS/2)]
             larm_angles = larm_angles[::-1]
             rarm_angles = np.pi*trj_angles[(self.robot.gs.N_ROUT_UNITS/2):]
  
             act.set_angles(larm_angles, rarm_angles)
             lp = (act.position_l-mins)/rngs + [row, col] 
             rp = (act.position_r-mins)/rngs + [row, col] 
             ps.append(lp)
             ps.append(rp)
       
        pos=None
        adj=None
        if len(ps) <= 0: 
            for row in range(raw_gw):
                for col in range(raw_gw):
                    ps.append(np.array([0.5,0.5])+[row,col])
            pos = np.vstack(ps)
            idcs = np.arange(len(pos))
            adj = np.vstack([ [idcs[i], idcs[i]] for i in range(len(pos)) ])

        else:  
            pos = np.vstack(ps)
            lj = len(act.position_l)
            idcs = np.arange(len(pos))
            adj = np.vstack([ [idcs[i], idcs[i+1]] for i in idcs[idcs%lj!=(lj-1)] ])
          
        gr.setData(pos=pos, adj=adj,  size=2 )

        # if self.robot.gs.goal_window_counter > self.robot.gs.GOAL_WINDOW*(98/100.) :
        for g in range(self.robot.gs.N_ECHO_UNITS):
            self.curves[g].setData(esn_data[g])
        
        self.update()

    def start(self) :
        self.resetWindow()
        self.time_id = self.startTimer(self.ts_duration)

    def stop(self) :
        self.killTimer(self.time_id)



#################################################################
#################################################################

class KinematicsView(QtGui.QWidget):
    
    def __init__(self, app, robot):
	
        super(KinematicsView, self).__init__()
        
        self.main_app = app
       
        self.resetWindow()

        self.LINE_WIDTH = 0.04
        self.AXIS_LINE_WIDTH = 0.01
        self.DOT_RADIUS = 0.06
        self.STIME = robot.stime
        self.robot = robot

	self.setWindowTitle("Kinematics")

        self.ts_duration = 10
        self.time_id = self.startTimer(self.ts_duration)
    
    def resetWindow(self):
        
        screen_resolution = self.main_app.desktop().screenGeometry()
        width, height = screen_resolution.width(), screen_resolution.height()
         
        self.LEFT = height*(14/20.)
        self.TOP = height*(1.2/20.)
        self.GAP = height*(1.5/20.)
        
        
        self.WINDOW_BOTTOM = -2.0 
        self.WINDOW_LEFT = -5.0
        self.WINDOW_WIDTH = 10.0
        self.WINDOW_HEIGHT = 6.0
        
        self.WINDOW_PX_WIDTH = self.WINDOW_WIDTH*self.GAP 
        self.WINDOW_PX_HEIGHT = self.WINDOW_HEIGHT*self.GAP
        self.setGeometry(self.LEFT, self.TOP, self.WINDOW_PX_WIDTH, self.WINDOW_PX_HEIGHT)



    def poly(self, pts):
        
        return QtGui.QPolygonF([QtCore.QPointF(p[0], p[1]) for p in pts]) 
 
    def paintEvent(self, event):
       
        painter =  QtGui.QPainter(self)
        w = self.contentsRect().width() 
        h = self.contentsRect().height() 
      
        # rescale window
        xr = float(w)  /self.WINDOW_WIDTH
        yr = float(h) /self.WINDOW_HEIGHT
        painter.scale(xr, -yr); 
        painter.translate(
                 (self.WINDOW_WIDTH +  self.WINDOW_LEFT),
                 -(self.WINDOW_HEIGHT + self.WINDOW_BOTTOM) ) 
    
        
        # with background and antialiasing
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        painter.setBrush(QtCore.Qt.white)
        painter.drawRect(
                self.WINDOW_LEFT, 
                self.WINDOW_BOTTOM,
                self.WINDOW_WIDTH, 
                self.WINDOW_HEIGHT)  

        (real_l_pos, real_r_pos, target_l_pos,
                target_r_pos, theor_l_pos, 
                theor_r_pos, sensors ) = self.robot.get_arm_positions()
        
        touches = self.robot.controller.touches
         
        # paint axes
        painter.setPen(QtGui.QPen(QtGui.QColor(QtCore.Qt.black), 
            self.AXIS_LINE_WIDTH, QtCore.Qt.DashLine)) 
        painter.drawLine( self.WINDOW_LEFT, 0, 
                self.WINDOW_LEFT+self.WINDOW_WIDTH,0) 
        painter.drawLine( 0, self.WINDOW_BOTTOM, 
                0, self.WINDOW_BOTTOM+self.WINDOW_HEIGHT)  


        def paint_arm(curr_pos, curr_color, curr_width):
            painter.setPen( QtGui.QPen( curr_color, self.LINE_WIDTH*curr_width) ) 
            painter.setBrush(curr_color) 
            painter.drawPolyline(self.poly(curr_pos)) 
            for x,y in zip(*curr_pos.T) : 
                painter.drawEllipse(QtCore.QRectF(
                    x - self.DOT_RADIUS*curr_width, 
                    y - self.DOT_RADIUS*curr_width, 
                    self.DOT_RADIUS*2*curr_width, 
                    self.DOT_RADIUS*2*curr_width)) 


        # paint arm
        
        curr_color = QtGui.QColor(255,100,100)
        curr_width = 3.0

        curr_pos = target_l_pos
        paint_arm(curr_pos, curr_color, curr_width)
        
        curr_pos = target_r_pos
        paint_arm(curr_pos, curr_color, curr_width)
  
        curr_color = QtGui.QColor(100,255,100)
        curr_width = 2.0

        curr_pos = theor_l_pos
        paint_arm(curr_pos, curr_color, curr_width)
        
        curr_pos = theor_r_pos
        paint_arm(curr_pos, curr_color, curr_width)
 
        curr_color = QtCore.Qt.black
        curr_width = 1

        curr_pos = real_l_pos
        paint_arm(curr_pos, curr_color, curr_width)
           
        curr_pos = real_r_pos
        paint_arm(curr_pos, curr_color, curr_width)
        
        
        if self.robot.gs.goal_selected :
            curr_pos = np.vstack([theor_l_pos[0], real_r_pos[0]])
            paint_arm(curr_pos, curr_color, curr_width)
               
        # paint sensors
        for i,sensor in zip(range(len(sensors)),sensors) :
        
            if touches[i] > 0 :
                curr_color = QtCore.Qt.red
            else:
                curr_color = QtCore.Qt.blue
 
            painter.setPen( QtGui.QPen( curr_color, self.LINE_WIDTH*curr_width) ) 
            painter.setBrush(curr_color)  
            curr_width = 4*np.abs(touches[i]+0.01)
            x,y  = sensor 
            painter.drawEllipse(QtCore.QRectF(
                x - self.DOT_RADIUS*curr_width, 
                y - self.DOT_RADIUS*curr_width, 
                self.DOT_RADIUS*2*curr_width, 
                self.DOT_RADIUS*2*curr_width)) 
        
    def timerEvent(self,event):
        
        self.update()
    
    def start(self) :
        self.resetWindow()

#################################################################
#################################################################

def graph_main(robot) :
    
    main_app = QtGui.QApplication([])
    abstraction_maps = GoalAbstractionMaps(main_app, robot)
    selection_maps = GoalSelectionMaps(main_app, robot)
    kin = KinematicsView(main_app, robot)
    main = Plotter(main_app, abstraction_maps, selection_maps, kin, robot)

    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()


