#!/usr/bin/env python
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

from PySide.QtCore import *
from PySide.QtGui import *
import numpy as np

import sys, time
import simulator as SM

# Graphical parameters - to be change here only

# scaled x coordinate of window starting point
WINDOW_LEFT = -10    
# scaled y coord of window starting point
WINDOW_BOTTOM = -5    
# scaled width of window
WINDOW_WIDTH = 20    
# scaled height of window
WINDOW_HEIGHT = 10    
# axis line width
AXIS_LINE_WIDTH = 0.005    
# segment line width
LINE_WIDTH = 0.03    
# radius of dots
DOT_RADIUS = 0.05    
# time duration of each timestep (msec)
TS_DURATION = 100    
# real width of window (pixels)
WINDOW_PX_WIDTH = 1000    
# real height of window (pixels)
WINDOW_PX_HEIGHT = 500    
# simulation time
STIME = 200000   

class Sim(QWidget):
    """

    Open a window and start the simulation.

    the self.simulator.learner object controls the movment of the arm
    and the position of the buzzer.

    Yellow dots show the sensory response of the torso to the buzzer,
    green dots show the sensory response of the torso to the hand
    (final edge of the arm).

    """
    def __init__(self, parent = None):
        
        QWidget.__init__(self, parent)
            
        self.window_bottom = WINDOW_BOTTOM
        self.window_left = WINDOW_LEFT
        self.window_width = WINDOW_WIDTH
        self.window_height = WINDOW_HEIGHT
        self.line_width = LINE_WIDTH
        self.axis_line_width = AXIS_LINE_WIDTH
        self.dot_radius = DOT_RADIUS
        self.ts_duration = TS_DURATION
        self.window_px_width = WINDOW_PX_WIDTH
        self.window_px_height = WINDOW_PX_HEIGHT
        self.STIME = STIME

        self.simulator = SM.Simulator()
 
        self.resize( self.window_px_width, self.window_px_height)

        # set ts_duration and timer
        self.t = 0
        self.time_id = self.startTimer(self.ts_duration)

        # flag for simulation running state
        self.go_toggle = True
        
        self.mem_toggle = True
        self.curr_toggle = True
        self.theoric_toggle = True
        self.rew_toggle = True

    def poly(self, pts):
        
        return QPolygonF([QPointF(p[0], p[1]) for p in pts]) 
 
    def paintEvent(self, event):
       
        painter =  QPainter(self)
        w = self.contentsRect().width() 
        h = self.contentsRect().height() 

        # rescale window
        painter.setWindow(self.window_left, self.window_bottom, 
                self.window_width, self.window_height) 
        painter.setRenderHint(QPainter.Antialiasing)
        painter.scale(1, -1); 

        # with background and antialiasing
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setBrush(Qt.white)
        painter.drawRect(painter.window())
    
        # paint axes
        painter.setPen(QPen(QColor(Qt.black), self.axis_line_width, Qt.DashLine)) 
        painter.drawLine(self.window_left, 0, self.window_left+self.window_width, 0) 
        painter.drawLine(0, self.window_bottom, 0, self.window_bottom+self.window_height) 
      
        # paint arm sensor memories 
        if self.mem_toggle :
            curr_green = QColor(0, 100, 0, 50)
            painter.setPen(QPen( curr_green, self.line_width) ) 
            painter.setBrush(curr_green) 
            curr_pos = self.simulator.controller.update_sensor_poses()
            X,Y = zip(*curr_pos)
            RAD = 0.5*(1.0 - self.simulator.predictor.memory)
            for x,y,rad  in zip(X, Y, RAD)  : 
                painter.drawEllipse(QRectF(x - rad, y - rad, rad*2, rad*2)) 
         
        # paint arm  
        if self.curr_toggle :
            painter.setPen( QPen( Qt.black, self.line_width*2) ) 
            painter.setBrush(Qt.black) 
            curr_pos = self.simulator.controller.get_arms_positions(SM.Controller.ACTUAL)
            painter.drawPolyline(self.poly(curr_pos)) 
            for x,y in zip(*curr_pos.T) : 
                painter.drawEllipse(QRectF(x - self.dot_radius, y - self.dot_radius, self.dot_radius*2, self.dot_radius*2)) 
        
        if self.rew_toggle and self.simulator.goal_switch :
            painter.setPen( QPen( QColor(255, 10, 10), self.line_width*2) ) 
            painter.setBrush(Qt.black) 
            curr_pos = self.simulator.controller.get_arms_positions(SM.Controller.DESIRED)
            painter.drawPolyline(self.poly(curr_pos)) 
            for x,y in zip(*curr_pos.T) : 
                painter.drawEllipse(QRectF(x - self.dot_radius, y - self.dot_radius, self.dot_radius*2, self.dot_radius*2)) 
                    
        if self.theoric_toggle :
            painter.setPen( QPen( QColor(0, 0, 255), self.line_width*2) ) 
            painter.setBrush(Qt.black) 
            curr_pos = self.simulator.controller.get_arms_positions(SM.Controller.COMMANDED)
            painter.drawPolyline(self.poly(curr_pos)) 
            for x,y in zip(*curr_pos.T) : 
                painter.drawEllipse(QRectF(x - self.dot_radius, y - self.dot_radius, self.dot_radius*2, self.dot_radius*2)) 
       
        # paint reward 
        if self.simulator.goal_switch :
            (x,y) = self.simulator.controller.touch_point
            rad = self.line_width*10 
            painter.setPen(QColor(200, 0, 0,50))
            painter.setBrush(QColor(200, 0, 0,50))  
            painter.drawEllipse(QRectF( x - rad, y - rad, 2*rad, 2*rad,)) 
            rad = self.line_width*3 
            painter.setPen(QColor(100, 0, 0))
            painter.setBrush(QColor(100, 0, 0))  
            painter.drawEllipse(QRectF( x - rad, y - rad, 2*rad, 2*rad,))             
  
        # paint current-trial story of rewards
        painter.setPen(Qt.red)
        painter.setBrush(Qt.red)
        x = np.linspace(-4,-1,self.simulator.TRIAL)
        x = np.hstack([x, x[::-1]])
        y = np.hstack([-3.0*np.ones(len(x))])
        painter.drawPolyline(self.poly(np.vstack([x,y]).T)) 

        painter.setPen(Qt.black)
        painter.setBrush(Qt.black)   
        x = x[self.simulator.trial_timestep]
        y = y[self.simulator.trial_timestep]
        rad = self.line_width*3
        painter.drawEllipse(QRectF(x - rad, y - rad, 2*rad, 2*rad,)) 
  

        # paint current time within the simulation
        painter.setPen(Qt.red)
        painter.setBrush(Qt.red)     
        xx = np.linspace(-4,-1,self.STIME+1)
        x = np.linspace(-4,-1, 4)
        y = -4.0*np.ones(4)
        painter.drawPolyline(self.poly(np.vstack([x,y]).T)) 
    
        painter.setPen(Qt.black)
        painter.setBrush(Qt.black)    
        x = xx[self.t]
        y = -4
        rad = self.line_width*2
        painter.drawEllipse(QRectF(x - rad, y - rad, 2*rad, 2*rad,)) 
        
        painter.setPen(Qt.white)
        painter.setBrush(Qt.white)    
        x = xx[self.t]
        y = -4
        rad = self.line_width
        painter.drawEllipse(QRectF(x - rad, y - rad, 2*rad, 2*rad,)) 
           
    def keyPressEvent(self, e):    
        """
        Up     accelerates the simulation
        Down   slowers the  simulation
        ESC    toggles stop-restart

        """
        if e.key() == Qt.Key_Up:
            self.killTimer(self.time_id)
            self.ts_duration -= 20
            if self.ts_duration < 1 :
                self.ts_duration = 1
            self.time_id = self.startTimer(self.ts_duration)
        
        elif e.key() == Qt.Key_Down:
            self.killTimer(self.time_id)
            self.ts_duration += 20  
            self.time_id = self.startTimer(self.ts_duration)
        
        elif e.key() == Qt.Key_Q:
            reply = QMessageBox.question(self, 'Message',
                    "Are you sure to quit?", QMessageBox.Yes | 
                    QMessageBox.No, QMessageBox.No) 
            if reply == QMessageBox.Yes:
                self.close()
 
        elif e.key() == Qt.Key_C:
            self.curr_toggle = not self.curr_toggle
        
        elif e.key() == Qt.Key_R:
            self.rew_toggle = not self.rew_toggle
        
        elif e.key() == Qt.Key_T:
            self.theoric_toggle = not self.theoric_toggle
       
        elif e.key() == Qt.Key_M:
            self.mem_toggle = not self.mem_toggle
                
        elif e.key() == Qt.Key_Escape:
            if self.go_toggle == True :
                self.killTimer(self.time_id)
            else: 
                self.time_id = self.startTimer(self.ts_duration)
            self.go_toggle = not self.go_toggle
           
    def closeEvent(self, event):
        """
        Ask confirmation and close simulation 
        """

        reply = QMessageBox.question(self, 'Message',
            "Are you sure to quit?", QMessageBox.Yes | 
            QMessageBox.No, QMessageBox.No)

        if reply == QMessageBox.Yes:
            event.accept()    
            self.close()
        else:
            event.ignore()  

    def timerEvent(self,event):
        """
            
        run a simulation step at each timer event 
        until self.simulator.learner.reservoir.STIME is reached

        """
        
        if event.timerId() == self.time_id:
            if self.t < self.STIME :
                self.step()
                self.update()
            else :
                self.killTimer(self.time_id)

    def close(self) :
        """
        
        close application

        """

        self.killTimer(self.time_id)
        QApplication.quit()
        sys.exit(0)

    def step(self):
        """
        
        simulation step and timestep update

        """
        self.simulator.step(self.t)

        self.t += 1

def main() :
    app = QApplication(sys.argv)
    
    sim = Sim()
    sim.show()
    
    sys.exit(app.exec_())
    


