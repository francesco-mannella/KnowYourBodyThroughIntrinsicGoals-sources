#!/usr/bin/python
# -*- coding: utf-8 -*-



#################################################################
#################################################################

from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg
from pyqtgraph.ptime import time
import numpy as np

import GoalMaker as gm

#################################################################
#################################################################


class Main:
    
    pos = np.array([0.0, 1.0])
    color = np.array([[255,255,255,255], [0,0,0,255]], dtype=np.ubyte)
    cmap = pg.ColorMap(pos, color)
    lut = cmap.getLookupTable(0.0, 1.0, 256)
        
    def __init__(self):
        
        self.app = QtGui.QApplication([])
        self.timer = QtCore.QTimer()
        self.win = pg.GraphicsView()
        self.layout = pg.GraphicsLayout(border=pg.mkPen({'color':'300','width':2}))
        self.win.setCentralItem(self.layout)

        self.views = []
        self.imgs = []

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

        self.tester = gm.GoalTesterSim()
        self.t = 0

        self.win.resize(950,950)
        self.win.show()

    def append_views(self, items=1, row=None, col=None, rowspan=1, colspan=1) :
        for x in xrange(items) :
            (view, img) = self.add_img(self.layout, row=row, col=col, rowspan=rowspan, colspan=colspan)
            self.views.append(view)
            self.imgs.append(img)


    def add_img(self,win, row=None, col=None, rowspan=1, colspan=1) :
        
        ## Add viewbox
        view = win.addViewBox(row=row, col=col, rowspan=rowspan, colspan=colspan)
        view.enableAutoRange()
        ## Create image item
        img = pg.ImageItem(border='w',lut=self.lut)
        view.addItem(img)

        return (view, img)

    def update(self):

            i1, i2, i3, sm1, sm2, sm3, h1, h2, o1, wsm1, wsm2, wsm3, wh1, wh2, wo1, wgr1, gr1 = self.tester.step(self.t)

            if np.hstack((np.isnan(wh1), np.isnan(wh2))).any() :
                self.stop()


            def reshape_weights(w):
                reshaped_w = []
                reshaped_w_raw = []
                single_w_raws = np.sqrt(len(w[0]))
                single_w_cols = single_w_raws

                n_single_w = len(w)
                out_raws = np.sqrt(n_single_w)
                for single_w, i in zip(w, xrange(n_single_w)):
                    reshaped_w_raw.append(
                        single_w.reshape(single_w_cols,single_w_raws, order="F"))
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
                single_w_raws = np.sqrt(len(w[0])/2)
                single_w_cols = 2*single_w_raws

                n_single_w = len(w)
                out_raws = np.sqrt(n_single_w)
                for single_w, i in zip(w, xrange(n_single_w)):
                    reshaped_w_raw.append(
                        single_w.reshape(single_w_cols,single_w_raws, order="F"))
                    if (i+1)%out_raws == 0:
                        reshaped_w.append(np.hstack(reshaped_w_raw))
                        reshaped_w_raw =[]
                reshaped_w = np.vstack(reshaped_w)

                return reshaped_w

            wsmm1 = reshape_weights(wsm1)
            wsmm2 = reshape_weights(wsm2)
            wsmm3 = reshape_weights(wsm3)

            raw_sm1 = np.sqrt(len(sm1))
            sm1 = sm1.reshape(raw_sm1,raw_sm1, order="F")
            sm2 = sm2.reshape(raw_sm1,raw_sm1, order="F")
            sm3 = sm3.reshape(raw_sm1,raw_sm1, order="F")

            whh1 = reshape_coupled_weights(wh1)
            whh2 = reshape_coupled_weights(wh2)
            woo1 = reshape_coupled_weights(wo1)

            raw_h1 = np.sqrt(len(h1))
            h1 = h1.reshape(raw_h1,raw_h1, order="F")
            h2 = h2.reshape(raw_h1,raw_h1, order="F")

            raw_inp = np.sqrt(len(i1))
            i1 = i1.reshape(raw_inp,raw_inp, order="F")
            i2 = i2.reshape(raw_inp,raw_inp, order="F")
            i3 = i3.reshape(raw_inp,raw_inp, order="F")

            raw_out = np.sqrt(len(o1))
            o1 = o1.reshape(raw_out, raw_out)

            sm_l = len(wsm1)
            wsm1_gr = wgr1[:,(0*sm_l):(1*sm_l)]
            wsm2_gr = wgr1[:,(1*sm_l):(2*sm_l)]
            wsm3_gr = wgr1[:,(2*sm_l):(3*sm_l)]

            wsm_gr = np.hstack([
                reshape_weights(wsm1_gr),
                reshape_weights(wsm2_gr),
                reshape_weights(wsm3_gr)
            ])

            h_l = len(h1)
            wh1_gr = wgr1[:,((3*sm_l) + 0*h_l):((3*sm_l) + 1*h_l)]
            wh2_gr = wgr1[:,((3*sm_l) + 1*h_l):((3*sm_l) + 2*h_l)]

            wh_gr = np.hstack([
                reshape_weights(wh1_gr),
                reshape_weights(wh2_gr)
            ])

            o_l = len(o1)
            wo_gr = wgr1[:,(3*sm_l+3*h_l):((3*sm_l+3*h_l) + o_l)]
            reshape_weights(wo_gr)

            raw_gr1 = np.sqrt(len(gr1))
            gr1 = gr1.reshape(raw_gr1,raw_gr1, order="F")

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

            self.imgs[18].setImage( gr1, levels=(0,1) )

            self.t += 1

            self.app.processEvents()  ## force complete redraw for every plot

    def start(self) :
        self.timer.timeout.connect(self.update)
        self.timer.start(0.001)

    def stop(self) :
        print "Graph connection closed\n"
        self.timer.stop()
        QtGui.QApplication.quit()


## Start Qt event loop unless running in interactive mode.
if __name__ == '__main__':
    import sys

    main = Main()
    main.start()

    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()

