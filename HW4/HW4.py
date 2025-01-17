#-*-coding:utf-8-*-
import sys
import cv2
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QGridLayout, QLabel, QPushButton
from PyQt5.QtGui import QImage, QPixmap #, QPainter
from PyQt5.QtCore import pyqtSlot, QDir, QFile, QIODevice, QTextStream
from Mainwindow import Ui_MainWindow
import matplotlib.pyplot as plt
from fun import * 
import os


class QmyMainWindow(QMainWindow):
    def __init__(self, parent = None):
        global fileName

        super().__init__(parent)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.ui.btnFile.clicked.connect(self.loadfile)
        self.ui.btn_fft.clicked.connect(self.fft)
        self.ui.btn_part2.clicked.connect(self.secondpart)
        self.ui.btn_homo.clicked.connect(self.homo)
        self.ui.btn_blur.clicked.connect(self.blur)
        fileName = ''
    # Load files
    def loadfile(self):

        global fileName
        global img_arr
        global dfred
        global dfgreen
        global dfblue
        global r_show
        global g_show
        global b_show
        global imgc
        curPath = QDir.currentPath()
        f = QFileDialog.getOpenFileName(self, 'openFile', '', '*.bmp;*.jpg;*.png;*.tif')
        fileName = f[0]

        if (fileName == ''):
            return

        img_arr = cv2.imdecode(np.fromfile(fileName,dtype=np.uint8),1)        
        dfred, r_show = fft(img_arr[:, :, 0])
        dfgreen, g_show = fft(img_arr[:, :, 1])
        dfblue, b_show = fft(img_arr[:, :, 2])

        if img_arr.shape[0] == 688:
            imgc = 1
        elif img_arr.shape[0] == 512:
            imgc = 2
        else:
            imgc = 3

        self.do_showImage1(fileName)
    # Part 1
    # FFT
    def fft(self):
        if (fileName == ''):
            return        

        result1 = np.dstack([r_show, g_show, b_show])
        self.do_showR1(result1)
        self.ui.lab_t1.setText(" Spectrum")

        r_p = phase_angle(dfred)
        g_p = phase_angle(dfgreen)
        b_p = phase_angle(dfblue)
        result2 = np.dstack([r_p,g_p,b_p])
        self.do_showR2(result2)
        self.ui.lab_t2.setText("Phase angle") 
        
        r_i = ifft(dfred)
        g_i = ifft(dfgreen)
        b_i = ifft(dfblue)
        result3 = np.dstack([r_i,g_i,b_i])
        self.do_showR3(result3)
        self.ui.lab_t3.setText("Inverse img") 
    # Part 2
    def secondpart(self):     
        if (fileName == ''):
            return
        self.ui.lab_r1.clear(), self.ui.lab_r2.clear(), self.ui.lab_r3.clear()
        self.ui.lab_t1.clear(), self.ui.lab_t2.clear(), self.ui.lab_t3.clear()

        if  self.ui.rbtn_high.isChecked():
            passchoose = 1
        elif self.ui.rbtn_low.isChecked():
            passchoose = 0

        if self.ui.rbtn_ideal.isChecked():
            self.ideal(passchoose)
        elif self.ui.rbtn_gaussion.isChecked():
            self.gaussion(passchoose)
        elif self.ui.rbtn_butter.isChecked():
            self.butter(passchoose)
        else:
            pass
    # Ideal filter
    def ideal(self,passchoose):
        if (fileName == ''): 
            return       

        r1 = ideal_filter(dfred,self.ui.ideal_cutoff.value(),passchoose)
        g1 = ideal_filter(dfgreen,self.ui.ideal_cutoff.value(),passchoose)
        b1 = ideal_filter(dfblue,self.ui.ideal_cutoff.value(),passchoose)
        r_i = ifft(r1)
        g_i = ifft(g1)
        b_i = ifft(b1)
        result = np.dstack([r_i,g_i,b_i])
        self.do_showR1(result)
        self.ui.lab_t1.setText("Ideal filter")
    # Gaussion filter
    def gaussion(self,passchoose):
        if (fileName == ''):
            return

        r1 = gauss_filter(dfred,self.ui.gaussion_cutoff.value(),passchoose)
        g1 = gauss_filter(dfgreen,self.ui.gaussion_cutoff.value(),passchoose)
        b1 = gauss_filter(dfblue,self.ui.gaussion_cutoff.value(),passchoose)
        r_i = ifft(r1)
        g_i = ifft(g1)
        b_i = ifft(b1)
        result = np.dstack([r_i,g_i,b_i])
        self.do_showR1(result)
        self.ui.lab_t1.setText("Gaussian filter")   
    # Butterworth filter
    def butter(self,passchoose):
        if (fileName == ''):
            return
        r1 = bufilter(dfred,self.ui.butter_cutoff.value(),self.ui.butter_order.value(),passchoose)
        g1 = bufilter(dfgreen,self.ui.butter_cutoff.value(),self.ui.butter_order.value(),passchoose)
        b1 = bufilter(dfblue,self.ui.butter_cutoff.value(),self.ui.butter_order.value(),passchoose)
        r_i = ifft(r1)
        g_i = ifft(g1)
        b_i = ifft(b1)
        result = np.dstack([r_i,g_i,b_i])
        self.do_showR1(result)
        self.ui.lab_t1.setText("Butter filter")
    # Homomorphic filter
    def homo(self):
        if (fileName == ''):
            return

        self.ui.lab_r1.clear(), self.ui.lab_r2.clear(), self.ui.lab_r3.clear()
        self.ui.lab_t1.clear(), self.ui.lab_t2.clear(), self.ui.lab_t3.clear()

        r1 = homo_filter(img_arr[:, :, 0],self.ui.spin_rh.value(),self.ui.spin_rl.value(),self.ui.spin_d0.value())
        g1 = homo_filter(img_arr[:, :, 1],self.ui.spin_rh.value(),self.ui.spin_rl.value(),self.ui.spin_d0.value())
        b1 = homo_filter(img_arr[:, :, 2],self.ui.spin_rh.value(),self.ui.spin_rl.value(),self.ui.spin_d0.value())
        result = np.dstack([r1,g1,b1])
        self.do_showR1(result)
        self.ui.lab_t1.setText("Homomorphic filter")
          
    # Part 4
    def blur(self):
        if (fileName == ''):
            return

        self.ui.lab_r1.clear(), self.ui.lab_r2.clear(), self.ui.lab_r3.clear()
        self.ui.lab_t1.clear(), self.ui.lab_t2.clear(), self.ui.lab_t3.clear()

        if  self.ui.rbtn_noise.isChecked():
            nchoose = 1
            self.ui.lab_t1.setText("Motion blurred + noise") 
        else:
            nchoose = 0
            self.ui.lab_t1.setText("Motion blurred") 

        if imgc == 3:
            r1,r2 = blur_filter(img_arr[:, :, 0])
            g1,g2 = blur_filter(img_arr[:, :, 1])
            b1,b2 = blur_filter(img_arr[:, :, 2])
            result = np.dstack([r1,g1,b1])
            result2 = np.dstack([r2,g2,b2])
            self.do_showR1(result)
            self.do_showR2(result2)
        else:            
            result1 = blur_filter1(imgc, nchoose)
            print('result: ', result1)
            result2 = blur_filter2(imgc, nchoose)
            result3 = blur_filter3(imgc, nchoose)
            self.do_showR1(result1)
            self.do_showR2(result2)
            self.do_showR3(result3)
                
        self.ui.lab_t2.setText("Inverse filter") 
        self.ui.lab_t3.setText("Wiener filter") 

    def do_showImage1(self, filename):
        image = QPixmap(filename)
        self.ui.lab_img.setPixmap(image)
        self.ui.lab_img.setScaledContents(True)

    def do_showR1(self, image): 
        image2 = image[:, :, 0]/3 + image[:, :, 1]/3 + image[:, :, 2]/3
        image2 = np.where(image2 > 255, 255, image2)
        image2 = np.where(image2 < 0, 0, image2)
        
        totalBytes = np.size(image2)
        bytesPerLine = int(totalBytes/image2.shape[0])
        Qimg2 = QImage(np.uint8(image2), image2.shape[1], image2.shape[0], bytesPerLine, QImage.Format_Grayscale8)

        Qimg2 = QPixmap(Qimg2)
        self.ui.lab_r1.setPixmap(Qimg2)
        self.ui.lab_r1.setScaledContents(True)

    def do_showR2(self, image): 
        image2 = image[:, :, 0]/3 + image[:, :, 1]/3 + image[:, :, 2]/3
        image2 = np.where(image2 > 255, 255, image2)
        image2 = np.where(image2 < 0, 0, image2)

        totalBytes = np.size(image2)
        bytesPerLine = int(totalBytes/image2.shape[0])
        Qimg2 = QImage(np.uint8(image2), image2.shape[1], image2.shape[0], bytesPerLine, QImage.Format_Grayscale8)

        Qimg2 = QPixmap(Qimg2)
        self.ui.lab_r2.setPixmap(Qimg2)
        self.ui.lab_r2.setScaledContents(True)

    def do_showR3(self, image): 
        image2 = image[:, :, 0]/3 + image[:, :, 1]/3 + image[:, :, 2]/3
        image2 = np.where(image2 > 255, 255, image2)
        image2 = np.where(image2 < 0, 0, image2)

        totalBytes = np.size(image2)
        bytesPerLine = int(totalBytes/image2.shape[0])
        Qimg2 = QImage(np.uint8(image2), image2.shape[1], image2.shape[0], bytesPerLine, QImage.Format_Grayscale8)
  
        Qimg2 = QPixmap(Qimg2)
        self.ui.lab_r3.setPixmap(Qimg2)
        self.ui.lab_r3.setScaledContents(True)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    form = QmyMainWindow()
    form.show()
    sys.exit(app.exec_())
