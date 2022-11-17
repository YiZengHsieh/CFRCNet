from cgi import test
import sys
import os
import shutil
import time
from threading import Thread
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtGui import QPixmap, QIcon
from PyQt5.QtGui import *
from UI_demo import Ui_ThesisDemo as Ui_MainWindow
# from UI_demo import Ui_MainWindow
import shutil
import cv2


class mywindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(mywindow, self).__init__()
        self.setupUi(self)
        self.path = []
        self.i = 0
        #self.model = model()

        self.choiceButton.clicked.connect(self.load_data)
        self.startButton.clicked.connect(self.result)
        #self.startButton.clicked.connect(self.predict)
        

    def load_data(self):
        src = str("./filenames/newfile.txt")
        des = str("./filenames/demo.txt")
        shutil.copy(src, des)
        file= QFileDialog.getOpenFileName(self, '選取檔案', '/media/cihci/0000678400004823/andy_master/code/demo/left', 'img(*.jpg *.png)')
        img_name1 = str(file)
        img_name1 = img_name1.split("/demo/left",1)
        img_name1 = img_name1[1]
        img_name1 = img_name1.split(".png",1)
        file_filter = img_name1[0]
        print(file_filter)

        path = "./filenames/demo.txt"
        f = open(path, 'a')
        f.write(str("demo/left"+file_filter +".png demo/right"+file_filter+".png demo/disparity_pfm"+file_filter+".pfm"))
        f.write("\n")
        f.close()
        # pixmap = QPixmap(file).scaled(216,300)
        # print(len(file))
        # pixmap = QPixmap(file).scaled(920,480)
        file = str(file)
        file_name = str(file).split('/')
        self.path.append(file_name[4])
        txt_file = open('./filename.txt' , 'w')
        txt_file.write(file)
        txt_file.close()
        # self.input_label.setText('Stereo Image : ' + file)
        #self.out_label.setPixmap(pixmap)
    
    def result(self):
        self.test = Test()
        print(self.path)
        
        #self.timer = QTimer()
        #self.timer.start(1000)
        #self.timer.timeout.connect(lambda:self.load_image(self.i))
        self.test.progress.connect(self.call_state)
        self.test.start()
        

    def load_image(self):
        self.i += 1
        readtxt = open('./filename.txt', 'r')
        img_name = readtxt.read()
        readtxt.close()

        img_name1 = str(img_name)
        img_name1 = img_name1.split("/demo/left/",1)
        img_name1 = img_name1[1]
        img_name1 = img_name1.split(".png",1)
        img_name = img_name1[0]
        disparity_path = str('/media/cihci/0000678400004823/andy_master/code/demo/output/'+img_name+'.png')
        if os.path.exists(disparity_path) :
            im = cv2.imread('/media/cihci/0000678400004823/andy_master/code/demo/left/'+img_name+'.png')
            im = cv2.resize(im, (640,360 ), interpolation=cv2.INTER_AREA)
            
            h , w, channel = im.shape
            byPer = 3* w
            cv2.cvtColor(im, cv2.COLOR_BGR2RGB, im)
            qimg = QImage(im.data, w, h, byPer, QImage.Format_RGB888)
            fpixmap = QPixmap.fromImage(qimg)
            self.basic_disparity.setPixmap(fpixmap)
            # shutil.copy('./data/train_left_20220707/'+img_name+'.png','./demo_file/acvnet/data'+img_name+'.png')

            im = cv2.imread('/media/cihci/0000678400004823/andy_master/code/demo/disparity/'+img_name+'.png')
            im = cv2.resize(im, (640,341), interpolation=cv2.INTER_AREA)
            h , w, channel = im.shape
            byPer = 3* w
            cv2.cvtColor(im, cv2.COLOR_BGR2RGB, im)
            qimg = QImage(im.data, w, h, byPer, QImage.Format_RGB888)
            fpixmap = QPixmap.fromImage(qimg)
            self.left_image.setPixmap(fpixmap)


            im = cv2.imread(disparity_path)
            im = cv2.resize(im, (640, 341), interpolation=cv2.INTER_AREA)
            h , w, channel = im.shape
            byPer = 3* w
            cv2.cvtColor(im, cv2.COLOR_BGR2RGB, im)
            qimg = QImage(im.data, w, h, byPer, QImage.Format_RGB888)
            fpixmap = QPixmap.fromImage(qimg)
            self.right_image.setPixmap(fpixmap)


    def call_state(self, msg):
        if msg == 1:
        #    self.timer.stop()
            self.test.quit()
            self.load_image()

class Test(QThread):
    progress = pyqtSignal(int)
    def __init__(self):
        super().__init__()
        #self.file = file

    def run(self):
        os.system("python save_disp_sceneflow.py")
        #print(self.file)
        self.progress.emit(1)



if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = mywindow()
    window.show()
    sys.exit(app.exec_()) 