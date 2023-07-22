import sys
# pip install pyqt5
import cv2
import numpy as np
from PyQt5 import QtGui
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.uic import loadUi
from functools import partial
class Camera(QMainWindow):
    def __init__(self):
        super(Camera,self).__init__()
        loadUi("camera.ui",self)
        self.start.clicked.connect(self.start_camera)
        self.stop.clicked.connect(self.stop_capture_video)

        self.thread ={}
    def stop_capture_video(self):
        self.thread[1].stop()
        print(self.thread[1].test)
    def start_camera(self):
        self.thread[1] =capture_video(index=1)
        self.thread[1].start()
        self.thread[1].signal.connect(self.show_wedcam)

    def show_wedcam(self, cv_img):
        qt_img = self.convert_cv_qt(cv_img)
        self.label.setPixmap(qt_img)
        return 1
    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(800, 600, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)
class capture_video(QThread):
    signal = pyqtSignal(np.ndarray)
    def __init__(self,index):
        self.index =index
        self.cap = None
        self.test =1

        print("start threading",self.index)
        super(capture_video,self).__init__()
    def run(self):
        self.cap =cv2.VideoCapture(0)
        while True:
            ret, cv_img =self.cap.read()
            if ret :
                self.signal.emit(cv_img)
        # self.cap.release()
    def stop(self):
        print("stop threading", self.index)
        self.cap.release()
        self.terminate()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainwin = Camera()
    mainwin.show()
    sys.exit(app.exec_())
