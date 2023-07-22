from PyQt5 import QtCore,QtGui,QtWidgets,uic
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QPixmap,QImage,QCursor
from PyQt5.QtCore import Qt
from PyQt5.QtCore import Qt, pyqtSlot, QTimer, QThread,pyqtSignal
from PyQt5.uic import loadUi
from PyQt5.QtGui import QColor
import  sys
from facebank import prepare_facebank
from cam_demo import inference
import openpyxl
import os
import torch
from torchvision import transforms as trans
import numpy as np
from util import *
from align_trans import *
from MTCNN import create_mtcnn_net
from face_model import MobileFaceNet, l2_norm
from facebank import load_facebank,save_facebank, prepare_facebank,info_class,save_attendance,update_class
import cv2
from cam_demo import load_model
import time
from datetime import datetime
import copy
import os
import urllib
os.environ['KMP_DUPLICATE_LIB_OK']='True'
def is_valid_time(value):
    try:
        datetime.strptime(value, "%Y-%m-%d %H:%M")  # Định dạng thời gian HH:MM:SS
        return True
    except ValueError:
        return False
class capture_video(QThread):
    signal = pyqtSignal(np.ndarray)
    def __init__(self,index,lop,name_model,table,localhost):
        self.index =index
        self.localhost = localhost
        print("start threading",self.index)
        super(capture_video,self).__init__()
        self.model = load_model(name_model)
        self.emb,self.name,self.mssv = load_facebank(lop)
        self.lop =lop
        print(self.name)
        self.diem_danh =[]
        self.cap =None
        self.thread_table = table
        self.name_diemdanh=[]

    def run(self):
        # self.cap =cv2.VideoCapture(0)
        # url ="http://192.168.1.9:8080/photo.jpg"
        print(11111)
        while True:
            print("abc")
            imgResp = urllib.request.urlopen(self.localhost)
            # print("imgResp",imgResp.status_code)

            # ret, cv_img = self.cap.read()
            if True  :
                imgNp = np.array(bytearray(imgResp.read()), dtype=np.uint8)
                cv_img = cv2.imdecode(imgNp, -1)
                try:
                    cv_img =cv2.flip(cv_img,1)
                    cv_img,res,score_100 =inference(cv_img,self.emb,self.name,self.mssv,self.model)
                    for name,score in zip(res,score_100):
                        if score > 70 and name not in self.diem_danh:
                            self.diem_danh.append(name)
                            self.name_diemdanh.append(self.mssv.index(name))

                    self.display_data()
                except Exception as e:
                    print(e)
                self.signal.emit(cv_img)
    def stop(self):
        print("stop threading", self.index)
        print(self.diem_danh)
        # self.cap.release()
        # cv2.release()
        self.terminate()
        now = datetime.now()
        current_time = now.strftime("%Y-%m-%d %H:%M")
        save_attendance(current_time,self.diem_danh,self.mssv,self.lop)

    def display_data(self):
        row_count = len(self.diem_danh)
        self.thread_table.setRowCount(row_count)
        self.thread_table.setColumnCount(2)
        for row, data in enumerate(self.diem_danh):
            item = QTableWidgetItem(data)
            name = QTableWidgetItem(self.name[self.name_diemdanh[row]])
            self.thread_table.setItem(row, 0, item)
            self.thread_table.setItem(row, 1, name)
        self.thread_table.resizeColumnsToContents()
        self.thread_table.resizeRowsToContents()

class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        loadUi("app.ui", self)
        self.num_model =1
        self.home_home.clicked.connect(self.show_home_page)
        self.home_class.clicked.connect(self.show_class_page)
        self.home_diemdanh.clicked.connect(self.show_diemdanh_page)
        self.page_class_themlop.clicked.connect(self.link_class)
        if not os.path.exists("class.txt"):
            with open("class.txt", "w") as file:
                pass
        options = self.load_options_from_file("class.txt")
        self.page_class_chonlop.addItems(options)
        self.page_diemdanh_chonlop.addItems(options)
        self.page_class_sumbit_chonlop.clicked.connect(self.submit_options)
        self.page_diemdanh_stop.clicked.connect(self.stop_capture_video)
        self.page_diemdanh_start.clicked.connect(self.start_camera)
        self.page_class_themsinhvien.clicked.connect(self.add_student)
        self.page_class_sua.clicked.connect(self.read_table_data)
        self.thread = {}

        # test right click
        self.page_class_table.horizontalHeader().setContextMenuPolicy(Qt.CustomContextMenu)
        self.page_class_table.horizontalHeader().customContextMenuRequested.connect(self.show_header_context_menu)
        self.page_class_table.verticalHeader().setContextMenuPolicy(Qt.CustomContextMenu)
        self.page_class_table.verticalHeader().customContextMenuRequested.connect(self.show_header_context_menu)
        self.page_class_xoa.clicked.connect(self.on_button_clicked)

    def on_button_clicked(self):
        # Hiển thị hộp thoại xác nhận
        selected_option = self.page_class_chonlop.currentText()
        message = QMessageBox()
        message.setIcon(QMessageBox.Question)
        message.setWindowTitle("Xác nhận")
        message.setText("Bạn đã chắc chắn muốn xóa không?")
        message.setStandardButtons(QMessageBox.Yes | QMessageBox.No)

        # Lấy lựa chọn của người dùng
        choice = message.exec_()

        # Nếu người dùng chọn "Có", hãy xóa dữ liệu
        if choice == QMessageBox.Yes:
            # Xóa dữ liệu
            with open('class.txt', 'r') as file:
                lines = file.readlines()
            for line in lines:
                if selected_option in line:
                    lines.remove(line)
            with open('class.txt', 'w') as file:
                file.writelines(lines)
        # Nếu người dùng chọn "Không", hãy bỏ qua
        else:
            print("Dữ liệu đã không bị xóa.")
    def show_header_context_menu(self, position):
        header = self.sender()
        index = header.logicalIndexAt(position)
        menu = QMenu(self)
        if header == self.page_class_table.horizontalHeader():
            delete_action = QAction("Xóa cột", self)
            add_column_action = QAction("Thêm cột", self)
            add_column_action.triggered.connect(lambda: self.add_column(index))
            delete_action.triggered.connect(lambda: self.delete_column(index))
            menu.addAction(delete_action)
            menu.addAction(add_column_action)
        elif header == self.page_class_table.verticalHeader():
            delete_action = QAction("Xóa hàng", self)
            delete_action.triggered.connect(lambda: self.delete_row(index))
            menu.addAction(delete_action)

        menu.exec_(header.viewport().mapToGlobal(position))

    def add_column(self, index):
        header_label, ok = QInputDialog.getText(self, "Thêm cột", "Nhập tiêu đề cột:")

        if ok:
            new_column_count = self.page_class_table.columnCount() + 1
            self.page_class_table.setColumnCount(new_column_count)

            header_item = QTableWidgetItem(header_label)
            self.page_class_table.setHorizontalHeaderItem(index + 1, header_item)
    def delete_column(self, column):
        self.page_class_table.removeColumn(column)

    def delete_row(self, row):
        self.page_class_table.removeRow(row)
    def read_table_data(self):
        try:
            selected_option = self.page_class_chonlop.currentText()
            dict_class =info_class(selected_option)
            row_count = self.page_class_table.rowCount()
            column_count = self.page_class_table.columnCount()
            header = self.get_table_header()
            print(header)
            list_mssv =[]
            for row in range(row_count):
                mssv = self.page_class_table.item(row, 0)
                if mssv is not None:
                    mssv = mssv.text()
                    list_mssv.append(mssv)
                    for column in range(2,column_count):
                        item = self.page_class_table.item(row, column)
                        if item is not None:
                            dict_class[mssv][header[column]] =item.text()
                        else:
                            dict_class[mssv][header[column]] =""
            print("list_mssv",list_mssv[0])
            copy_dict = copy.deepcopy(dict_class)
            for key in copy_dict.keys():
                if key not in list_mssv:
                    dict_class.pop(key)
            copy_dict = copy.deepcopy(dict_class)

            for key in copy_dict[list_mssv[0]].keys():
                dem =0
                if key != "name" and key != "emb" and key not in header:
                    print(key)

                    for mssv in copy_dict.keys():
                        dem += 1
                        dict_class[mssv].pop(key)

            save_facebank(dict_class,selected_option)
        except Exception as e:
            print(e)

    def get_table_header(self):
        column_count = self.page_class_table.columnCount()
        header = []
        for column in range(column_count):
            item = self.page_class_table.horizontalHeaderItem(column)
            if item is not None:
                header.append(item.text())
            else:
                header.append("")

        return header

    def add_student(self):
        try :
            selected_option = self.page_class_chonlop.currentText()
            folder_path = QFileDialog.getExistingDirectory(None, "Chọn thư mục", "/")
            print(selected_option)
            update_class(selected_option,folder_path)
            # self.repaint()
        except Exception as e:
            print(e)
    def stop_capture_video(self):
        self.thread[1].stop()
    def start_camera(self):
        selected_option = self.page_diemdanh_chonlop.currentText()
        text = self.page_diemdanh_localhost.text()
        print(text)
        try :
            self.thread[1] = capture_video(index=1, lop=selected_option,
                                           name_model=self.num_model,table =self.page_diemdanh_table,localhost =text)
            self.thread[1].start()
            self.thread[1].signal.connect(self.show_wedcam)
        except Exception as e:
            print(e)
    def show_wedcam(self,cv_img):
        qt_img = self.convert_cv_qt(cv_img)
        self.page_diemdanh_camera.setPixmap(qt_img)
    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(800, 600, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)
    def image_error1(self,er):
        # Tạo một cửa sổ thông báo
        msg_box = QMessageBox()
        msg_box.setText("Image Error")
        msg_box.setWindowTitle("Lỗi ")
        error_message = "Một số ảnh không có người. Các ảnh bị lỗi:\n"
        error_message += "\n".join(er)
        error_message += "\nNếu tất cả hình ảnh đều không có người thì lớp sẽ không được tạo!"
        print("error_message",error_message)
        msg_box.setIcon(QMessageBox.Warning)
        msg_box.setDetailedText(error_message)
        msg_box.setStandardButtons(QMessageBox.Ok)
        msg_box.setDefaultButton(QMessageBox.Ok)
        msg_box.setStyleSheet("QLabel{min-width:200 px; font-size: 18px;} QPushButton{ width:100px; font-size: 12px; }");
        # Hiển thị cửa sổ thông báo và chờ người dùng đóng
        msg_box.exec_()
    def image_error0(self,er):
        # Tạo một cửa sổ thông báo
        msg_box = QMessageBox()
        msg_box.setText("Tạo lớp thành công")
        msg_box.setWindowTitle("Thông báo ")
        msg_box.setIcon(QMessageBox.Information)
        msg_box.setStandardButtons(QMessageBox.Ok)
        msg_box.setDefaultButton(QMessageBox.Ok)
        msg_box.setStyleSheet("QLabel{min-width:200 px; font-size: 18px;} QPushButton{ width:100px; font-size: 12px; }");
        # Hiển thị cửa sổ thông báo và chờ người dùng đóng

        msg_box.exec_()
    def error_chonlop(self):
        msg_box = QMessageBox()
        msg_box.setText("Lỗi lớp được chọn không hợp lệ")
        msg_box.setWindowTitle("Lỗi")
        msg_box.setIcon(QMessageBox.Critical)
        msg_box.setStandardButtons(QMessageBox.Ok)
        msg_box.setDefaultButton(QMessageBox.Ok)
        msg_box.setStyleSheet(
            "QLabel{min-width:200 px; font-size: 18px;} QPushButton{ width:100px; font-size: 12px; }");
        # Hiển thị cửa sổ thông báo và chờ người dùng đóng
        msg_box.exec_()
    def submit_options(self):
        # Khởi tạo layout chính
        selected_option = self.page_class_chonlop.currentText()
        # TODO: Send the selected option to the server
        print("Selected option:", selected_option)
        try:
            # Khởi tạo bảng
            # self.page_class_excel_layout.addWidget(self.page_class_table)
            my_dict = info_class(selected_option)
            name_header = list(my_dict[next(iter(my_dict))].keys())
            embs,list_name,list_mssv = load_facebank(selected_option)
            name_header[0] ="MSSV"
            name_header[1]="Họ và Tên"
            self.page_class_table.setRowCount(len(my_dict))
            self.page_class_table.setColumnCount(len(name_header))
            # Đặt tiêu đề cho các cột
            self.page_class_table.setHorizontalHeaderLabels(name_header)
            name_header.remove("MSSV")
            name_header.remove("Họ và Tên")
            print(name_header)
            name_header = sorted(name_header)
            print(name_header)
            # Duyệt qua các phần tử trong my_dict và đặt giá trị vào bảng
            row_index = 0
            for key in list_mssv:
                item_key = QTableWidgetItem(key)
                item_key.setFlags(item_key.flags() ^ Qt.ItemIsEditable)
                self.page_class_table.setItem(row_index, 0, item_key)

                item_name = QTableWidgetItem(my_dict[key]["name"])
                item_name.setFlags(item_name.flags() ^ Qt.ItemIsEditable)
                self.page_class_table.setItem(row_index, 1, item_name)
                col_index = 2
                for t  in name_header:
                    value =str(my_dict[key][t])
                    item = QTableWidgetItem(value)
                    self.page_class_table.setItem(row_index, col_index,item)
                    col_index+=1
                    if value == "0" and is_valid_time(t):
                        item.setBackground(QColor("darkGray"))
                    elif value == "1" and is_valid_time(t):
                        item.setBackground(QColor("lightgreen"))
                row_index += 1
            self.page_class_table.resizeColumnsToContents()
            self.page_class_table.resizeRowsToContents()
            # self.page_class_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        except Exception as e :
            print("error",e)
            self.error_chonlop()
    def load_options_from_file(self, filename):
        with open(filename, "r") as file:
            options = file.read().splitlines()
        print("class",options)
        return options

    def start_diemdanh(self):
        selected_option = self.page_diemdanh_chonlop.currentText()
        self.start_video_thread(lop=selected_option)

    def link_class(self):
        folder_path = QFileDialog.getExistingDirectory(None, "Chọn thư mục", "/")
        print("Đường dẫn được chọn:", folder_path)
        er = prepare_facebank(folder_path)
        self.repaint()
        if len(er)!=0:
            self.image_error1(er)
        else:
            self.image_error0(er)
        print(er)
        options = self.load_options_from_file("class.txt")
        self.page_class_chonlop.addItems(options)
        self.page_diemdanh_chonlop.addItems(options)
        self.repaint()

    def show_home_page(self):
        # Hiển thị trang Home
        self.stackedWidget.setCurrentWidget(self.page_home)
    def show_class_page(self):
        # Hiển thị trang Lớp
        self.stackedWidget.setCurrentWidget(self.page_class)
    def show_diemdanh_page(self):
        # Hiển thị trang Lớp
        self.stackedWidget.setCurrentWidget(self.page_diemdanh)


app = QApplication(sys.argv)
window = QtWidgets.QStackedWidget()
main_f = MainWindow()
# window.setCentralWidget(main_f)
window.addWidget(main_f)
window.setCurrentIndex(0)
window.showMaximized()
app.exec_()

