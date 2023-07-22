import sys
from PyQt5.QtWidgets import QApplication, QWidget, QTableWidget, QHBoxLayout
from PyQt5.QtWidgets import  *
class MyWidget(QWidget):
    def __init__(self):
        super().__init__()

        # Tạo một QTableWidget
        self.table = QTableWidget()

        # Thêm các cột vào QTableWidget
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(["Name", "Age", "Gender"])

        # Thêm các hàng vào QTableWidget
        self.table.setRowCount(5)
        self.table.setItem(0, 0, QTableWidgetItem("John Doe"))
        self.table.setItem(0, 1, QTableWidgetItem("25"))
        self.table.setItem(0, 2, QTableWidgetItem("Male"))
        self.table.setItem(1, 0, QTableWidgetItem("Jane Doe"))
        self.table.setItem(1, 1, QTableWidgetItem("23"))
        self.table.setItem(1, 2, QTableWidgetItem("Female"))
        self.table.setItem(2, 0, QTableWidgetItem("Peter Smith"))
        self.table.setItem(2, 1, QTableWidgetItem("30"))
        self.table.setItem(2, 2, QTableWidgetItem("Male"))
        self.table.setItem(3, 0, QTableWidgetItem("Mary Jones"))
        self.table.setItem(3, 1, QTableWidgetItem("21"))
        self.table.setItem(3, 2, QTableWidgetItem("Female"))
        self.table.setItem(4, 0, QTableWidgetItem("David Brown"))
        self.table.setItem(4, 1, QTableWidgetItem("27"))
        self.table.setItem(4, 2, QTableWidgetItem("Male"))

        # Tạo một QHBoxLayout
        self.layout = QHBoxLayout()

        # Thêm QTableWidget vào QHBoxLayout
        self.layout.addWidget(self.table)

        # Đặt layout cho QWidget
        self.setLayout(self.layout)

    def sortByColumn(self, column):
        # Lấy dữ liệu từ QTableWidget
        data = []
        for row in range(self.table.rowCount()):
            data.append([self.table.item(row, column).text()])

        # Sắp xếp dữ liệu
        data.sort()

        # Tải lại dữ liệu đã sắp xếp vào QTableWidget
        for row in range(len(data)):
            self.table.setItem(row, column, QTableWidgetItem(data[row][0]))

if __name__ == "__main__":
    # Tạo một ứng dụng Qt5
    application = QApplication(sys.argv)

    # Tạo một widget
    widget = MyWidget()

    # Hiển thị widget
    widget.show()

    # Chạy ứng dụng
    sys.exit(application.exec_())