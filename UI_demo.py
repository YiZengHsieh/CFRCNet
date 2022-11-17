# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'thesis.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_ThesisDemo(object):
    def setupUi(self, ThesisDemo):
        ThesisDemo.setObjectName("ThesisDemo")
        ThesisDemo.setEnabled(True)
        ThesisDemo.resize(1378, 838)
        self.centralwidget = QtWidgets.QWidget(ThesisDemo)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.gridLayoutWidget.setGeometry(QtCore.QRect(0, 50, 1371, 651))
        self.gridLayoutWidget.setObjectName("gridLayoutWidget")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.gridLayoutWidget)
        self.gridLayout_2.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.left_image = QtWidgets.QLabel(self.gridLayoutWidget)
        self.left_image.setText("")
        self.left_image.setAlignment(QtCore.Qt.AlignBottom|QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing)
        self.left_image.setObjectName("left_image")
        self.gridLayout_2.addWidget(self.left_image, 0, 0, 1, 1)
        self.right_image = QtWidgets.QLabel(self.gridLayoutWidget)
        self.right_image.setText("")
        self.right_image.setAlignment(QtCore.Qt.AlignBottom|QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft)
        self.right_image.setObjectName("right_image")
        self.gridLayout_2.addWidget(self.right_image, 0, 1, 1, 1)
        self.adj_disparity = QtWidgets.QLabel(self.gridLayoutWidget)
        self.adj_disparity.setText("")
        self.adj_disparity.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        self.adj_disparity.setObjectName("adj_disparity")
        self.gridLayout_2.addWidget(self.adj_disparity, 1, 1, 1, 1)
        self.basic_disparity = QtWidgets.QLabel(self.gridLayoutWidget)
        self.basic_disparity.setText("")
        self.basic_disparity.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTop|QtCore.Qt.AlignTrailing)
        self.basic_disparity.setObjectName("basic_disparity")
        self.gridLayout_2.addWidget(self.basic_disparity, 1, 0, 1, 1)
        self.choiceButton = QtWidgets.QPushButton(self.centralwidget)
        self.choiceButton.setGeometry(QtCore.QRect(0, 720, 291, 71))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(36)
        font.setBold(True)
        font.setItalic(False)
        font.setWeight(75)
        self.choiceButton.setFont(font)
        self.choiceButton.setObjectName("choiceButton")
        self.startButton = QtWidgets.QPushButton(self.centralwidget)
        self.startButton.setGeometry(QtCore.QRect(300, 720, 311, 71))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(36)
        font.setBold(True)
        font.setItalic(False)
        font.setWeight(75)
        self.startButton.setFont(font)
        self.startButton.setObjectName("startButton")
        self.input_label = QtWidgets.QLabel(self.centralwidget)
        self.input_label.setGeometry(QtCore.QRect(500, 620, 771, 41))
        self.input_label.setText("")
        self.input_label.setObjectName("input_label")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(0, -10, 631, 71))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(36)
        font.setBold(True)
        font.setItalic(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(640, -10, 631, 71))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(36)
        font.setBold(True)
        font.setItalic(True)
        font.setWeight(75)
        self.label_2.setFont(font)
        self.label_2.setAlignment(QtCore.Qt.AlignCenter)
        self.label_2.setObjectName("label_2")
        ThesisDemo.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(ThesisDemo)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1378, 20))
        self.menubar.setObjectName("menubar")
        ThesisDemo.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(ThesisDemo)
        self.statusbar.setObjectName("statusbar")
        ThesisDemo.setStatusBar(self.statusbar)

        self.retranslateUi(ThesisDemo)
        QtCore.QMetaObject.connectSlotsByName(ThesisDemo)

    def retranslateUi(self, ThesisDemo):
        _translate = QtCore.QCoreApplication.translate
        ThesisDemo.setWindowTitle(_translate("ThesisDemo", "MainWindow"))
        self.choiceButton.setText(_translate("ThesisDemo", "Choose File"))
        self.startButton.setText(_translate("ThesisDemo", "Run Program"))
        self.label.setText(_translate("ThesisDemo", "Ground Truth"))
        self.label_2.setText(_translate("ThesisDemo", "Our Purposed"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    ThesisDemo = QtWidgets.QMainWindow()
    ui = Ui_ThesisDemo()
    ui.setupUi(ThesisDemo)
    ThesisDemo.show()
    sys.exit(app.exec_())
