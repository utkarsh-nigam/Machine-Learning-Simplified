'''
from PyQt5.QtWidgets import (QMainWindow, QApplication, QWidget, QPushButton, QAction, QComboBox, QLabel,
                             QGridLayout, QCheckBox, QGroupBox, QVBoxLayout, QHBoxLayout, QLineEdit, QPlainTextEdit,
                             QInputDialog, QFileDialog, QTableView, QFormLayout, QScrollArea, QSpinBox)

from PyQt5.QtGui import QIcon
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtCore import Qt, QAbstractTableModel, QVariant, QModelIndex
import sys
import shelve



class Main(QMainWindow):
    def __init__(self):
        super(Main, self).__init__()
        self.Title = "File Browser"
        self.initUi()

    def initUi(self):
        self.i=0
        self.catDict=dict()
        self.setWindowTitle(self.Title)
        self.main_widget = QWidget(self)
        self.layout = QGridLayout(self.main_widget)

        # main button
        # self.addButton = QPushButton('button to add other widgets')
        # self.addButton.clicked.connect(self.addWidget)
        # self.delButton = QPushButton('button to del other widgets')
        # self.delButton.clicked.connect(self.delWidget)

        # scroll area widget contents - layout
        # self.scrollLayout = QFormLayout()
        #
        # # scroll area widget contents
        # self.scrollWidget = QWidget()
        # self.scrollWidget.setLayout(self.scrollLayout)
        #
        # # scroll area
        # self.scrollArea = QScrollArea()
        # self.scrollArea.setWidgetResizable(True)
        # self.scrollArea.setWidget(self.scrollWidget)

        # main layout
        # self.mainLayout = QGridLayout()



        self.catCurrentCount = 0
        self.catDict = dict()
        self.catCount = QSpinBox()
        self.catCount.setRange(2, 5)
        self.catCount.setValue(2)
        self.catCount.valueChanged.connect(self.catCountUpdate)

        self.catScrollLayout = QFormLayout()

        # scroll area widget contents
        self.catScrollWidget = QWidget()
        self.catScrollWidget.setLayout(self.catScrollLayout)

        # scroll area
        self.catScrollArea = QScrollArea()
        self.catScrollArea.setWidgetResizable(True)
        self.catScrollArea.setWidget(self.catScrollWidget)

        # add all main to the main vLayout
        self.layout.addWidget(QLabel("Choose number of categories:"), 0, 0, 1, 1)
        self.layout.addWidget(self.catCount, 0, 1, 1, 1)
        self.layout.addWidget(self.catScrollArea, 1, 0, 7, 2)


        # central widget
        # self.centralWidget = QWidget()
        # self.centralWidget.setLayout(self.mainLayout)

        # set central widget
        self.setCentralWidget(self.main_widget)
        self.resize(1800, 900)
        self.show()
        self.addWidget()
        self.addWidget()

    def catCountUpdate(self):
        if self.catCount.value() > self.catCurrentCount:
            while (self.catCount.value() > self.catCurrentCount):
                #self.catCurrentCount += 1
                self.addWidget()
        elif self.catCount.value() < self.catCurrentCount:
            while (self.catCount.value() < self.catCurrentCount):
                self.delWidget()
                #self.catCurrentCount -= 1


    def addWidget(self):
        self.catCurrentCount += 1
        s = "s" + str(self.catCurrentCount) +"Layout"
        self.catDict[s] = QLineEdit("Enter Category " + str(self.catCurrentCount))
        self.catDict[s].setEnabled(True)
        # self.s=QLineEdit('I am in Test widget')
        self.catScrollLayout.addRow(self.catDict[s])

        # self.i+=1
        # self.txtEstimatorCount.setText("35")


    def delWidget(self):
        t = "s" + str(self.catCurrentCount)
        print(self.catDict[t+"Layout"].text())
        self.catDict[t+"Layout"].deleteLater()
        self.catCurrentCount -= 1
        # to_delete = self.scrollLayout.takeAt(self.scrollLayout.count() - 1)
        # if to_delete is not None:
        #     while to_delete.count():
        #         item = to_delete.takeAt(0)
        #         widget = item.widget()
        #         if widget is not None:
        #             widget.deleteLater()
        #         else:
        #             pass



        # t="s"+str(self.i)
        # self.scrollLayout.removeWidget(self.variablesDict[t])
        # self.i -= 1

# class Test(QWidget):
#     def __init__(self, parent=None):
#         super(Test, self).__init__(parent)

#         self.pushButton = QPushButton('I am in Test widget')

#         layout = QHBoxLayout()
#         layout.addWidget(self.pushButton)
#         self.setLayout(layout)

app = QApplication(sys.argv)
myWidget = Main()
myWidget.show()
app.exec_()



import pandas as pd
df=pd.read_csv("HR-Employee-Attrition.csv")
print(df.columns)
list=df.dtypes.tolist()

print(list)
dataTypeDict={"int64":"Continuous","object":"Categorical"}
for val in list:
    print(dataTypeDict[str(val)])

'''

# !/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
from PyQt5.QtGui import QIcon

from PyQt5.QtCore import (QDate, QDateTime, QRegExp, QSortFilterProxyModel, Qt,
                          QTime)
from PyQt5.QtGui import QStandardItemModel
from PyQt5.QtWidgets import (QApplication, QCheckBox, QComboBox, QGridLayout,
                             QGroupBox, QHBoxLayout, QLabel, QLineEdit, QTreeView, QVBoxLayout,
                             QWidget)


class App(QWidget):
    FROM, SUBJECT, DATE = range(3)

    def __init__(self):
        super().__init__()
        self.title = 'PyQt5 Treeview Example - pythonspot.com'
        self.left = 10
        self.top = 10
        self.width = 640
        self.height = 240
        self.initUI()

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        self.dataGroupBox = QGroupBox("Inbox")
        self.dataView = QTreeView()
        self.dataView.setRootIsDecorated(False)
        # self.dataView.setAlternatingRowColors(True)

        dataLayout = QHBoxLayout()
        dataLayout.addWidget(self.dataView)
        self.dataGroupBox.setLayout(dataLayout)

        # model = self.createMailModel(self)
        self.model=QStandardItemModel(0, 3)
        self.model.setHeaderData(0, Qt.Horizontal, "From")
        self.model.setHeaderData(1, Qt.Horizontal, "Subject")
        self.model.setHeaderData(2, Qt.Horizontal, "Date")
        self.dataView.setModel(self.model)
        self.addMail('service@github.com', 'Your Github Donation', '03/25/2017 02:05 PM')
        self.addMail('support@github.com', 'Github Projects', '02/02/2017 03:05 PM')
        self.addMail('service@phone.com', 'Your Phone Bill', '01/01/2017 04:05 PM')
        self.addMail('service@github.com', 'Your Github Donation', '03/25/2017 02:05 PM')
        self.addMail('support@github.com', 'Github Projects', '02/02/2017 03:05 PM')
        self.addMail('service@phone.com', 'Your Phone Bill', '01/01/2017 04:05 PM')
        self.addMail('service@github.com', 'Your Github Donation', '03/25/2017 02:05 PM')
        self.addMail('support@github.com', 'Github Projects', '02/02/2017 03:05 PM')
        self.addMail('service@phone.com', 'Your Phone Bill', '01/01/2017 04:05 PM')
        self.addMail('service@github.com', 'Your Github Donation', '03/25/2017 02:05 PM')
        self.addMail('support@github.com', 'Github Projects', '02/02/2017 03:05 PM')
        self.addMail('service@phone.com', 'Your Phone Bill', '01/01/2017 04:05 PM')

        mainLayout = QVBoxLayout()
        mainLayout.addWidget(self.dataGroupBox)
        self.setLayout(mainLayout)

        self.show()

    # def createMailModel(self, parent):
    #     model = QStandardItemModel(0, 3, parent)
    #     model.setHeaderData(self.FROM, Qt.Horizontal, "From")
    #     model.setHeaderData(self.SUBJECT, Qt.Horizontal, "Subject")
    #     model.setHeaderData(self.DATE, Qt.Horizontal, "Date")
    #     return model

    def addMail(self, mailFrom, subject, date):
        self.model.insertRow(0)
        self.model.setData(self.model.index(0, 0), mailFrom)
        self.model.setData(self.model.index(0, 1), subject)
        self.model.setData(self.model.index(0, 2), date)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())




