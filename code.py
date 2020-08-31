import sys,os
required_packages=["PyQt5","scipy","itertools","random","matplotlib","pandas","numpy","sklearn","graphviz","pydotplus","collections","warnings","seaborn"]

# for my_package in required_packages:
#     try:
#         command_string="conda install "+ my_package+ " --yes"
#         os.system(command_string)
#     except:
#         count=1

import shelve
import pickle

from PyQt5.QtWidgets import (QMainWindow, QApplication, QWidget, QPushButton, QAction, QComboBox, QLabel, QFrame,
                             QGridLayout, QCheckBox, QGroupBox, QVBoxLayout, QHBoxLayout, QLineEdit, QPlainTextEdit,
                             QInputDialog, QFileDialog, QTableView, QSpinBox)

from PyQt5.QtGui import QIcon
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtCore import Qt, QAbstractTableModel, QVariant, QModelIndex

from itertools import cycle
import random
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QSizePolicy, QFormLayout, QRadioButton, QScrollArea, QMessageBox
from PyQt5.QtGui import QPixmap

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import pandas as pd
import numpy as np
from numpy.polynomial.polynomial import polyfit

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc, log_loss, brier_score_loss
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn import feature_selection
from sklearn import metrics
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import label_binarize

# Libraries to display decision tree
from pydotplus import graph_from_dot_data
import collections
#from sklearn.tree import export_graphviz


import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
from textwrap import wrap

import random
import seaborn as sns

#%%-----------------------------------------------------------------------
import os
os.environ["PATH"] += os.pathsep + 'C:\\Program Files (x86)\\graphviz-2.38\\release\\bin'
#%%-----------------------------------------------------------------------


#::--------------------------------
# Deafault font size for all the windows
#::--------------------------------
font_size_window = 'font-size:15px'
font_size_model = 'font-size:12px'

#data=pd.DataFrame()



global data
global featuresList
global featuresDtypes
global ratioFeatures
global ordinalFeatures
global nominalFeatures
global featureCategoryMapping
global categoryNameList
global fileNameGlobal
global featureTypeOptions
global dataTypeDict
global categoryNameList
global categoryNameVariables
global targetClasses
global targetVariable


class PandasModel(QAbstractTableModel):
    def __init__(self, df = pd.DataFrame(), parent=None):
        QAbstractTableModel.__init__(self, parent=parent)
        self._df = df

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role != Qt.DisplayRole:
            return QVariant()

        if orientation == Qt.Horizontal:
            try:
                return self._df.columns.tolist()[section]
            except (IndexError, ):
                return QVariant()
        elif orientation == Qt.Vertical:
            try:
                # return self.df.index.tolist()
                return self._df.index.tolist()[section]
            except (IndexError, ):
                return QVariant()

    def data(self, index, role=Qt.DisplayRole):
        if role != Qt.DisplayRole:
            return QVariant()

        if not index.isValid():
            return QVariant()

        return QVariant(str(self._df.iloc[index.row(), index.column()]))

    def setData(self, index, value, role):
        row = self._df.index[index.row()]
        col = self._df.columns[index.column()]
        if hasattr(value, 'toPyObject'):
            # PyQt4 gets a QVariant
            value = value.toPyObject()
        else:
            # PySide gets an unicode
            dtype = self._df[col].dtype
            if dtype != object:
                value = None if value == '' else dtype.type(value)
        self._df.set_value(row, col, value)
        return True

    def rowCount(self, parent=QModelIndex()):
        return len(self._df.index)

    def columnCount(self, parent=QModelIndex()):
        return len(self._df.columns)

    def sort(self, column, order):
        colname = self._df.columns.tolist()[column]
        self.layoutAboutToBeChanged.emit()
        self._df.sort_values(colname, ascending= order == Qt.AscendingOrder, inplace=True)
        self._df.reset_index(inplace=True, drop=True)
        self.layoutChanged.emit()


class showTable(QMainWindow):

    def __init__(self):
        super(showTable, self).__init__()
        self.Title = "File Browser"
        self.initUi()

    def initUi(self):
        self.setWindowTitle(self.Title)
        self.setStyleSheet(font_size_window)
        self.main_widget = QWidget(self)
        self.layout = QGridLayout(self.main_widget)
        # self.vLayout = QVBoxLayout()
        # self.hLayout = QHBoxLayout()
        self.pathLE = QLineEdit()
        # self.hLayout.addWidget(self.pathLE)
        self.loadBtn = QPushButton("Select File")

        #self.hLayout.addWidget(self.loadBtn)
        #self.vLayout.addLayout(self.hLayout)
        self.pandasTv = QTableView()
        self.rowCount = QComboBox()
        self.rowCount.addItems(["100", "500", "1000", "5000"])
        self.rowCount.currentIndexChanged.connect(self.showData)
        #self.vLayout.addWidget(self.pandasTv)
        self.layout.addWidget(self.loadBtn, 0, 0, 1, 1)
        self.layout.addWidget(self.pathLE, 0, 1, 1, 5)
        self.layout.addWidget(QLabel(""),0,6,1,1)
        self.layout.addWidget(QLabel("Show Rows:"), 0,7 ,1,1)
        self.layout.addWidget(self.rowCount, 0, 8, 1, 1)

        self.layout.addWidget(self.pandasTv, 1, 0, 9, 9)
        #self.loadFile()
        self.loadBtn.clicked.connect(self.loadFile)
        self.pandasTv.setSortingEnabled(True)
        self.setCentralWidget(self.main_widget)
        self.resize(1800, 900)
        self.show()




    def loadFile(self):
        global data
        global featuresList
        global featuresDtypes
        global ratioFeatures
        global ordinalFeatures
        global nominalFeatures
        global featureCategoryMapping
        global categoryNameList
        global fileNameGlobal


        ratioFeatures=[]
        ordinalFeatures=[]
        nominalFeatures=[]
        featureCategoryMapping=dict()
        categoryNameList=["No Category"]

        fileName, _ = QFileDialog.getOpenFileName(self, "Open File", "", "CSV Files (*.csv)");
        fileNameGlobal = fileName
        self.pathLE.setText(fileName)
        self.df = pd.read_csv(fileName)
        data=self.df.copy()
        featuresList = data.columns.tolist()
        featuresDtypes=data.dtypes.tolist()
        for i in range(len(featuresList)):
            if (str(featuresDtypes[i])=="int64"):
                ratioFeatures.append(featuresList[i])
            else:
                nominalFeatures.append(featuresList[i])
        self.showData()
        #print(df.head())
    def showData(self):
        try:
            model = PandasModel(self.df.head(int(self.rowCount.currentText())))
            #print(model)
            self.pandasTv.setModel(model)
        except:
            pass

class viewData(QMainWindow):

    def __init__(self):
        super(viewData, self).__init__()
        self.Title = "Data"
        self.initUi()

    def initUi(self):
        self.setWindowTitle(self.Title)
        self.setStyleSheet(font_size_window)
        self.main_widget = QWidget(self)
        self.layout = QGridLayout(self.main_widget)
        # self.vLayout = QVBoxLayout()
        # self.hLayout = QHBoxLayout()
        self.pathLE = QLineEdit()
        self.pathLE.setText(fileNameGlobal)
        self.pathLE.setEnabled(False)
        # self.hLayout.addWidget(self.pathLE)
        #self.loadBtn = QPushButton("Select File")

        #self.hLayout.addWidget(self.loadBtn)
        #self.vLayout.addLayout(self.hLayout)
        self.pandasTv = QTableView()
        self.rowCount = QComboBox()
        self.rowCount.addItems(["100", "500", "1000", "5000"])
        self.rowCount.currentIndexChanged.connect(self.showData)
        #self.vLayout.addWidget(self.pandasTv)
        self.layout.addWidget(QLabel("File Source:"), 0, 0, 1, 1)
        self.layout.addWidget(self.pathLE, 0, 1, 1, 5)
        self.layout.addWidget(QLabel(""), 0, 6, 1, 1)
        self.layout.addWidget(QLabel("Show Rows:"), 0,7 ,1,1)
        self.layout.addWidget(self.rowCount, 0, 8, 1, 1)

        self.layout.addWidget(self.pandasTv, 1, 0, 9, 9)
        #self.loadFile()
        self.pandasTv.setSortingEnabled(True)
        self.setCentralWidget(self.main_widget)
        self.resize(1800, 900)
        self.show()
        self.showData()

    def showData(self):
        try:
            self.df=data.copy()
            model = PandasModel(self.df.head(int(self.rowCount.currentText())))
            #print(model)
            self.pandasTv.setModel(model)
        except:
            print("Error")


class VariableInformation(QMainWindow):
    #::---------------------------------------------------------
    # This class creates a canvas with a plot to show the
    # distribution of continuous features in the dataset
    #::---------------------------------------------------------
    send_fig = pyqtSignal(str)

    def __init__(self):

        super(VariableInformation, self).__init__()

        self.Title = "Variable Information"
        self.main_widget = QWidget(self)

        self.catWidgetStatus = False
        self.catNameWidgetStatus = False
        self.featureWidgetStatus = False
        self.featureCatWidgetStatus = False

        self.setWindowTitle(self.Title)
        self.setStyleSheet(font_size_window)
        self.layout = QGridLayout(self.main_widget)

        self.viewWidget=QGroupBox('Categorise Variables')
        self.viewWidgetLayout = QGridLayout()
        self.viewWidget.setLayout(self.viewWidgetLayout)

        self.viewDataWidget=QPushButton("View Data")
        self.viewDataWidget.clicked.connect(self.viewDataWidgetFunction)

        self.catWidget = QGroupBox('Categorise Variables')
        self.catWidgetLayout = QGridLayout()
        self.catWidget.setLayout(self.catWidgetLayout)

        self.selectTargetWidget = QGroupBox('Select Target Variable')
        self.selectTargetWidgetLayout = QGridLayout()
        self.selectTargetWidget.setLayout(self.selectTargetWidgetLayout)

        self.featureDetailWidget = QGroupBox('Feature Details')
        self.featureDetailWidgetLayout = QGridLayout()
        self.featureDetailWidget.setLayout(self.featureDetailWidgetLayout)

        self.catCheck = QComboBox()
        self.catCheck.addItems(["No","Yes"])
        self.catCheck.currentIndexChanged.connect(self.catCheckUpdate)

        self.chooseTarget = QComboBox()
        self.chooseTarget.addItems([""])

        self.catCurrentCount = 0
        self.catDict = dict()
        self.catCount = QSpinBox()
        self.catCount.setRange(2, 5)
        self.catCount.setValue(2)
        self.catCount.valueChanged.connect(self.catCountUpdate)
        self.catCount.setEnabled(self.catNameWidgetStatus)


        self.catScrollLayout = QFormLayout()

        self.featureScrollLayout = QFormLayout()

        # scroll area widget contents
        self.catScrollWidget = QWidget()
        self.catScrollWidget.setLayout(self.catScrollLayout)
        self.catScrollWidget.setEnabled(self.catNameWidgetStatus)

        self.featureScrollWidget = QWidget()
        self.featureScrollWidget.setLayout(self.featureScrollLayout)

        # scroll area
        self.catScrollArea = QScrollArea()
        self.catScrollArea.setWidgetResizable(True)
        self.catScrollArea.setWidget(self.catScrollWidget)

        self.featureScrollArea = QScrollArea()
        self.featureScrollArea.setWidgetResizable(True)
        self.featureScrollArea.setWidget(self.featureScrollWidget)


        self.goBackToCategory = QPushButton("Back")
        self.goBackToCategory.clicked.connect(self.categoryWidgetFunction)

        self.goBackToFeature = QPushButton("Back")
        self.goBackToFeature.clicked.connect(self.featureWidgetFunction)

        self.viewFeatureDetails = QPushButton("Proceed")
        self.viewFeatureDetails.clicked.connect(self.featureWidgetFunction)

        self.targetVariableSelection = QPushButton("Proceed")
        self.targetVariableSelection.clicked.connect(self.targetWidgetFunction)


        self.saveInfo = QPushButton("Save and Exit")
        self.saveInfo.clicked.connect(self.saveVariableDetailsFunction)


        # add all main to the main vLayout
        self.viewWidgetLayout.addWidget(QLabel("Click to View Data:"), 0, 0, 2, 2)
        self.viewWidgetLayout.addWidget(self.viewDataWidget, 0, 2, 2, 2)


        self.catWidgetLayout.addWidget(QLabel("Add Categories:"), 0, 0, 1, 1)
        self.catWidgetLayout.addWidget(self.catCheck, 0, 1, 1, 1)
        self.catWidgetLayout.addWidget(QLabel("Choose number of categories:"), 1, 0, 1, 1)
        self.catWidgetLayout.addWidget(self.catCount, 1, 1, 1, 1)
        self.catWidgetLayout.addWidget(self.catScrollArea, 2, 0, 7, 2)
        self.catWidgetLayout.addWidget(self.viewFeatureDetails, 9, 0, 1, 2)
        self.catWidget.setEnabled(True)

        self.selectTargetWidgetLayout.addWidget(QLabel("Choose Target Variable:"), 0, 0, 1, 1)
        self.selectTargetWidgetLayout.addWidget(self.chooseTarget, 0, 1, 1, 1)
        self.selectTargetWidgetLayout.addWidget(self.goBackToFeature, 1, 0, 1, 1)
        self.selectTargetWidgetLayout.addWidget(self.saveInfo, 1, 1, 1, 1)
        self.selectTargetWidget.setEnabled(False)

        self.featureDetailWidgetLayout.addWidget(self.featureScrollArea, 0, 0, 9, 4)
        self.featureDetailWidgetLayout.addWidget(self.goBackToCategory, 9, 0, 1, 2)
        self.featureDetailWidgetLayout.addWidget(self.targetVariableSelection, 9, 2, 1, 2)
        self.featureDetailWidget.setEnabled(False)


        self.layout.addWidget(self.viewWidget, 0, 0, 2, 2)
        self.layout.addWidget(self.catWidget, 2, 0, 6, 2)
        self.layout.addWidget(self.selectTargetWidget, 8, 0, 2, 2)
        self.layout.addWidget(self.featureDetailWidget, 0, 3, 10, 4)




        self.setCentralWidget(self.main_widget)
        self.resize(1800, 900)
        self.show()
        self.addWidget()
        self.addWidget()

        self.flag=0

    def viewDataWidgetFunction(self):
        self.w = viewData()
        self.w.show()

    def catCheckUpdate(self):
        if (self.catCheck.currentText() == "Yes"):
            self.catNameWidgetStatus = True

        else:
            self.catNameWidgetStatus = False
        self.catCount.setEnabled(self.catNameWidgetStatus)
        self.catScrollWidget.setEnabled(self.catNameWidgetStatus)


    def catCountUpdate(self):
        if self.catCount.value() > self.catCurrentCount:
            while (self.catCount.value() > self.catCurrentCount):
                self.addWidget()
        elif self.catCount.value() < self.catCurrentCount:
            while (self.catCount.value() < self.catCurrentCount):
                self.delWidget()

    def addWidget(self):
        self.catCurrentCount += 1
        s = "s" + str(self.catCurrentCount)
        self.catDict[s] = QLineEdit("Enter Category " + str(self.catCurrentCount))
        self.catScrollLayout.addRow(self.catDict[s])

    def delWidget(self):
        t = "s" + str(self.catCurrentCount)
        self.catDict[t].deleteLater()
        self.catCurrentCount -= 1


    def categoryWidgetFunction(self):
        #self.featureWidgetStatus=True

        self.selectTargetWidget.setEnabled(False)
        self.featureDetailWidget.setEnabled(False)
        self.catWidget.setEnabled(True)

    def featureWidgetFunction(self):
        #self.featureWidgetStatus=True
        self.catWidget.setEnabled(False)
        self.selectTargetWidget.setEnabled(False)
        self.featureDetailWidget.setEnabled(True)
        if (self.flag==1):
            for j in range(len(featuresList)):
                self.featureDict["f" + str(j)].deleteLater()
        global featureTypeOptions
        global dataTypeDict
        global categoryNameList
        dataTypeDict = {"int64": "Continuous","float64": "Continuous", "object": "Categorical"}
        if (self.catCheck.currentText()=="Yes"):
            self.featureCatWidgetStatus=True
            categoryNameList=[]
            for i in range(self.catCount.value()):
                s = "s" + str(i+1)
                print(s,self.catDict[s].text())
                categoryNameList.append(self.catDict[s].text())
        else:
            self.featureCatWidgetStatus=False
            categoryNameList=["No Category"]
        print(categoryNameList)
        self.featureCurrentCount = 0
        self.featureDict = dict()
        for currentFeature in featuresList:
            s = "f" + str(self.featureCurrentCount)

            self.featureDict[s] = QGroupBox()
            self.featureDict[s+"Layout"] = QGridLayout()
            self.featureDict[s].setLayout(self.featureDict[s+"Layout"])

            self.featureDict[s + "Label"] = QLabel(str(currentFeature))
            self.featureDict[s + "TypeList"] = QComboBox()
            self.featureDict[s + "TypeList"].addItems(["Continuous", "Categorical", "Ordinal"])
            self.featureDict[s + "TypeList"].setCurrentText(dataTypeDict[str(featuresDtypes[self.featureCurrentCount])])

            self.featureDict[s + "catList"] = QComboBox()
            self.featureDict[s + "catList"].addItems(categoryNameList)
            self.featureDict[s + "catList"].setEnabled(self.featureCatWidgetStatus)

            self.featureDict[s + "Layout"].addWidget(self.featureDict[s + "Label"], 0, 0, 1, 2)
            self.featureDict[s + "Layout"].addWidget(self.featureDict[s + "TypeList"], 0, 2, 1, 1)
            self.featureDict[s + "Layout"].addWidget(self.featureDict[s + "catList"], 0, 3, 1, 1)

            self.featureScrollLayout.addRow(self.featureDict[s])
            self.featureCurrentCount += 1
        self.flag=1

    def targetWidgetFunction(self):
        #self.featureWidgetStatus=True
        self.catWidget.setEnabled(False)
        self.featureDetailWidget.setEnabled(False)
        self.saveInfo.setEnabled(False)
        self.selectTargetWidget.setEnabled(True)
        global ratioFeatures
        global ordinalFeatures
        global nominalFeatures

        ratioFeatures = []
        nominalFeatures = []
        ordinalFeatures = []
        featureCategoryMapping = dict()
        global categoryNameVariables
        categoryNameVariables = dict()
        print(categoryNameList)
        for val in categoryNameList:
            categoryNameVariables[val] = []

        for k in range(len(featuresList)):
            a = "f" + str(k)
            if (self.featureDict[a + "TypeList"].currentText() == "Continuous"):
                ratioFeatures.append(featuresList[k])
            elif (self.featureDict[a + "TypeList"].currentText() == "Categorical"):
                nominalFeatures.append(featuresList[k])
            else:
                ordinalFeatures.append(featuresList[k])

            categoryNameVariables[self.featureDict[a + "catList"].currentText()].append(featuresList[k])
                # featureCategoryMapping[featuresList[k]]=self.featureDict[a + "catList"].currentText()

        if len(nominalFeatures)>0:
            self.chooseTarget.clear()
            self.chooseTarget.addItems(nominalFeatures)
            self.saveInfo.setEnabled(True)
        else:
            self.chooseTarget.clear()
            self.chooseTarget.addItems(["No Categorical Variable"])

        # print("Ratio Features:", ratioFeatures)
        # print("Nominal Features:", nominalFeatures)
        # print("Ordinal Features:", ordinalFeatures)

    def saveVariableDetailsFunction(self):
        global ratioFeatures
        global ordinalFeatures
        global nominalFeatures
        global targetClasses
        global targetVariable
        global featuresList
        targetVariable=self.chooseTarget.currentText()
        targetClasses=data[targetVariable].unique()
        targetClasses=sorted(list(targetClasses))
        featuresList.remove(targetVariable)
        if targetVariable in ratioFeatures:
            ratioFeatures.remove(targetVariable)
        elif targetVariable in nominalFeatures:
            nominalFeatures.remove(targetVariable)
        else:
            ordinalFeatures.remove(targetVariable)


        #print(targetVariable,targetClasses)
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getSaveFileName(self, "QFileDialog.getSaveFileName()", "",
                                                  "Shelve Files (*.out)", options=options)
        if fileName:
            #print(fileName)
            #fileName += '.out'
            my_shelf = shelve.open(fileName, 'n')  # 'n' for new
            globalKeys = list(globals().keys())
            for key in globalKeys:
                #print(key)
                try:
                    my_shelf[key] = globals()[key]
                except TypeError:
                    i=1
                    #
                    # __builtins__, my_shelf, and imported modules can not be shelved.
                    #
                    #print('ERROR shelving: {0}'.format(key))
            my_shelf.close()
        self.close()




class VariableDistribution(QMainWindow):
    #::---------------------------------------------------------
    # This class creates a canvas with a plot to show the
    # distribution of continuous features in the dataset
    #::---------------------------------------------------------
    send_fig = pyqtSignal(str)

    def __init__(self):
        super(VariableDistribution, self).__init__()
        print(categoryNameList)
        if (len(categoryNameList)>1):
            self.catWidgetStatus=True
        else:
            self.catWidgetStatus = False

        self.filterDataItemList=["All Data"]
        self.filterDataItemList+= [targetVariable + ": " + s for s in targetClasses]

        self.Title = "EDA: Variable Distribution"
        self.main_widget = QWidget(self)

        self.setWindowTitle(self.Title)
        self.setStyleSheet(font_size_window)

        self.fig = Figure()
        self.ax = self.fig.add_subplot(111)
        self.axes=[self.ax]
        self.canvas = FigureCanvas(self.fig)

        self.canvas.setSizePolicy(QSizePolicy.Expanding,QSizePolicy.Expanding)

        print("Ratio Features:", ratioFeatures)
        print("Nominal Features:", nominalFeatures)
        print("Ordinal Features:", ordinalFeatures)

        self.canvas.updateGeometry()
        self.featuresList=list(set(ratioFeatures) & set(categoryNameVariables[categoryNameList[0]]))
        self.dropdown1 = QComboBox()
        self.dropdown1.addItems(categoryNameList)
        self.dropdown1.currentIndexChanged.connect(self.updateCategory)
        self.dropdown1.setEnabled(self.catWidgetStatus)
        self.dropdown2 = QComboBox()
        self.label = QLabel("A plot:")

        self.chooseFeature = QWidget(self)
        self.chooseFeature.layout = QGridLayout(self.chooseFeature)

        self.chooseFeature.layout.addWidget(QLabel("Select Feature Category:"), 0, 0, 1, 1)
        self.chooseFeature.layout.addWidget(self.dropdown1, 0, 1, 1, 1)
        self.chooseFeature.layout.addWidget(QLabel("Select Features:"), 1, 0, 1, 1)
        self.chooseFeature.layout.addWidget(self.dropdown2, 1, 1, 1, 1)


        self.filter_data = QWidget(self)
        self.filter_data.layout = QGridLayout(self.filter_data)

        self.filter_data.layout.addWidget(QLabel("Choose Data Filter:"), 0, 0, 1, 1)

        self.filterDropDown = QComboBox()
        self.filterDropDown.addItems(self.filterDataItemList)
        self.filterDropDown.currentIndexChanged.connect(self.update)
        self.filter_data.layout.addWidget(self.filterDropDown, 0, 1, 1, 3)

        self.btnCreateGraph = QPushButton("Create Graph")
        self.btnCreateGraph.clicked.connect(self.update)
        self.filter_data.layout.addWidget(self.btnCreateGraph, 1, 0, 1, 4)

        self.groupBox1 = QGroupBox('Distribution')
        self.groupBox1Layout = QVBoxLayout()
        self.groupBox1.setLayout(self.groupBox1Layout)
        self.groupBox1Layout.addWidget(self.canvas)

        self.groupBox2 = QGroupBox('Summary')
        self.groupBox2Layout = QVBoxLayout()
        self.groupBox2.setLayout(self.groupBox2Layout)
        self.graph_summary = QPlainTextEdit()
        self.groupBox2Layout.addWidget(self.graph_summary)

        self.layout = QGridLayout(self.main_widget)


        self.layout.addWidget(self.chooseFeature, 0, 0, 2, 2)
        self.layout.addWidget(QLabel(""), 0, 2, 2, 1)
        self.layout.addWidget(self.filter_data, 0, 3, 2, 2)
        self.layout.addWidget(self.groupBox1, 2, 0, 5, 5)

        self.setCentralWidget(self.main_widget)
        self.resize(1200, 700)
        self.show()
        self.updateCategory()

    def updateCategory(self):
        self.dropdown2.clear()
        feature_category = self.dropdown1.currentText()
        self.featuresList = list(set(ratioFeatures) & set(categoryNameVariables[feature_category]))
        del feature_category
        self.dropdown2.addItems(self.featuresList)
        del self.featuresList

    # def onFilterClicked(self):
    #     self.filter_radio_button = self.sender()
    #     if self.filter_radio_button.isChecked():
    #         self.set_Filter=self.filter_radio_button.filter
    #         self.update()

    def update(self):
        colors=["b", "r", "g", "y", "k", "c"]
        self.ax.clear()
        cat1 = self.dropdown2.currentText()
        self.setFilter=self.filterDropDown.currentText()
        self.setFilter = self.setFilter.replace(targetVariable+': ', '')
        if (self.setFilter in targetClasses):
            self.filtered_data=data.copy()
            self.filtered_data = self.filtered_data[self.filtered_data[targetVariable]==self.setFilter]
        else:
            self.filtered_data = data.copy()

        self.ax.hist(self.filtered_data[cat1], bins=10, facecolor='green', alpha=0.5)
        self.ax.set_title(cat1)
        self.ax.set_xlabel(cat1)
        self.ax.set_ylabel("Count")
        self.ax.grid(True)
        self.fig.tight_layout()
        self.fig.canvas.draw_idle()
        del cat1
        del self.filtered_data


class VariableRelation(QMainWindow):
    #::---------------------------------------------------------
    # This class creates a canvas with a plot to show the relation
    # between a continuous - continuous variable (scatter plot) and
    # categorical - continuous variable (box plot)
    #::---------------------------------------------------------
    send_fig = pyqtSignal(str)

    def __init__(self):
        super(VariableRelation, self).__init__()
        if (len(categoryNameList)>1):
            self.catWidgetStatus=True
        else:
            self.catWidgetStatus = False

        self.filterDataItemList = ["All Data"]
        self.filterDataItemList += [targetVariable + ": " + s for s in targetClasses]

        self.Title = "EDA: Variable Relation"
        self.main_widget = QWidget(self)

        self.setWindowTitle(self.Title)
        self.setStyleSheet(font_size_window)

        self.fig = Figure()
        self.ax = self.fig.add_subplot(111)
        self.axes=[self.ax]
        self.canvas = FigureCanvas(self.fig)

        self.canvas.setSizePolicy(QSizePolicy.Expanding,QSizePolicy.Expanding)

        self.canvas.updateGeometry()
        self.featuresList1 = categoryNameVariables[categoryNameList[0]]
        self.featuresList2 = categoryNameVariables[categoryNameList[0]]

        self.filterBox1 = QGroupBox('Feature 1')
        self.filterBox1Layout = QGridLayout()
        self.filterBox1.setLayout(self.filterBox1Layout)

        self.dropdown1 = QComboBox()
        self.dropdown1.addItems(categoryNameList)
        self.dropdown1.currentIndexChanged.connect(self.updateCategory1)
        self.dropdown1.setEnabled(self.catWidgetStatus)
        self.dropdown2 = QComboBox()
        self.dropdown2.addItems(self.featuresList1)
        self.dropdown2.currentIndexChanged.connect(self.checkifsame)
        self.dropdown2.setEnabled(self.catWidgetStatus)
        self.filterBox1Layout.addWidget(QLabel("Select Feature Category:"),0,0)
        self.filterBox1Layout.addWidget(self.dropdown1,0,1)
        self.filterBox1Layout.addWidget(QLabel("Select Feature:"),1,0)
        self.filterBox1Layout.addWidget(self.dropdown2,1,1)

        self.filterBox2 = QGroupBox('Feature 2')
        self.filterBox2Layout = QGridLayout()
        self.filterBox2.setLayout(self.filterBox2Layout)

        self.dropdown3 = QComboBox()
        self.dropdown3.addItems(categoryNameList)
        self.dropdown3.currentIndexChanged.connect(self.updateCategory2)
        self.dropdown4 = QComboBox()
        self.dropdown4.addItems(self.featuresList2)
        self.filterBox2Layout.addWidget(QLabel("Select Feature Category:"),0,0)
        self.filterBox2Layout.addWidget(self.dropdown3,0,1)
        self.filterBox2Layout.addWidget(QLabel("Select Feature:"),1,0)
        self.filterBox2Layout.addWidget(self.dropdown4,1,1)

        self.filter_data = QWidget(self)
        self.filter_data.layout = QGridLayout(self.filter_data)

        self.filter_data.layout.addWidget(QLabel("Choose Data Filter:"), 0, 0, 1, 1)

        self.filterDropDown = QComboBox()
        self.filterDropDown.addItems(self.filterDataItemList)
        self.filterDropDown.currentIndexChanged.connect(self.update)
        self.filter_data.layout.addWidget(self.filterDropDown, 0, 1, 1, 3)

        # self.btnCreateGraph = QPushButton("Create Graph")
        # self.btnCreateGraph.clicked.connect(self.update)
        # self.filter_data.layout.addWidget(self.btnCreateGraph, 1, 0, 1, 4)

        self.groupBox1 = QGroupBox('Feature Relation')
        self.groupBox1Layout = QVBoxLayout()
        self.groupBox1.setLayout(self.groupBox1Layout)
        self.groupBox1Layout.addWidget(self.canvas)

        self.groupBox2 = QGroupBox('Summary')
        self.groupBox2Layout = QVBoxLayout()
        self.groupBox2.setLayout(self.groupBox2Layout)
        self.graph_summary = QPlainTextEdit()
        self.groupBox2Layout.addWidget(self.graph_summary)


        self.layout = QGridLayout(self.main_widget)
        self.layout.addWidget(self.filterBox1,0,0,2,2)
        self.layout.addWidget(QLabel(""), 0, 2, 2, 1)
        self.layout.addWidget(self.filterBox2,0,3,2,2)
        self.layout.addWidget(self.filter_data,2,0,1,2)
        self.layout.addWidget(QLabel(""), 2, 2, 1, 1)
        self.btnCreateGraph = QPushButton("Create Graph")
        self.btnCreateGraph.clicked.connect(self.update)
        self.layout.addWidget(self.btnCreateGraph, 2, 3, 1, 2)
        self.layout.addWidget(self.groupBox1,3,0,7,5)

        self.setCentralWidget(self.main_widget)
        self.resize(1200, 700)
        self.show()
        self.updateCategory2()

    def checkifsame(self):
        if(self.dropdown2.currentText() in ratioFeatures):
            type1="continuous"
        else:
            type1 = "categorical"

        if(self.dropdown2.currentText()==self.dropdown4.currentText()):
            self.updateCategory2()

        if(type1=="categorical"):
            self.ifbothcategroical()

    def ifbothcategroical(self):
        self.dropdown4.clear()
        feature_category2 = self.dropdown3.currentText()
        self.featuresList2 = list(set(ratioFeatures) & set(categoryNameVariables[feature_category2]))

        if (self.dropdown2.currentText() in self.featuresList2):
            self.featuresList2.remove(self.dropdown2.currentText())
        self.dropdown4.addItems(self.featuresList2)


    def updateCategory1(self):
        self.dropdown2.clear()
        feature_category1 = self.dropdown1.currentText()
        self.featuresList1=categoryNameVariables[feature_category1]
        self.dropdown2.addItems(self.featuresList1)


    def updateCategory2(self):
        self.dropdown4.clear()
        feature_category2 = self.dropdown3.currentText()
        self.featuresList2 = categoryNameVariables[feature_category2]

        if(self.dropdown2.currentText() in self.featuresList2):
            self.featuresList2.remove(self.dropdown2.currentText())
        self.dropdown4.addItems(self.featuresList2)
        self.checkifsame()


    # def onFilterClicked(self):
    #     self.filter_radio_button = self.sender()
    #     if self.filter_radio_button.isChecked():
    #         self.set_Filter=self.filter_radio_button.filter
    #         self.update()

    def update(self):
        colors=["b", "r", "g", "y", "k", "c"]
        self.ax.clear()
        self.setFilter = self.filterDropDown.currentText()
        self.setFilter = self.setFilter.replace(targetVariable + ': ', '')
        if (self.setFilter in targetClasses):
            self.filtered_data = data.copy()
            self.filtered_data = self.filtered_data[self.filtered_data[targetVariable] == self.setFilter]
        else:
            self.filtered_data = data.copy()

        graph_feature1 = self.dropdown2.currentText()
        graph_feature2 = self.dropdown4.currentText()

        if((graph_feature1 in ratioFeatures) and (graph_feature2 in ratioFeatures)):
            x_axis_data = self.filtered_data[graph_feature1]
            y_axis_data = self.filtered_data[graph_feature2]
            self.ax.scatter(x_axis_data, y_axis_data)
            b, m = polyfit(x_axis_data, y_axis_data, 1)
            self.ax.plot(x_axis_data, b + m * x_axis_data, '-', color="orange")

            # vtitle = graph_feature2 + " Vs " + graph_feature1
            # self.ax.set_title(vtitle)
            self.ax.set_xlabel(graph_feature1)
            self.ax.set_ylabel(graph_feature2)
            self.ax.grid(True)

        else:
            if(graph_feature1 in ratioFeatures):
                continuous_data=graph_feature1
                categorical_data=graph_feature2
            else:
                continuous_data = graph_feature2
                categorical_data = graph_feature1
            graph_data = self.filtered_data[[graph_feature1, graph_feature2]]
            my_pt = pd.pivot_table(graph_data, index=graph_data.index,columns=categorical_data, values=continuous_data,aggfunc=np.sum)
            my_pt = pd.DataFrame(my_pt.to_records())
            my_pt=my_pt.drop(columns=['index'])
            class_names_x = my_pt.columns.values.tolist()
            my_np = my_pt.values
            mask = ~np.isnan(my_np)
            my_np_2 = [d[m] for d, m in zip(my_np.T, mask.T)]
            class_names_x = my_pt.columns.values.tolist()
            self.ax.boxplot(my_np_2)
            self.ax.set_xlabel(categorical_data)
            self.ax.set_ylabel(continuous_data)
            self.ax.set_xticklabels(class_names_x)

        self.fig.tight_layout()
        self.fig.canvas.draw_idle()


class TargetRelation(QMainWindow):
    #::---------------------------------------------------------
    # This class creates a canvas with a plot to compare the
    # variables between Attrition:Yes and Attrition:No
    # For continuous variables, it shows the Minimum, Median,
    # Mean and Maximum Values. For categorical variables,
    # it shows the count of values for each distinct element.
    #::---------------------------------------------------------
    send_fig = pyqtSignal(str)

    def __init__(self):

        super(TargetRelation, self).__init__()
        if (len(categoryNameList)>1):
            self.catWidgetStatus=True
        else:
            self.catWidgetStatus = False

        self.Title = "EDA: Target Relation"
        self.main_widget = QWidget(self)

        self.setWindowTitle(self.Title)
        self.setStyleSheet(font_size_window)

        self.fig = Figure()
        self.ax = self.fig.add_subplot(1,1,1)
        self.axes=[self.ax]
        self.canvas = FigureCanvas(self.fig)

        self.canvas.setSizePolicy(QSizePolicy.Expanding,QSizePolicy.Expanding)

        self.canvas.updateGeometry()
        self.featuresList=categoryNameVariables[categoryNameList[0]]
        self.dropdown1 = QComboBox()
        self.dropdown1.addItems(categoryNameList)
        self.dropdown1.currentIndexChanged.connect(self.updateCategory)
        self.dropdown2 = QComboBox()
        self.label = QLabel("A plot:")
        self.filter_data = QWidget(self)
        self.filter_data.layout = QGridLayout(self.filter_data)
        self.btnCreateGraph = QPushButton("Create Graph")
        self.btnCreateGraph.clicked.connect(self.update)

        self.groupBox1 = QGroupBox('Relation between Target Classes')
        self.groupBox1Layout = QVBoxLayout()
        self.groupBox1.setLayout(self.groupBox1Layout)
        self.groupBox1Layout.addWidget(self.canvas)

        self.groupBox2 = QGroupBox('Summary')
        self.groupBox2Layout = QVBoxLayout()
        self.groupBox2.setLayout(self.groupBox2Layout)
        self.graph_summary = QPlainTextEdit()
        self.groupBox2Layout.addWidget(self.graph_summary)

        self.layout = QGridLayout(self.main_widget)
        self.layout.addWidget(QLabel("Select Feature Category:"),0,0,1,1)
        self.layout.addWidget(self.dropdown1,0,1,1,1)
        self.layout.addWidget(QLabel(""), 0, 2, 1, 1)
        self.layout.addWidget(QLabel("Select Features:"),0,3,1,1)
        self.layout.addWidget(self.dropdown2,0,4,1,1)
        self.layout.addWidget(self.btnCreateGraph, 1, 0, 1, 5)
        self.layout.addWidget(self.groupBox1,2,0,6,5)

        self.setCentralWidget(self.main_widget)
        self.resize(1200, 700)
        self.show()
        self.updateCategory()

    def updateCategory(self):
        self.dropdown2.clear()
        feature_category = self.dropdown1.currentText()
        self.featuresList=categoryNameVariables[feature_category]
        self.dropdown2.addItems(self.featuresList)

    def update(self):
        colors=["b", "r", "g", "y", "k", "c"]
        self.ax.clear()
        # self.ax2.clear()
        self.filtered_data = data.copy()
        graph_feature1 = self.dropdown2.currentText()
        category_values=[]
        #val1 = []
        # valList = []
        # if len(targetClasses)>2:
            # axis1Title = targetVariable + ": " + targetClasses[0]
            # axis2Title = targetVariable + ": " + targetClasses[1]
        flagCount = 0
        for variableName in targetClasses:
            valList = []
            if (graph_feature1 not in ratioFeatures):
                #category_values=targetClasses

                self.filtered_data["Count"] = 1
                self.filtered_data[graph_feature1] = self.filtered_data[graph_feature1].astype(str)
                category_values = self.filtered_data[graph_feature1].unique().tolist()

                tempData = self.filtered_data[self.filtered_data[targetVariable] == variableName]
                # tempData2 = self.filtered_data[self.filtered_data[targetVariable] == targetClasses[1]]

                my_pt = pd.pivot_table(tempData, index=graph_feature1, values="Count", aggfunc=np.sum)
                my_pt = pd.DataFrame(my_pt.to_records())
                my_dict = dict(zip(my_pt[graph_feature1], my_pt["Count"]))
                print(my_dict)
                # my_pt_no = pd.pivot_table(tempData2, index=graph_feature1, values="Count", aggfunc=np.sum)
                # my_pt_no = pd.DataFrame(my_pt_no.to_records())
                # my_dict_no = dict(zip(my_pt_no[graph_feature1], my_pt_no["Count"]))
                category_values=sorted(category_values)
                for temp_value in category_values:

                    if temp_value in (my_dict.keys()):
                        valList.append(my_dict[temp_value])
                    else:
                        valList.append(0)
                # print(valList)

            else:
                category_values = ["Max", "Median", "Mean", "Min"]

                tempData = self.filtered_data[self.filtered_data[targetVariable] == variableName]
                valList.append(tempData[graph_feature1].max())
                valList.append(round(tempData[graph_feature1].median(skipna=True), 1))
                valList.append(round(tempData[graph_feature1].mean(skipna=True), 1))
                valList.append(tempData[graph_feature1].min())
            print(category_values)
            print(valList)

            if (flagCount == 0):
                t1=self.ax.bar(category_values, valList, label=variableName)
                graphStack=valList.copy()
                hGraph = 0
                for r1 in t1:
                    plotValue = valList[hGraph]
                    h1 = graphStack[hGraph]
                    if plotValue >1000:
                        plotValue=str(round(plotValue/1000,1))+"k"
                    elif plotValue > 1000000:
                            plotValue = str(round(plotValue / 1000000, 1)) + "M"
                    else:
                        plotValue=str(plotValue)
                    self.ax.text(r1.get_x() + r1.get_width() / 2., h1 / 2., plotValue, ha="center", va="center",
                                 color="white")
                    hGraph += 1
                flagCount = 1
            else:
                t1=self.ax.bar(category_values, valList, bottom = graphStack, label=variableName)
                hGraph = 0
                for r1 in t1:
                    plotValue=valList[hGraph]
                    h1 = graphStack[hGraph] + plotValue/2.
                    if plotValue >1000:
                        plotValue=str(round(plotValue/1000,1))+"k"
                    elif plotValue > 1000000:
                            plotValue = str(round(plotValue / 1000000, 1)) + "M"
                    else:
                        plotValue=str(plotValue)
                    self.ax.text(r1.get_x() + r1.get_width() / 2., h1, plotValue, ha="center", va="center",
                                 color="white")
                    hGraph += 1
                graphStack = [graphStack[i] + valList[i] for i in range(len(valList))]

            # hGraph=0
            # for r1 in t1:
            #     h1 = graphStack[hGraph]
            #     self.ax.text(r1.get_x() + r1.get_width() / 2., h1 / 2., "%d" % h1, ha="center", va="center",
            #              color="white")
            #     print(r1)
            #     hGraph+=1
        # self.ax.set_title(axis1Title)
        self.ax.axis()
        self.ax.legend()
        self.ax.grid(False)

        self.fig.tight_layout()
        self.fig.canvas.draw_idle()


class RandomForest(QMainWindow):
    #::--------------------------------------------------------------------------------
    # Random Forest Classifier using the attrition dataset
    # the methods in this class are
    #       _init_ : initialize the class
    #       initUi : creates the canvas and all the elements in the canvas
    #       update : populates the elements of the canvas base on the parameters
    #               chosen by the user
    #::---------------------------------------------------------------------------------
    send_fig = pyqtSignal(str)

    def __init__(self):
        super(RandomForest, self).__init__()
        self.Title = "Random Forest Classifier"
        self.initUi()

    def initUi(self):
        self.setWindowTitle(self.Title)
        self.setStyleSheet(font_size_window)

        self.main_widget = QWidget(self)

        self.layout = QGridLayout(self.main_widget)

        self.groupBox1 = QGroupBox('Random Forest Features')
        self.groupBox1Layout= QGridLayout()
        self.groupBox1.setLayout(self.groupBox1Layout)
        self.groupBox1.setMinimumWidth(400)

        self.featureScrollLayout = QFormLayout()
        self.featureScrollWidget = QWidget()
        self.featureScrollWidget.setLayout(self.featureScrollLayout)

        self.featureScrollArea = QScrollArea()
        self.featureScrollArea.setStyleSheet(font_size_model)
        self.featureScrollArea.setWidgetResizable(True)
        self.featureScrollArea.setWidget(self.featureScrollWidget)

        self.featureCurrentCount = 0
        self.featureDict = dict()
        while self.featureCurrentCount < len(featuresList):
            s = "f" + str(self.featureCurrentCount)

            self.featureDict[s] = QFrame()
            self.featureDict[s + "Layout"] = QGridLayout()
            self.featureDict[s].setLayout(self.featureDict[s + "Layout"])
            self.featureDict[s + "Layout"].setContentsMargins(0,7,0,7)

            self.featureDict[s + "CheckBox"] = QCheckBox(featuresList[self.featureCurrentCount],self)
            self.featureDict[s + "CheckBox"].setMaximumWidth(200)

            self.featureDict[s + "CheckBox"].setChecked(True)
            self.featureDict[s + "Layout"].addWidget(self.featureDict[s + "CheckBox"], 0, 0, 1, 1)
            self.featureCurrentCount+=1

            if (self.featureCurrentCount<len(featuresList)):
                t = "f" + str(self.featureCurrentCount)
                self.featureDict[t + "CheckBox"] = QCheckBox(featuresList[self.featureCurrentCount], self)
                self.featureDict[t + "CheckBox"].setMaximumWidth(200)
                self.featureDict[t + "CheckBox"].setChecked(True)
                self.featureDict[s + "Layout"].addWidget(self.featureDict[t + "CheckBox"], 0, 1, 1, 1)
                self.featureCurrentCount += 1
            else:
                self.featureDict[s + "Layout"].addWidget(QLabel(""), 0, 1, 1, 1)


            self.featureScrollLayout.addRow(self.featureDict[s])


        # self.feature0 = QCheckBox(features_list[0],self)
        # self.feature1 = QCheckBox(features_list[1],self)
        # self.feature2 = QCheckBox(features_list[2], self)
        # self.feature3 = QCheckBox(features_list[3], self)
        # self.feature4 = QCheckBox(features_list[4],self)
        # self.feature5 = QCheckBox(features_list[5],self)
        # self.feature6 = QCheckBox(features_list[6], self)
        # self.feature7 = QCheckBox(features_list[7], self)
        # self.feature8 = QCheckBox(features_list[8], self)
        # self.feature9 = QCheckBox(features_list[9], self)
        # self.feature10 = QCheckBox(features_list[10], self)
        # self.feature11 = QCheckBox(features_list[11], self)
        # self.feature12 = QCheckBox(features_list[12], self)
        # self.feature13 = QCheckBox(features_list[13], self)
        # self.feature14 = QCheckBox(features_list[14], self)
        # self.feature15 = QCheckBox(features_list[15], self)
        # self.feature16 = QCheckBox(features_list[16], self)
        # self.feature17 = QCheckBox(features_list[17], self)
        # self.feature18 = QCheckBox(features_list[18], self)
        # self.feature19 = QCheckBox(features_list[19], self)
        # self.feature20 = QCheckBox(features_list[20], self)
        # self.feature21 = QCheckBox(features_list[21], self)
        # self.feature22 = QCheckBox(features_list[22], self)
        # self.feature23 = QCheckBox(features_list[23], self)
        # self.feature24 = QCheckBox(features_list[24], self)
        # self.feature25 = QCheckBox(features_list[25], self)
        # self.feature26 = QCheckBox(features_list[26], self)
        # self.feature27 = QCheckBox(features_list[27], self)
        # self.feature28 = QCheckBox(features_list[28], self)
        # self.feature29 = QCheckBox(features_list[29], self)
        # self.feature0.setChecked(True)
        # self.feature1.setChecked(True)
        # self.feature2.setChecked(True)
        # self.feature3.setChecked(True)
        # self.feature4.setChecked(True)
        # self.feature5.setChecked(True)
        # self.feature6.setChecked(True)
        # self.feature7.setChecked(True)
        # self.feature8.setChecked(True)
        # self.feature9.setChecked(True)
        # self.feature10.setChecked(True)
        # self.feature11.setChecked(True)
        # self.feature12.setChecked(True)
        # self.feature13.setChecked(True)
        # self.feature14.setChecked(True)
        # self.feature15.setChecked(True)
        # self.feature16.setChecked(True)
        # self.feature17.setChecked(True)
        # self.feature18.setChecked(True)
        # self.feature19.setChecked(True)
        # self.feature20.setChecked(True)
        # self.feature21.setChecked(True)
        # self.feature22.setChecked(True)
        # self.feature23.setChecked(True)
        # self.feature24.setChecked(True)
        # self.feature25.setChecked(True)
        # self.feature26.setChecked(True)
        # self.feature27.setChecked(True)
        # self.feature28.setChecked(True)
        # self.feature29.setChecked(True)

        self.lblPercentTest = QLabel('Percentage for Test :')
        self.lblPercentTest.adjustSize()
        self.lblPercentTest.setMaximumWidth(200)

        self.txtPercentTest = QLineEdit(self)
        self.txtPercentTest.setText("20")
        self.txtPercentTest.setMaximumWidth(200)

        self.lblEstimatorCount = QLabel('Number of Trees:')
        self.lblEstimatorCount.adjustSize()
        self.lblEstimatorCount.setMaximumWidth(200)

        self.txtEstimatorCount = QLineEdit(self)
        self.txtEstimatorCount.setText("35")
        self.txtEstimatorCount.setMaximumWidth(200)

        self.btnExecute = QPushButton("Run Model")
        self.btnExecute.clicked.connect(self.update)
        self.btnExecute.setMaximumWidth(200)

        self.btnSave = QPushButton("Save Model")
        self.btnSave.clicked.connect(self.saveModel)
        self.btnSave.setEnabled(False)
        self.btnSave.setMaximumWidth(200)


        self.groupBox1Layout.addWidget(self.featureScrollArea, 0, 0, 15, 2)
        self.groupBox1Layout.addWidget(self.lblPercentTest, 15, 0,1,1)
        self.groupBox1Layout.addWidget(self.txtPercentTest, 15, 1,1,1)
        self.groupBox1Layout.addWidget(self.lblEstimatorCount, 16, 0, 1, 1)
        self.groupBox1Layout.addWidget(self.txtEstimatorCount, 16, 1, 1, 1)
        self.groupBox1Layout.addWidget(self.btnExecute, 17, 0,1,1)
        self.groupBox1Layout.addWidget(self.btnSave, 17, 1,1,1)

        self.groupBox2 = QGroupBox('Results from the model')
        self.groupBox2Layout = QVBoxLayout()
        self.groupBox2.setLayout(self.groupBox2Layout)
        self.groupBox2.setMinimumSize(400, 50)

        self.lblResults = QLabel('Results:')
        self.lblResults.adjustSize()
        self.txtResults = QPlainTextEdit()

        self.groupBox2Layout.addWidget(self.lblResults)
        self.groupBox2Layout.addWidget(self.txtResults)

        self.groupBox3 = QGroupBox('Summary and Comparison')
        self.groupBox3Layout = QVBoxLayout()
        self.groupBox3.setLayout(self.groupBox3Layout)
        self.groupBox3.setMinimumSize(400, 50)
        self.lbl_current_model_summary = QLabel('Summary:')
        self.current_model_summary = QWidget(self)
        self.current_model_summary.layout = QFormLayout(self.current_model_summary)

        self.txtCurrentAccuracy = QLineEdit()
        self.txtCurrentPrecision = QLineEdit()
        self.txtCurrentRecall = QLineEdit()
        self.txtCurrentF1score = QLineEdit()
        self.current_model_summary.layout.addRow('Accuracy:', self.txtCurrentAccuracy)
        self.current_model_summary.layout.addRow('Precision:', self.txtCurrentPrecision)
        self.current_model_summary.layout.addRow('Recall:', self.txtCurrentRecall)
        self.current_model_summary.layout.addRow('F1 Score:', self.txtCurrentF1score)
        self.lbl_other_models = QLabel('Other Models Accuracy:')
        self.other_models = QWidget(self)
        self.other_models.layout = QFormLayout(self.other_models)

        self.txtAccuracy_lr = QLineEdit()
        self.txtAccuracy_knn = QLineEdit()
        self.txtAccuracy_dt = QLineEdit()
        self.other_models.layout.addRow('Logistic:', self.txtAccuracy_lr)
        self.other_models.layout.addRow('KNN:', self.txtAccuracy_knn)
        self.other_models.layout.addRow('Decision Trees:', self.txtAccuracy_dt)

        self.groupBox3Layout.addWidget(self.lbl_current_model_summary)
        self.groupBox3Layout.addWidget(self.current_model_summary)
        self.groupBox3Layout.addWidget(self.lbl_other_models)
        self.groupBox3Layout.addWidget(self.other_models)

        #::--------------------------------------
        # Graphic 1 : Confusion Matrix
        #::--------------------------------------

        self.fig = Figure()
        self.ax1 = self.fig.add_subplot(111)
        self.axes=[self.ax1]
        self.canvas = FigureCanvas(self.fig)

        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.canvas.updateGeometry()

        self.groupBoxG1 = QGroupBox('Confusion Matrix')
        self.groupBoxG1Layout= QVBoxLayout()
        self.groupBoxG1.setLayout(self.groupBoxG1Layout)

        self.groupBoxG1Layout.addWidget(self.canvas)

        #::---------------------------------------
        # Graphic 2 : AUC Score Vs Number of Trees
        #::---------------------------------------

        self.fig2 = Figure()
        self.ax2 = self.fig2.add_subplot(111)
        self.axes2 = [self.ax2]
        self.canvas2 = FigureCanvas(self.fig2)

        self.canvas2.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.canvas2.updateGeometry()

        self.groupBoxG2 = QGroupBox('AUC Score Vs Number of Trees')
        self.groupBoxG2Layout = QVBoxLayout()
        self.groupBoxG2.setLayout(self.groupBoxG2Layout)

        self.groupBoxG2Layout.addWidget(self.canvas2)

        #::-------------------------------------------
        # Graphic 3 : Importance of Features
        #::-------------------------------------------

        self.fig3 = Figure()
        self.ax3 = self.fig3.add_subplot(111)
        self.axes3 = [self.ax3]
        self.canvas3 = FigureCanvas(self.fig3)

        self.canvas3.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.canvas3.updateGeometry()

        self.groupBoxG3 = QGroupBox('Importance of Features')
        self.groupBoxG3Layout = QVBoxLayout()
        self.groupBoxG3.setLayout(self.groupBoxG3Layout)
        self.groupBoxG3Layout.addWidget(self.canvas3)

        #::--------------------------------------------
        # Graphic 4 : ROC Curve by Class
        #::--------------------------------------------

        self.fig4 = Figure()
        self.ax4 = self.fig4.add_subplot(111)
        self.axes4 = [self.ax4]
        self.canvas4 = FigureCanvas(self.fig4)

        self.canvas4.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.canvas4.updateGeometry()

        self.groupBoxG4 = QGroupBox('ROC Curve by Class')
        self.groupBoxG4Layout = QVBoxLayout()
        self.groupBoxG4.setLayout(self.groupBoxG4Layout)
        self.groupBoxG4Layout.addWidget(self.canvas4)

        #::-------------------------------------------------
        # End of graphs
        #::-------------------------------------------------

        self.layout.addWidget(self.groupBox1,0,0,2,1)
        self.layout.addWidget(self.groupBoxG1,0,1,1,1)
        self.layout.addWidget(self.groupBoxG3,0,2,1,1)
        self.layout.addWidget(self.groupBox2,0,3,1,1)
        self.layout.addWidget(self.groupBoxG2,1,1,1,1)
        self.layout.addWidget(self.groupBoxG4,1,2,1,1)
        self.layout.addWidget(self.groupBox3,1,3,1,1)

        self.setCentralWidget(self.main_widget)
        self.resize(1800, 900)
        self.show()

    def update(self):
        self.list_corr_features = pd.DataFrame([])
        # self.featureCount=0

        for i in range (len(featuresList)):
            s = "f" + str(i) + "CheckBox"
            if (self.featureDict[s].isChecked()):
                if len(self.list_corr_features) == 0:
                    self.list_corr_features = data[featuresList[i]]
                else:
                    self.list_corr_features = pd.concat([self.list_corr_features, data[featuresList[i]]], axis=1)


        try:
            vtest_per = float(self.txtPercentTest.text())
            if(vtest_per<100 and vtest_per>0):
                pass
            else:
                vtest_per = 20
                self.txtPercentTest.setText(str(vtest_per))
        except:
            vtest_per=20
            self.txtPercentTest.setText(str(vtest_per))

        try:
            estimator_input = round(float(self.txtEstimatorCount.text()))
            if (estimator_input < 1000 and estimator_input > 0):
                pass
            else:
                estimator_input = 35
                self.txtEstimatorCount.setText(str(estimator_input))
        except:
            estimator_input=35
            self.txtEstimatorCount.setText(str(estimator_input))

        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()
        self.ax4.clear()
        self.txtResults.clear()
        self.txtResults.setUndoRedoEnabled(False)

        vtest_per = vtest_per / 100

        X_dt =  self.list_corr_features
        y_dt = data[targetVariable]
        X_columns=X_dt.columns.tolist()
        labelencoder_columns= list(set(X_columns) & set(ordinalFeatures))
        one_hot_encoder_columns=list(set(X_columns) & set(nominalFeatures))

        class_le = LabelEncoder()
        class_ohe=OneHotEncoder()

        temp = X_columns.copy()
        for ohe_val in one_hot_encoder_columns:
            temp.remove(ohe_val)
        temp_X_dt=X_dt[temp]
        for le_val in labelencoder_columns:
            temp_X_dt[le_val] = class_le.fit_transform(temp_X_dt[le_val])
        X_dt=pd.concat((temp_X_dt,pd.get_dummies(X_dt[one_hot_encoder_columns])),1)
        print("Y_DT(Prev): Values:\n", y_dt, "\n\n")
        y_dt = class_le.fit_transform(y_dt)
        print("Y_DT(After): Values:\n",y_dt,"\n\n")
        X_train, X_test, y_train, y_test = train_test_split(X_dt, y_dt, test_size=vtest_per, random_state=500)

        # specify random forest classifier
        self.clf_rf = RandomForestClassifier(n_estimators=estimator_input, random_state=500)

        # perform training
        self.clf_rf.fit(X_train, y_train)

        #-----------------------------------------------------------------------

        # prediction on test using all features
        y_pred = self.clf_rf.predict(X_test)
        y_pred_score = self.clf_rf.predict_proba(X_test)


        # confusion matrix for RandomForest
        conf_matrix = confusion_matrix(y_test, y_pred)

        # clasification report

        self.ff_class_rep = classification_report(y_test, y_pred)
        self.txtResults.appendPlainText(self.ff_class_rep)

        # accuracy score

        self.ff_accuracy_score = accuracy_score(y_test, y_pred) * 100
        self.txtCurrentAccuracy.setText(str(self.ff_accuracy_score))

        # precision score

        self.ff_precision_score = precision_score(y_test, y_pred,average='micro')*100
        self.txtCurrentPrecision.setText(str(self.ff_precision_score))

        # recall score

        self.ff_recall_score = recall_score(y_test, y_pred,average='micro') * 100
        self.txtCurrentRecall.setText(str(self.ff_recall_score))

        # f1_score

        self.ff_f1_score = f1_score(y_test, y_pred, average='micro')
        self.txtCurrentF1score.setText(str(self.ff_f1_score))

        #::------------------------------------
        ##  Ghaph1 :
        ##  Confusion Matrix
        #::------------------------------------
        class_names1 = list([" "])
        class_names1+=list(targetClasses)
        class_names1 = ['\n'.join(wrap(l, 12)) for l in class_names1]

        self.ax1.matshow(conf_matrix, cmap= plt.cm.get_cmap('Blues', 14))
        self.ax1.set_yticklabels(class_names1,fontsize=8)
        self.ax1.set_xticklabels(class_names1,rotation = 90,fontsize=8)
        self.ax1.set_xlabel('Predicted label')
        self.ax1.set_ylabel('True label')

        for i in range(len(targetClasses)):
            for j in range(len(targetClasses)):
                self.ax1.text(j, i, str(conf_matrix[i][j]))

        self.fig.tight_layout()
        self.fig.canvas.draw_idle()

        #::----------------------------------------
        ## Graph 2 - AUC Score vs Number of Trees
        #::----------------------------------------

        auc_test = []
        auc_train = []
        estimator_count= [1, 2, 4, 8, 16, 32, 50, 64, 100, 200, 300]
        # Might take some time
        for i in estimator_count:
            self.rf_graph = RandomForestClassifier(n_estimators=i)
            self.rf_graph.fit(X_train, y_train)
            temp_train_pred = self.rf_graph.predict(X_train)
            temp_test_pred = self.rf_graph.predict(X_test)
            # classifier = OneVsRestClassifier(self.rf_graph)
            # y_pred = self.rf_graph.predict(X_train)
            # false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train[:, i], y_score[:, i])
            if len(targetClasses) > 2:
                auc_train.append(self.multiclass_roc_auc_score(y_train,temp_train_pred))
                # y_pred = self.rf_graph.predict(X_test)
                # false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test[:, i], y_score[:, i])
                auc_test.append(self.multiclass_roc_auc_score(y_test,temp_test_pred))
            else:
                false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, temp_train_pred)
                auc_train.append(auc(false_positive_rate, true_positive_rate))
                false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, temp_test_pred)
                auc_test.append(auc(false_positive_rate, true_positive_rate))

        self.ax2.plot(estimator_count,auc_train , color='blue', label="Train AUC")
        self.ax2.plot(estimator_count, auc_test, color='red', label="Test AUC")
        self.ax2.set_xlabel('Number of Trees')
        self.ax2.set_ylabel('AUC Score')
        self.ax2.legend(loc="lower right")

        self.fig2.tight_layout()
        self.fig2.canvas.draw_idle()

        #::----------------------------------------
        ## Graph 3 - Importance of Features
        #::----------------------------------------

        importances = self.clf_rf.feature_importances_

        # convert the importances into one-dimensional 1darray with corresponding df column names as axis labels
        f_importances = pd.Series(importances, X_dt.columns)
        # f_importances= ['\n'.join(wrap(l, 20)) for l in f_importances]
        # sort the array in descending order of the importances
        f_importances.sort_values(ascending=False, inplace=True)
        f_importances=f_importances[0:10]
        X_Features = f_importances.index
        y_Importance = list(f_importances)
        X_Features=['\n'.join(wrap(l, 20)) for l in X_Features]

        self.ax3.barh(X_Features, y_Importance)
        self.ax3.set_aspect('auto')
        # self.ax3.set_yticks(['\n'.join(wrap(l, 12)) for l in X_Features])
        self.ax3.tick_params(labelsize=8)
        # plt.ylabel(fontdict=)

        # show the plot
        self.fig3.tight_layout()
        self.fig3.canvas.draw_idle()

        #::-----------------------------------------------------
        # Graph 4 - ROC Curve by Class
        #::-----------------------------------------------------

        if len(targetClasses)>2:
            y_test_bin = label_binarize(y_dt, classes=range(len(targetClasses)))
            n_classes = y_test_bin.shape[1]
            print(n_classes)
            X_train_temp, X_test_temp, y_train_temp, y_test_temp = train_test_split(X_dt, y_test_bin, test_size=vtest_per, random_state=500)

            classifier = OneVsRestClassifier(self.clf_rf)
            y_score = classifier.fit(X_train_temp, y_train_temp).predict_proba(X_test_temp)
            #str_classes = targetClasses
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            for i in range(n_classes):
                fpr[i], tpr[i], _ = roc_curve(y_test_temp[:, i], y_score[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])

            for i in range(n_classes):
                self.ax4.plot(fpr[i], tpr[i],
                         label='{0} (area = {1:0.2f})'
                               ''.format(targetClasses[i], roc_auc[i]))
        else:
            y_test_bin = pd.get_dummies(y_test).to_numpy()
            n_classes = y_test_bin.shape[1]

            # From the sckict learn site
            # https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            for i in range(n_classes):
                fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_score[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])
            # print(pd.get_dummies(y_test).to_numpy().ravel())

            # print("\n\n********************************\n\n")
            # print(y_pred_score.ravel())
            # Compute micro-average ROC curve and ROC area
            fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_pred_score.ravel())

            roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
            lw = 2
            str_classes = ['No', 'Yes']
            colors = cycle(['magenta', 'darkorange'])
            for i, color in zip(range(n_classes), colors):
                self.ax4.plot(fpr[i], tpr[i], color=color, lw=lw,
                              label='{0} (area = {1:0.2f})'
                                    ''.format(targetClasses[i], roc_auc[i]))

        self.ax4.plot([0, 1], [0, 1], 'k--')
        self.ax4.set_xlim([0.0, 1.0])
        self.ax4.set_ylim([0.0, 1.05])
        self.ax4.set_xlabel('False Positive Rate')
        self.ax4.set_ylabel('True Positive Rate')
        self.ax4.legend(loc="lower right")

        # show the plot
        self.fig4.tight_layout()
        self.fig4.canvas.draw_idle()



        # y_test_bin = pd.get_dummies(y_test).to_numpy()
        # n_classes = y_test_bin.shape[1]
        #
        # fpr = dict()
        # tpr = dict()
        # roc_auc = dict()
        # for i in range(n_classes):
        #     fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_score[:, i])
        #     roc_auc[i] = auc(fpr[i], tpr[i])
        #
        # # Compute micro-average ROC curve and ROC area
        # fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_pred_score.ravel())
        #
        # roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        # lw = 2
        # str_classes= targetClasses
        # colors = cycle(['magenta', 'darkorange'])
        # for i, color in zip(range(n_classes), colors):
        #     self.ax4.plot(fpr[i], tpr[i], color=color, lw=lw,
        #              label='{0} (area = {1:0.2f})'
        #                    ''.format(str_classes[i], roc_auc[i]))
        #
        # self.ax4.plot([0, 1], [0, 1], 'k--', lw=lw)
        # self.ax4.set_xlim([0.0, 1.0])
        # self.ax4.set_ylim([0.0, 1.05])
        # self.ax4.set_xlabel('False Positive Rate')
        # self.ax4.set_ylabel('True Positive Rate')
        # self.ax4.legend(loc="lower right")
        #
        # # show the plot
        # self.fig4.tight_layout()
        # self.fig4.canvas.draw_idle()

        #::-----------------------------------------------------
        # Other Models Comparison
        #::-----------------------------------------------------

        self.other_clf_lr=LogisticRegression(random_state=500)
        self.other_clf_lr.fit(X_train, y_train)
        y_pred_lr = self.other_clf_lr.predict(X_test)
        self.accuracy_lr=accuracy_score(y_test, y_pred_lr) *100
        self.txtAccuracy_lr.setText(str(self.accuracy_lr))

        self.other_clf_dt = DecisionTreeClassifier(criterion="gini")
        self.other_clf_dt.fit(X_train, y_train)
        y_pred_dt = self.other_clf_dt.predict(X_test)
        self.accuracy_dt = accuracy_score(y_test, y_pred_dt) * 100
        self.txtAccuracy_dt.setText(str(self.accuracy_dt))

        self.other_clf_knn = KNeighborsClassifier(n_neighbors=9)
        self.other_clf_knn.fit(X_train, y_train)
        y_pred_knn = self.other_clf_knn.predict(X_test)
        self.accuracy_knn = accuracy_score(y_test, y_pred_knn) * 100
        self.txtAccuracy_knn.setText(str(self.accuracy_knn))

        self.btnSave.setEnabled(True)

    def saveModel(self):
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getSaveFileName(self, "QFileDialog.getSaveFileName()", "",
                                                  "SAV Files (*.sav)", options=options)
        if fileName:
            # print(fileName)
            # fileName += '.out'
            pickle.dump(self.clf_rf, open(fileName, 'wb'))

    def multiclass_roc_auc_score(self, y_test, y_pred, average="micro"):
        lb = LabelBinarizer()
        lb.fit(y_test)
        y_test = lb.transform(y_test)
        y_pred = lb.transform(y_pred)
        return roc_auc_score(y_test, y_pred, average=average)

    def multiclass_roc_curve(self, y_test, y_pred, average="micro"):
        lb = LabelBinarizer()
        lb.fit(y_test)
        y_test = lb.transform(y_test)
        y_pred = lb.transform(y_pred)
        return roc_curve(y_test, y_pred, average=average)


class DecisionTree(QMainWindow):
    #::--------------------------------------------------------------------------------
    # Decision Tree Classifier using the attrition dataset
    # the methods in this class are
    #       _init_ : initialize the class
    #       initUi : creates the canvas and all the elements in the canvas
    #       update : populates the elements of the canvas base on the parameters
    #               chosen by the user
    #::---------------------------------------------------------------------------------
    send_fig = pyqtSignal(str)

    def __init__(self):
        super(DecisionTree, self).__init__()
        self.Title = "Decision Tree Classifier"
        self.initUi()

    def initUi(self):
        self.setWindowTitle(self.Title)
        self.setStyleSheet(font_size_window)

        self.main_widget = QWidget(self)

        self.layout = QGridLayout(self.main_widget)

        self.groupBox1 = QGroupBox('Decision Tree Features')
        self.groupBox1Layout= QGridLayout()   # Grid
        self.groupBox1.setLayout(self.groupBox1Layout)
        self.groupBox1.setMinimumWidth(400)

        self.featureScrollLayout = QFormLayout()
        self.featureScrollWidget = QWidget()
        self.featureScrollWidget.setLayout(self.featureScrollLayout)

        self.featureScrollArea = QScrollArea()
        self.featureScrollArea.setStyleSheet(font_size_model)
        self.featureScrollArea.setWidgetResizable(True)
        self.featureScrollArea.setWidget(self.featureScrollWidget)

        self.featureCurrentCount = 0
        self.featureDict = dict()
        while self.featureCurrentCount < len(featuresList):
            s = "f" + str(self.featureCurrentCount)

            self.featureDict[s] = QFrame()
            self.featureDict[s + "Layout"] = QGridLayout()
            self.featureDict[s].setLayout(self.featureDict[s + "Layout"])
            self.featureDict[s + "Layout"].setContentsMargins(0, 7, 0, 7)

            self.featureDict[s + "CheckBox"] = QCheckBox(featuresList[self.featureCurrentCount], self)
            self.featureDict[s + "CheckBox"].setMaximumWidth(200)

            self.featureDict[s + "CheckBox"].setChecked(True)
            self.featureDict[s + "Layout"].addWidget(self.featureDict[s + "CheckBox"], 0, 0, 1, 1)
            self.featureCurrentCount += 1

            if (self.featureCurrentCount < len(featuresList)):
                t = "f" + str(self.featureCurrentCount)
                self.featureDict[t + "CheckBox"] = QCheckBox(featuresList[self.featureCurrentCount], self)
                self.featureDict[t + "CheckBox"].setMaximumWidth(200)
                self.featureDict[t + "CheckBox"].setChecked(True)
                self.featureDict[s + "Layout"].addWidget(self.featureDict[t + "CheckBox"], 0, 1, 1, 1)
                self.featureCurrentCount += 1
            else:
                self.featureDict[s + "Layout"].addWidget(QLabel(""), 0, 1, 1, 1)

            self.featureScrollLayout.addRow(self.featureDict[s])

        self.lblPercentTest = QLabel('Percentage for Test :')
        self.lblPercentTest.adjustSize()
        self.lblPercentTest.setMaximumWidth(200)

        self.txtPercentTest = QLineEdit(self)
        self.txtPercentTest.setText("20")
        self.txtPercentTest.setMaximumWidth(200)

        self.btnExecute = QPushButton("Run Model")
        self.btnExecute.clicked.connect(self.update)
        self.btnExecute.setMaximumWidth(200)

        self.btnSave = QPushButton("Save Model")
        self.btnSave.clicked.connect(self.saveModel)
        self.btnSave.setEnabled(False)
        self.btnSave.setMaximumWidth(200)

        self.groupBox1Layout.addWidget(self.featureScrollArea, 0, 0, 16, 2)
        self.groupBox1Layout.addWidget(self.lblPercentTest, 16, 0,1,1)
        self.groupBox1Layout.addWidget(self.txtPercentTest, 16, 1,1,1)
        self.groupBox1Layout.addWidget(self.btnExecute, 17, 0,1,1)
        self.groupBox1Layout.addWidget(self.btnSave, 17, 1, 1, 1)

        self.groupBox2 = QGroupBox('Results from the model')
        self.groupBox2Layout = QVBoxLayout()
        self.groupBox2.setLayout(self.groupBox2Layout)
        self.groupBox2.setMinimumSize(400, 50)

        self.lblResults = QLabel('Results:')
        self.lblResults.adjustSize()
        self.txtResults = QPlainTextEdit()

        self.groupBox2Layout.addWidget(self.lblResults)
        self.groupBox2Layout.addWidget(self.txtResults)

        self.groupBox3 = QGroupBox('Summary and Comparison')
        self.groupBox3Layout = QVBoxLayout()
        self.groupBox3.setLayout(self.groupBox3Layout)
        self.groupBox3.setMinimumSize(400, 50)
        self.lbl_current_model_summary = QLabel('Summary:')
        self.current_model_summary = QWidget(self)
        self.current_model_summary.layout = QFormLayout(self.current_model_summary)

        self.txtCurrentAccuracy = QLineEdit()
        self.txtCurrentPrecision = QLineEdit()
        self.txtCurrentRecall = QLineEdit()
        self.txtCurrentF1score = QLineEdit()
        self.current_model_summary.layout.addRow('Accuracy:', self.txtCurrentAccuracy)
        self.current_model_summary.layout.addRow('Precision:', self.txtCurrentPrecision)
        self.current_model_summary.layout.addRow('Recall:', self.txtCurrentRecall)
        self.current_model_summary.layout.addRow('F1 Score:', self.txtCurrentF1score)
        self.lbl_other_models = QLabel('Other Models Accuracy:')
        self.other_models = QWidget(self)
        self.other_models.layout = QFormLayout(self.other_models)

        self.txtAccuracy_lr = QLineEdit()
        self.txtAccuracy_knn = QLineEdit()
        self.txtAccuracy_rf = QLineEdit()
        self.other_models.layout.addRow('Logistic:', self.txtAccuracy_lr)
        self.other_models.layout.addRow('KNN:', self.txtAccuracy_knn)
        self.other_models.layout.addRow('Random Forest:', self.txtAccuracy_rf)

        self.groupBox3Layout.addWidget(self.lbl_current_model_summary)
        self.groupBox3Layout.addWidget(self.current_model_summary)
        self.groupBox3Layout.addWidget(self.lbl_other_models)
        self.groupBox3Layout.addWidget(self.other_models)

        #::--------------------------------------
        # Graphic 1 : Confusion Matrix
        #::--------------------------------------

        self.fig = Figure()
        self.ax1 = self.fig.add_subplot(111)
        self.axes=[self.ax1]
        self.canvas = FigureCanvas(self.fig)

        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.canvas.updateGeometry()

        self.groupBoxG1 = QGroupBox('Confusion Matrix')
        self.groupBoxG1Layout= QVBoxLayout()
        self.groupBoxG1.setLayout(self.groupBoxG1Layout)

        self.groupBoxG1Layout.addWidget(self.canvas)

        #::---------------------------------------
        # Graphic 2 : Decision Tree Graph
        #::---------------------------------------

        self.labelImage = QLabel(self)
        self.image = QPixmap()
        self.labelImage.setPixmap(self.image)
        self.image_area = QScrollArea()
        self.image_area.setWidget(self.labelImage)
        self.labelImage.setPixmap(QPixmap("temp_background.png"))
        self.labelImage.adjustSize()

        self.groupBoxG2 = QGroupBox('Decision Tree Graph')
        self.groupBoxG2Layout = QVBoxLayout()
        self.groupBoxG2.setLayout(self.groupBoxG2Layout)
        self.groupBoxG2Layout.addWidget(self.image_area)

        #::-------------------------------------------
        # Graphic 3 : Importance of Features
        #::-------------------------------------------

        self.fig3 = Figure()
        self.ax3 = self.fig3.add_subplot(111)
        self.axes3 = [self.ax3]
        self.canvas3 = FigureCanvas(self.fig3)

        self.canvas3.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.canvas3.updateGeometry()

        self.groupBoxG3 = QGroupBox('Importance of Features')
        self.groupBoxG3Layout = QVBoxLayout()
        self.groupBoxG3.setLayout(self.groupBoxG3Layout)
        self.groupBoxG3Layout.addWidget(self.canvas3)

        #::--------------------------------------------
        # Graphic 4 : ROC Curve by Class
        #::--------------------------------------------

        self.fig4 = Figure()
        self.ax4 = self.fig4.add_subplot(111)
        self.axes4 = [self.ax4]
        self.canvas4 = FigureCanvas(self.fig4)

        self.canvas4.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.canvas4.updateGeometry()

        self.groupBoxG4 = QGroupBox('ROC Curve by Class')
        self.groupBoxG4Layout = QVBoxLayout()
        self.groupBoxG4.setLayout(self.groupBoxG4Layout)
        self.groupBoxG4Layout.addWidget(self.canvas4)

        #::-------------------------------------------------
        # End of graphs
        #::-------------------------------------------------

        self.layout.addWidget(self.groupBox1,0,0,2,1)
        self.layout.addWidget(self.groupBoxG1,0,1,1,1)
        self.layout.addWidget(self.groupBoxG3,0,2,1,1)
        self.layout.addWidget(self.groupBox2,0,3,1,1)
        self.layout.addWidget(self.groupBoxG2,1,1,1,1)
        self.layout.addWidget(self.groupBoxG4,1,2,1,1)
        self.layout.addWidget(self.groupBox3,1,3,1,1)

        self.setCentralWidget(self.main_widget)
        self.resize(1800, 900)
        self.show()

    def update(self):
        self.list_corr_features = pd.DataFrame([])
        # self.featureCount=0

        for i in range(len(featuresList)):
            s = "f" + str(i) + "CheckBox"
            if (self.featureDict[s].isChecked()):
                if len(self.list_corr_features) == 0:
                    self.list_corr_features = data[featuresList[i]]
                else:
                    self.list_corr_features = pd.concat([self.list_corr_features, data[featuresList[i]]], axis=1)

        try:
            vtest_per = float(self.txtPercentTest.text())
            if(vtest_per<100 and vtest_per>0):
                pass
            else:
                vtest_per = 20
                self.txtPercentTest.setText(str(vtest_per))
        except:
            vtest_per=20
            self.txtPercentTest.setText(str(vtest_per))

        self.ax1.clear()
        self.ax3.clear()
        self.ax4.clear()
        self.txtResults.clear()
        self.txtResults.setUndoRedoEnabled(False)

        vtest_per = vtest_per / 100

        X_dt = self.list_corr_features
        y_dt = data[targetVariable]
        X_columns = X_dt.columns.tolist()
        labelencoder_columns = list(set(X_columns) & set(ordinalFeatures))
        one_hot_encoder_columns = list(set(X_columns) & set(nominalFeatures))

        class_le = LabelEncoder()
        class_ohe = OneHotEncoder()

        temp = X_columns.copy()
        for ohe_val in one_hot_encoder_columns:
            temp.remove(ohe_val)
        temp_X_dt = X_dt[temp]
        for le_val in labelencoder_columns:
            temp_X_dt[le_val] = class_le.fit_transform(temp_X_dt[le_val])
        X_dt = pd.concat((temp_X_dt, pd.get_dummies(X_dt[one_hot_encoder_columns])), 1)
        y_dt = class_le.fit_transform(y_dt)
        X_train, X_test, y_train, y_test = train_test_split(X_dt, y_dt, test_size=vtest_per, random_state=500)

        # specify decision tree classifier
        self.clf_dt = DecisionTreeClassifier(criterion="gini")

        # perform training
        self.clf_dt.fit(X_train, y_train)

        #-----------------------------------------------------------------------

        # prediction on test using all features
        y_pred = self.clf_dt.predict(X_test)
        y_pred_score = self.clf_dt.predict_proba(X_test)


        # confusion matrix for Decision Tree
        conf_matrix = confusion_matrix(y_test, y_pred)

        # clasification report

        self.ff_class_rep = classification_report(y_test, y_pred)
        self.txtResults.appendPlainText(self.ff_class_rep)

        # accuracy score

        self.ff_accuracy_score = accuracy_score(y_test, y_pred) * 100
        self.txtCurrentAccuracy.setText(str(self.ff_accuracy_score))

        # precision score

        self.ff_precision_score = precision_score(y_test, y_pred,average='micro')*100
        self.txtCurrentPrecision.setText(str(self.ff_precision_score))

        # recall score

        self.ff_recall_score = recall_score(y_test, y_pred,average='micro') * 100
        self.txtCurrentRecall.setText(str(self.ff_recall_score))

        # f1_score

        self.ff_f1_score = f1_score(y_test, y_pred,average='micro')
        self.txtCurrentF1score.setText(str(self.ff_f1_score))

        #::------------------------------------
        ##  Ghaph1 :
        ##  Confusion Matrix
        #::------------------------------------
        class_names1 = list([" "])
        class_names1 += list(targetClasses)
        class_names1 = ['\n'.join(wrap(l, 12)) for l in class_names1]

        self.ax1.matshow(conf_matrix, cmap=plt.cm.get_cmap('Blues', 14))
        self.ax1.set_yticklabels(class_names1, fontsize=8)
        self.ax1.set_xticklabels(class_names1, rotation=90, fontsize=8)
        self.ax1.set_xlabel('Predicted label')
        self.ax1.set_ylabel('True label')

        for i in range(len(targetClasses)):
            for j in range(len(targetClasses)):
                self.ax1.text(j, i, str(conf_matrix[i][j]))

        self.fig.tight_layout()
        self.fig.canvas.draw_idle()

        #::----------------------------------------
        ## Graph 2 - Decision Tree Graph
        #::----------------------------------------

        dot_data = export_graphviz(self.clf_dt,
                                        feature_names=X_train.columns,
                                        class_names=targetClasses,
                                        out_file=None,
                                        filled=True,
                                        rounded=True,
                                        max_depth=3)

        graph = graph_from_dot_data(dot_data)
        colors = ('turquoise', 'orange', 'blue')
        edges = collections.defaultdict(list)

        for edge in graph.get_edge_list():
            edges[edge.get_source()].append(int(edge.get_destination()))

        for edge in edges:
            edges[edge].sort()
            for i in range(2):
                dest = graph.get_node(str(edges[edge][i]))[0]
                dest.set_fillcolor(colors[i])
        graph.set_size('"15,15!"')
        graph.write_png('DecisionTree_Attrition.png')
        self.labelImage.setPixmap(QPixmap("DecisionTree_Attrition.png"))
        self.labelImage.adjustSize()

        #::----------------------------------------
        ## Graph 3 - Importance of Features
        #::----------------------------------------

        importances = self.clf_dt.feature_importances_

        # convert the importances into one-dimensional 1darray with corresponding df column names as axis labels
        f_importances = pd.Series(importances, X_dt.columns)

        # sort the array in descending order of the importances
        f_importances.sort_values(ascending=False, inplace=True)
        f_importances=f_importances[0:10]
        X_Features = f_importances.index
        y_Importance = list(f_importances)
        X_Features = ['\n'.join(wrap(l, 20)) for l in X_Features]

        self.ax3.barh(X_Features, y_Importance )
        self.ax3.set_aspect('auto')
        self.ax3.tick_params(labelsize=8)

        # show the plot
        self.fig3.tight_layout()
        self.fig3.canvas.draw_idle()

        #::-----------------------------------------------------
        # Graph 4 - ROC Curve by Class
        #::-----------------------------------------------------

        if len(targetClasses) > 2:
            y_test_bin = label_binarize(y_dt, classes=range(len(targetClasses)))
            n_classes = y_test_bin.shape[1]
            # print(n_classes)
            X_train_temp, X_test_temp, y_train_temp, y_test_temp = train_test_split(X_dt, y_test_bin,
                                                                                    test_size=vtest_per,
                                                                                    random_state=500)

            classifier = OneVsRestClassifier(self.clf_dt)
            y_score = classifier.fit(X_train_temp, y_train_temp).predict_proba(X_test_temp)
            # str_classes = targetClasses
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            for i in range(n_classes):
                fpr[i], tpr[i], _ = roc_curve(y_test_temp[:, i], y_score[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])

            for i in range(n_classes):
                self.ax4.plot(fpr[i], tpr[i],
                              label='{0} (area = {1:0.2f})'
                                    ''.format(targetClasses[i], roc_auc[i]))
        else:
            y_test_bin = pd.get_dummies(y_test).to_numpy()
            n_classes = y_test_bin.shape[1]

            # From the sckict learn site
            # https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            for i in range(n_classes):
                fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_score[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])
            # print(pd.get_dummies(y_test).to_numpy().ravel())

            # print("\n\n********************************\n\n")
            # print(y_pred_score.ravel())
            # Compute micro-average ROC curve and ROC area
            fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_pred_score.ravel())

            roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
            lw = 2
            str_classes = ['No', 'Yes']
            colors = cycle(['magenta', 'darkorange'])
            for i, color in zip(range(n_classes), colors):
                self.ax4.plot(fpr[i], tpr[i], color=color, lw=lw,
                              label='{0} (area = {1:0.2f})'
                                    ''.format(targetClasses[i], roc_auc[i]))

        self.ax4.plot([0, 1], [0, 1], 'k--')
        self.ax4.set_xlim([0.0, 1.0])
        self.ax4.set_ylim([0.0, 1.05])
        self.ax4.set_xlabel('False Positive Rate')
        self.ax4.set_ylabel('True Positive Rate')
        self.ax4.legend(loc="lower right")

        # show the plot
        self.fig4.tight_layout()
        self.fig4.canvas.draw_idle()

        #::-----------------------------------------------------
        # Other Models Comparison
        #::-----------------------------------------------------

        self.other_clf_lr=LogisticRegression(random_state=500)
        self.other_clf_lr.fit(X_train, y_train)
        y_pred_lr = self.other_clf_lr.predict(X_test)
        self.accuracy_lr=accuracy_score(y_test, y_pred_lr) *100
        self.txtAccuracy_lr.setText(str(self.accuracy_lr))

        self.other_clf_rf = RandomForestClassifier(n_estimators=100, random_state=500)
        self.other_clf_rf.fit(X_train, y_train)
        y_pred_rf = self.other_clf_rf.predict(X_test)
        self.accuracy_rf = accuracy_score(y_test, y_pred_rf) * 100
        self.txtAccuracy_rf.setText(str(self.accuracy_rf))

        self.other_clf_knn = KNeighborsClassifier(n_neighbors=9)
        self.other_clf_knn.fit(X_train, y_train)
        y_pred_knn = self.other_clf_knn.predict(X_test)
        self.accuracy_knn = accuracy_score(y_test, y_pred_knn) * 100
        self.txtAccuracy_knn.setText(str(self.accuracy_knn))

        self.btnSave.setEnabled(True)

    def saveModel(self):
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getSaveFileName(self, "QFileDialog.getSaveFileName()", "",
                                                  "SAV Files (*.sav)", options=options)
        if fileName:
            # print(fileName)
            # fileName += '.out'
            pickle.dump(self.clf_rf, open(fileName, 'wb'))

    def multiclass_roc_auc_score(self, y_test, y_pred, average="micro"):
        lb = LabelBinarizer()
        lb.fit(y_test)
        y_test = lb.transform(y_test)
        y_pred = lb.transform(y_pred)
        return roc_auc_score(y_test, y_pred, average=average)

    def multiclass_roc_curve(self, y_test, y_pred, average="micro"):
        lb = LabelBinarizer()
        lb.fit(y_test)
        y_test = lb.transform(y_test)
        y_pred = lb.transform(y_pred)
        return roc_curve(y_test, y_pred, average=average)


class LogisticRegressionClassifier(QMainWindow):
    #::--------------------------------------------------------------------------------
    # Logistic Regression using the attrition dataset
    # the methods in this class are
    #       _init_ : initialize the class
    #       initUi : creates the canvas and all the elements in the canvas
    #       update : populates the elements of the canvas base on the parameters
    #               chosen by the user
    #::---------------------------------------------------------------------------------
    send_fig = pyqtSignal(str)

    def __init__(self):
        super(LogisticRegressionClassifier, self).__init__()
        self.Title = "Logistic Regression"
        self.initUi()

    def initUi(self):
        self.setWindowTitle(self.Title)
        self.setStyleSheet(font_size_window)

        self.main_widget = QWidget(self)

        self.layout = QGridLayout(self.main_widget)

        self.groupBox1 = QGroupBox('Logistic Regression Features')
        self.groupBox1Layout= QGridLayout()
        self.groupBox1.setLayout(self.groupBox1Layout)
        self.groupBox1.setMinimumWidth(400)

        self.featureScrollLayout = QFormLayout()
        self.featureScrollWidget = QWidget()
        self.featureScrollWidget.setLayout(self.featureScrollLayout)

        self.featureScrollArea = QScrollArea()
        self.featureScrollArea.setStyleSheet(font_size_model)
        self.featureScrollArea.setWidgetResizable(True)
        self.featureScrollArea.setWidget(self.featureScrollWidget)

        self.featureCurrentCount = 0
        self.featureDict = dict()
        while self.featureCurrentCount < len(featuresList):
            s = "f" + str(self.featureCurrentCount)

            self.featureDict[s] = QFrame()
            self.featureDict[s + "Layout"] = QGridLayout()
            self.featureDict[s].setLayout(self.featureDict[s + "Layout"])
            self.featureDict[s + "Layout"].setContentsMargins(0, 7, 0, 7)

            self.featureDict[s + "CheckBox"] = QCheckBox(featuresList[self.featureCurrentCount], self)
            self.featureDict[s + "CheckBox"].setMaximumWidth(200)

            self.featureDict[s + "CheckBox"].setChecked(True)
            self.featureDict[s + "Layout"].addWidget(self.featureDict[s + "CheckBox"], 0, 0, 1, 1)
            self.featureCurrentCount += 1

            if (self.featureCurrentCount < len(featuresList)):
                t = "f" + str(self.featureCurrentCount)
                self.featureDict[t + "CheckBox"] = QCheckBox(featuresList[self.featureCurrentCount], self)
                self.featureDict[t + "CheckBox"].setMaximumWidth(200)
                self.featureDict[t + "CheckBox"].setChecked(True)
                self.featureDict[s + "Layout"].addWidget(self.featureDict[t + "CheckBox"], 0, 1, 1, 1)
                self.featureCurrentCount += 1
            else:
                self.featureDict[s + "Layout"].addWidget(QLabel(""), 0, 1, 1, 1)

            self.featureScrollLayout.addRow(self.featureDict[s])

        self.lblPercentTest = QLabel('Percentage for Test :')
        self.lblPercentTest.adjustSize()
        self.lblPercentTest.setMaximumWidth(200)

        self.txtPercentTest = QLineEdit(self)
        self.txtPercentTest.setText("20")
        self.txtPercentTest.setMaximumWidth(200)

        self.btnExecute = QPushButton("Run Model")
        self.btnExecute.clicked.connect(self.update)
        self.btnExecute.setMaximumWidth(200)

        self.btnSave = QPushButton("Save Model")
        self.btnSave.clicked.connect(self.saveModel)
        self.btnSave.setEnabled(False)
        self.btnSave.setMaximumWidth(200)

        self.groupBox1Layout.addWidget(self.featureScrollArea, 0, 0, 16, 2)
        self.groupBox1Layout.addWidget(self.lblPercentTest, 16, 0, 1, 1)
        self.groupBox1Layout.addWidget(self.txtPercentTest, 16, 1, 1, 1)
        self.groupBox1Layout.addWidget(self.btnExecute, 17, 0, 1, 1)
        self.groupBox1Layout.addWidget(self.btnSave, 17, 1, 1, 1)

        self.groupBox2 = QGroupBox('Results from the model')
        self.groupBox2Layout = QVBoxLayout()
        self.groupBox2.setLayout(self.groupBox2Layout)
        self.groupBox2.setMinimumSize(400, 50)

        self.lblResults = QLabel('Results:')
        self.lblResults.adjustSize()
        self.txtResults = QPlainTextEdit()

        self.groupBox2Layout.addWidget(self.lblResults)
        self.groupBox2Layout.addWidget(self.txtResults)

        self.groupBox3 = QGroupBox('Summary and Comparison')
        self.groupBox3Layout = QVBoxLayout()
        self.groupBox3.setLayout(self.groupBox3Layout)
        self.groupBox3.setMinimumSize(400, 50)
        self.lbl_current_model_summary = QLabel('Summary:')
        self.current_model_summary = QWidget(self)
        self.current_model_summary.layout = QFormLayout(self.current_model_summary)

        self.txtCurrentAccuracy = QLineEdit()
        self.txtCurrentPrecision = QLineEdit()
        self.txtCurrentRecall = QLineEdit()
        self.txtCurrentF1score = QLineEdit()
        self.current_model_summary.layout.addRow('Accuracy:', self.txtCurrentAccuracy)
        self.current_model_summary.layout.addRow('Precision:', self.txtCurrentPrecision)
        self.current_model_summary.layout.addRow('Recall:', self.txtCurrentRecall)
        self.current_model_summary.layout.addRow('F1 Score:', self.txtCurrentF1score)

        self.lbl_other_models = QLabel('Other Models Accuracy:')
        self.other_models = QWidget(self)
        self.other_models.layout = QFormLayout(self.other_models)

        self.txtAccuracy_rf = QLineEdit()
        self.txtAccuracy_knn = QLineEdit()
        self.txtAccuracy_dt = QLineEdit()
        self.other_models.layout.addRow('Random Forest:', self.txtAccuracy_rf)
        self.other_models.layout.addRow('KNN:', self.txtAccuracy_knn)
        self.other_models.layout.addRow('Decision Trees:', self.txtAccuracy_dt)

        self.groupBox3Layout.addWidget(self.lbl_current_model_summary)
        self.groupBox3Layout.addWidget(self.current_model_summary)
        self.groupBox3Layout.addWidget(self.lbl_other_models)
        self.groupBox3Layout.addWidget(self.other_models)

        #::--------------------------------------
        # Graphic 1 : Confusion Matrix
        #::--------------------------------------

        self.fig = Figure()
        self.ax1 = self.fig.add_subplot(111)
        self.axes=[self.ax1]
        self.canvas = FigureCanvas(self.fig)

        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.canvas.updateGeometry()

        self.groupBoxG1 = QGroupBox('Confusion Matrix')
        self.groupBoxG1Layout= QVBoxLayout()
        self.groupBoxG1.setLayout(self.groupBoxG1Layout)

        self.groupBoxG1Layout.addWidget(self.canvas)

        #::---------------------------------------
        # Graphic 2 : Calibration Curve
        #::---------------------------------------

        self.fig2 = Figure()
        self.ax2 = self.fig2.add_subplot(111)
        self.axes2 = [self.ax2]
        self.canvas2 = FigureCanvas(self.fig2)

        self.canvas2.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.canvas2.updateGeometry()

        self.groupBoxG2 = QGroupBox('Calibration Curve')
        self.groupBoxG2Layout = QVBoxLayout()
        self.groupBoxG2.setLayout(self.groupBoxG2Layout)

        self.groupBoxG2Layout.addWidget(self.canvas2)

        #::-------------------------------------------
        # Graphic 3 : Cross Validation Score
        #::-------------------------------------------

        self.fig3 = Figure()
        self.ax3 = self.fig3.add_subplot(111)
        self.axes3 = [self.ax3]
        self.canvas3 = FigureCanvas(self.fig3)

        self.canvas3.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.canvas3.updateGeometry()

        self.groupBoxG3 = QGroupBox('Cross Validation Score')
        self.groupBoxG3Layout = QVBoxLayout()
        self.groupBoxG3.setLayout(self.groupBoxG3Layout)
        self.groupBoxG3Layout.addWidget(self.canvas3)

        #::--------------------------------------------
        # Graphic 4 : ROC Curve by class
        #::--------------------------------------------

        self.fig4 = Figure()
        self.ax4 = self.fig4.add_subplot(111)
        self.axes4 = [self.ax4]
        self.canvas4 = FigureCanvas(self.fig4)

        self.canvas4.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.canvas4.updateGeometry()

        self.groupBoxG4 = QGroupBox('ROC Curve by Class')
        self.groupBoxG4Layout = QVBoxLayout()
        self.groupBoxG4.setLayout(self.groupBoxG4Layout)
        self.groupBoxG4Layout.addWidget(self.canvas4)

        #::-------------------------------------------------
        # End of graphs
        #::-------------------------------------------------

        self.layout.addWidget(self.groupBox1,0,0,2,1)
        self.layout.addWidget(self.groupBoxG1,0,1,1,1)
        self.layout.addWidget(self.groupBoxG3,0,2,1,1)
        self.layout.addWidget(self.groupBox2,0,3,1,1)
        self.layout.addWidget(self.groupBoxG2,1,1,1,1)
        self.layout.addWidget(self.groupBoxG4,1,2,1,1)
        self.layout.addWidget(self.groupBox3,1,3,1,1)

        self.setCentralWidget(self.main_widget)
        self.resize(1800, 900)
        self.show()

    def update(self):
        self.list_corr_features = pd.DataFrame([])
        # self.featureCount=0

        for i in range(len(featuresList)):
            s = "f" + str(i) + "CheckBox"
            if (self.featureDict[s].isChecked()):
                if len(self.list_corr_features) == 0:
                    self.list_corr_features = data[featuresList[i]]
                else:
                    self.list_corr_features = pd.concat([self.list_corr_features, data[featuresList[i]]], axis=1)

        try:
            vtest_per = float(self.txtPercentTest.text())
            if(vtest_per<100 and vtest_per>0):
                pass
            else:
                vtest_per = 20
                self.txtPercentTest.setText(str(vtest_per))
        except:
            vtest_per=20
            self.txtPercentTest.setText(str(vtest_per))

        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()
        self.ax4.clear()
        self.txtResults.clear()
        self.txtResults.setUndoRedoEnabled(False)

        vtest_per = vtest_per / 100

        X_dt = self.list_corr_features
        y_dt = data[targetVariable]
        X_columns = X_dt.columns.tolist()
        labelencoder_columns = list(set(X_columns) & set(ordinalFeatures))
        one_hot_encoder_columns = list(set(X_columns) & set(nominalFeatures))

        class_le = LabelEncoder()
        class_ohe = OneHotEncoder()

        temp = X_columns.copy()
        for ohe_val in one_hot_encoder_columns:
            temp.remove(ohe_val)
        temp_X_dt = X_dt[temp]
        for le_val in labelencoder_columns:
            temp_X_dt[le_val] = class_le.fit_transform(temp_X_dt[le_val])
        X_dt = pd.concat((temp_X_dt, pd.get_dummies(X_dt[one_hot_encoder_columns])), 1)
        y_dt = class_le.fit_transform(y_dt)
        X_train, X_test, y_train, y_test = train_test_split(X_dt, y_dt, test_size=vtest_per, random_state=500)

        # specify logistic regression
        self.clf_lr = LogisticRegression()

        # perform training
        self.clf_lr.fit(X_train, y_train)

        #-----------------------------------------------------------------------

        # prediction on test using all features
        y_pred = self.clf_lr.predict(X_test)
        y_pred_score = self.clf_lr.predict_proba(X_test)

        # confusion matrix for RandomForest
        conf_matrix = confusion_matrix(y_test, y_pred)

        # clasification report

        self.ff_class_rep = classification_report(y_test, y_pred)
        self.txtResults.appendPlainText(self.ff_class_rep)

        # accuracy score

        self.ff_accuracy_score = accuracy_score(y_test, y_pred) * 100
        self.txtCurrentAccuracy.setText(str(self.ff_accuracy_score))

        # precision score

        self.ff_precision_score = precision_score(y_test, y_pred,average='micro')
        self.txtCurrentPrecision.setText(str(self.ff_precision_score))

        # recall score

        self.ff_recall_score = recall_score(y_test, y_pred,average='micro') * 100
        self.txtCurrentRecall.setText(str(self.ff_recall_score))

        # f1_score

        self.ff_f1_score = f1_score(y_test, y_pred,average='micro')
        self.txtCurrentF1score.setText(str(self.ff_f1_score))

        #::------------------------------------
        ##  Graph1 :
        ##  Confusion Matrix
        #::------------------------------------
        class_names1 = list([" "])
        class_names1 += list(targetClasses)
        class_names1 = ['\n'.join(wrap(l, 12)) for l in class_names1]

        self.ax1.matshow(conf_matrix, cmap=plt.cm.get_cmap('Blues', 14))
        self.ax1.set_yticklabels(class_names1, fontsize=8)
        self.ax1.set_xticklabels(class_names1, rotation=90, fontsize=8)
        self.ax1.set_xlabel('Predicted label')
        self.ax1.set_ylabel('True label')

        for i in range(len(targetClasses)):
            for j in range(len(targetClasses)):
                self.ax1.text(j, i, str(conf_matrix[i][j]))

        self.fig.tight_layout()
        self.fig.canvas.draw_idle()

        #::----------------------------------------
        ## Graph 2 - Calibration Curve
        #::----------------------------------------
        # y_test_bin = label_binarize(y_dt, classes=range(len(targetClasses)))
        # n_classes = y_test_bin.shape[1]
        # # print(n_classes)
        # X_train_temp, X_test_temp, y_train_temp, y_test_temp = train_test_split(X_dt, y_test_bin,
        #                                                                         test_size=vtest_per,
        #                                                                         random_state=500)
        # classifier = OneVsRestClassifier(self.clf_lr)
        # y_score = classifier.fit(X_train_temp, y_train_temp).predict_proba(X_test_temp)
        #
        # logreg_y, logreg_x = calibration_curve(y_test_temp, y_pred_score, n_bins=10)
        # self.ax2.plot(logreg_x, logreg_y, marker='o', linewidth=1)
        # self.ax2.plot(np.linspace(0,1,10), np.linspace(0,1,10), linewidth=1, color="black")
        # self.ax2.set_xlabel('Predicted probability')
        # self.ax2.set_ylabel('True probability in each bin')
        #
        # # show the plot
        # self.fig2.tight_layout()
        # self.fig2.canvas.draw_idle()

        #::----------------------------------------
        ## Graph 3 - Cross Validation Score
        #::----------------------------------------

        scores_arr = []

        for val in (X_train.columns):
            cvs_X = X_test[val].values.reshape(-1, 1)
            scores = cross_val_score(self.clf_lr, cvs_X, y_test, cv=5)
            scores_arr.append(scores.mean())

        f_importances = pd.Series(scores_arr, X_train.columns)

        f_importances.sort_values(ascending=False, inplace=True)
        f_importances=f_importances[0:20]
        X_Features = f_importances.index
        y_Importance = list(f_importances)
        X_Features = ['\n'.join(wrap(l, 20)) for l in X_Features]
        max_value=f_importances.max()
        min_value = f_importances.min()
        self.ax3.barh(X_Features, y_Importance )
        self.ax3.set_xlim(min_value-(min_value*0.05),max_value+(max_value*0.05))
        self.ax3.tick_params(labelsize=8)

        # show the plot
        self.fig3.tight_layout()
        self.fig3.canvas.draw_idle()

        #::-----------------------------------------------------
        # Graph 4 - ROC Curve by Class
        #::-----------------------------------------------------

        if len(targetClasses) > 2:
            y_test_bin = label_binarize(y_dt, classes=range(len(targetClasses)))
            n_classes = y_test_bin.shape[1]
            # print(n_classes)
            X_train_temp, X_test_temp, y_train_temp, y_test_temp = train_test_split(X_dt, y_test_bin,
                                                                                    test_size=vtest_per,
                                                                                    random_state=500)

            classifier = OneVsRestClassifier(self.clf_lr)
            y_score = classifier.fit(X_train_temp, y_train_temp).predict_proba(X_test_temp)
            # str_classes = targetClasses
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            for i in range(n_classes):
                fpr[i], tpr[i], _ = roc_curve(y_test_temp[:, i], y_score[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])

            for i in range(n_classes):
                self.ax4.plot(fpr[i], tpr[i],
                              label='{0} (area = {1:0.2f})'
                                    ''.format(targetClasses[i], roc_auc[i]))
        else:
            y_test_bin = pd.get_dummies(y_test).to_numpy()
            n_classes = y_test_bin.shape[1]

            # From the sckict learn site
            # https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            for i in range(n_classes):
                fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_score[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])
            # print(pd.get_dummies(y_test).to_numpy().ravel())

            # print("\n\n********************************\n\n")
            # print(y_pred_score.ravel())
            # Compute micro-average ROC curve and ROC area
            fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_pred_score.ravel())

            roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
            lw = 2
            str_classes = ['No', 'Yes']
            colors = cycle(['magenta', 'darkorange'])
            for i, color in zip(range(n_classes), colors):
                self.ax4.plot(fpr[i], tpr[i], color=color, lw=lw,
                              label='{0} (area = {1:0.2f})'
                                    ''.format(targetClasses[i], roc_auc[i]))

        self.ax4.plot([0, 1], [0, 1], 'k--')
        self.ax4.set_xlim([0.0, 1.0])
        self.ax4.set_ylim([0.0, 1.05])
        self.ax4.set_xlabel('False Positive Rate')
        self.ax4.set_ylabel('True Positive Rate')
        self.ax4.legend(loc="lower right")

        # show the plot
        self.fig4.tight_layout()
        self.fig4.canvas.draw_idle()

        #::-----------------------------------------------------
        # Other Models Comparison
        #::-----------------------------------------------------

        self.other_clf_rf=RandomForestClassifier(n_estimators=100, random_state=500)
        self.other_clf_rf.fit(X_train, y_train)
        y_pred_rf = self.other_clf_rf.predict(X_test)
        self.accuracy_rf=accuracy_score(y_test, y_pred_rf) *100
        self.txtAccuracy_rf.setText(str(self.accuracy_rf))

        self.other_clf_dt = DecisionTreeClassifier(criterion="gini")
        self.other_clf_dt.fit(X_train, y_train)
        y_pred_dt = self.other_clf_dt.predict(X_test)
        self.accuracy_dt = accuracy_score(y_test, y_pred_dt) * 100
        self.txtAccuracy_dt.setText(str(self.accuracy_dt))

        self.other_clf_knn = KNeighborsClassifier(n_neighbors=9)
        self.other_clf_knn.fit(X_train, y_train)
        y_pred_knn = self.other_clf_knn.predict(X_test)
        self.accuracy_knn = accuracy_score(y_test, y_pred_knn) * 100
        self.txtAccuracy_knn.setText(str(self.accuracy_knn))

        self.btnSave.setEnabled(True)

    def saveModel(self):
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getSaveFileName(self, "QFileDialog.getSaveFileName()", "",
                                                  "SAV Files (*.sav)", options=options)
        if fileName:
            # print(fileName)
            # fileName += '.out'
            pickle.dump(self.clf_rf, open(fileName, 'wb'))

    def multiclass_roc_auc_score(self, y_test, y_pred, average="micro"):
        lb = LabelBinarizer()
        lb.fit(y_test)
        y_test = lb.transform(y_test)
        y_pred = lb.transform(y_pred)
        return roc_auc_score(y_test, y_pred, average=average)

    def multiclass_roc_curve(self, y_test, y_pred, average="micro"):
        lb = LabelBinarizer()
        lb.fit(y_test)
        y_test = lb.transform(y_test)
        y_pred = lb.transform(y_pred)
        return roc_curve(y_test, y_pred, average=average)


class KNNClassifier(QMainWindow):
    #::--------------------------------------------------------------------------------
    # K Nearest Neighbours Classifier using the happiness dataset
    # the methods in this class are
    #       _init_ : initialize the class
    #       initUi : creates the canvas and all the elements in the canvas
    #       update : populates the elements of the canvas base on the parameters
    #               chosen by the user
    #::---------------------------------------------------------------------------------
    send_fig = pyqtSignal(str)

    def __init__(self):
        super(KNNClassifier, self).__init__()
        self.Title = "K-Nearest Neighbor"
        self.initUi()

    def initUi(self):
        self.setWindowTitle(self.Title)
        self.setStyleSheet(font_size_window)

        self.main_widget = QWidget(self)

        self.layout = QGridLayout(self.main_widget)

        self.groupBox1 = QGroupBox('K - Nearest Neighbor Features')
        self.groupBox1Layout= QGridLayout()
        self.groupBox1.setLayout(self.groupBox1Layout)
        self.groupBox1.setMinimumWidth(400)

        self.featureScrollLayout = QFormLayout()
        self.featureScrollWidget = QWidget()
        self.featureScrollWidget.setLayout(self.featureScrollLayout)

        self.featureScrollArea = QScrollArea()
        self.featureScrollArea.setStyleSheet(font_size_model)
        self.featureScrollArea.setWidgetResizable(True)
        self.featureScrollArea.setWidget(self.featureScrollWidget)

        self.featureCurrentCount = 0
        self.featureDict = dict()
        while self.featureCurrentCount < len(featuresList):
            s = "f" + str(self.featureCurrentCount)

            self.featureDict[s] = QFrame()
            self.featureDict[s + "Layout"] = QGridLayout()
            self.featureDict[s].setLayout(self.featureDict[s + "Layout"])
            self.featureDict[s + "Layout"].setContentsMargins(0, 7, 0, 7)

            self.featureDict[s + "CheckBox"] = QCheckBox(featuresList[self.featureCurrentCount], self)
            self.featureDict[s + "CheckBox"].setMaximumWidth(200)

            self.featureDict[s + "CheckBox"].setChecked(True)
            self.featureDict[s + "Layout"].addWidget(self.featureDict[s + "CheckBox"], 0, 0, 1, 1)
            self.featureCurrentCount += 1

            if (self.featureCurrentCount < len(featuresList)):
                t = "f" + str(self.featureCurrentCount)
                self.featureDict[t + "CheckBox"] = QCheckBox(featuresList[self.featureCurrentCount], self)
                self.featureDict[t + "CheckBox"].setMaximumWidth(200)
                self.featureDict[t + "CheckBox"].setChecked(True)
                self.featureDict[s + "Layout"].addWidget(self.featureDict[t + "CheckBox"], 0, 1, 1, 1)
                self.featureCurrentCount += 1
            else:
                self.featureDict[s + "Layout"].addWidget(QLabel(""), 0, 1, 1, 1)

            self.featureScrollLayout.addRow(self.featureDict[s])

        self.lblPercentTest = QLabel('Percentage for Test :')
        self.lblPercentTest.adjustSize()
        self.lblPercentTest.setMaximumWidth(200)

        self.txtPercentTest = QLineEdit(self)
        self.txtPercentTest.setText("20")
        self.txtPercentTest.setMaximumWidth(200)

        self.lblNeighbourCount = QLabel('Neighbours:')
        self.lblNeighbourCount.adjustSize()
        self.lblNeighbourCount.setMaximumWidth(200)

        self.txtNeighbourCount = QLineEdit(self)
        self.txtNeighbourCount.setText("9")
        self.txtNeighbourCount.setMaximumWidth(200)

        self.btnExecute = QPushButton("Run Model")
        self.btnExecute.clicked.connect(self.update)
        self.btnExecute.setMaximumWidth(200)

        self.btnSave = QPushButton("Save Model")
        self.btnSave.clicked.connect(self.saveModel)
        self.btnSave.setEnabled(False)
        self.btnSave.setMaximumWidth(200)

        self.groupBox1Layout.addWidget(self.featureScrollArea, 0, 0, 15, 2)
        self.groupBox1Layout.addWidget(self.lblPercentTest, 15, 0,1,1)
        self.groupBox1Layout.addWidget(self.txtPercentTest, 15, 1,1,1)
        self.groupBox1Layout.addWidget(self.lblNeighbourCount, 16, 0, 1, 1)
        self.groupBox1Layout.addWidget(self.txtNeighbourCount, 16, 1, 1, 1)
        self.groupBox1Layout.addWidget(self.btnExecute, 17, 0,1,1)
        self.groupBox1Layout.addWidget(self.btnSave, 17, 1, 1, 1)

        self.groupBox2 = QGroupBox('Results from the model')
        self.groupBox2Layout = QVBoxLayout()
        self.groupBox2.setLayout(self.groupBox2Layout)
        self.groupBox2.setMinimumSize(400, 50)

        self.lblResults = QLabel('Results:')
        self.lblResults.adjustSize()
        self.txtResults = QPlainTextEdit()

        self.groupBox2Layout.addWidget(self.lblResults)
        self.groupBox2Layout.addWidget(self.txtResults)

        self.groupBox3 = QGroupBox('Summary and Comparison')
        self.groupBox3Layout = QVBoxLayout()
        self.groupBox3.setLayout(self.groupBox3Layout)
        self.groupBox3.setMinimumSize(400, 50)
        self.lbl_current_model_summary = QLabel('Summary:')
        self.current_model_summary = QWidget(self)
        self.current_model_summary.layout = QFormLayout(self.current_model_summary)

        self.txtCurrentAccuracy = QLineEdit()
        self.txtCurrentPrecision = QLineEdit()
        self.txtCurrentRecall = QLineEdit()
        self.txtCurrentF1score = QLineEdit()
        self.current_model_summary.layout.addRow('Accuracy:', self.txtCurrentAccuracy)
        self.current_model_summary.layout.addRow('Precision:', self.txtCurrentPrecision)
        self.current_model_summary.layout.addRow('Recall:', self.txtCurrentRecall)
        self.current_model_summary.layout.addRow('F1 Score:', self.txtCurrentF1score)

        self.lbl_other_models = QLabel('Other Models Accuracy:')
        self.other_models = QWidget(self)
        self.other_models.layout = QFormLayout(self.other_models)

        self.txtAccuracy_rf = QLineEdit()
        self.txtAccuracy_lr = QLineEdit()
        self.txtAccuracy_dt = QLineEdit()
        self.other_models.layout.addRow('Random Forest:', self.txtAccuracy_rf)
        self.other_models.layout.addRow('Logistic Regression:', self.txtAccuracy_lr)
        self.other_models.layout.addRow('Decision Trees:', self.txtAccuracy_dt)

        self.groupBox3Layout.addWidget(self.lbl_current_model_summary)
        self.groupBox3Layout.addWidget(self.current_model_summary)
        self.groupBox3Layout.addWidget(self.lbl_other_models)
        self.groupBox3Layout.addWidget(self.other_models)


        #::--------------------------------------
        # Graphic 1 : Confusion Matrix
        #::--------------------------------------

        self.fig = Figure()
        self.ax1 = self.fig.add_subplot(111)
        self.axes=[self.ax1]
        self.canvas = FigureCanvas(self.fig)

        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.canvas.updateGeometry()

        self.groupBoxG1 = QGroupBox('Confusion Matrix')
        self.groupBoxG1Layout= QVBoxLayout()
        self.groupBoxG1.setLayout(self.groupBoxG1Layout)

        self.groupBoxG1Layout.addWidget(self.canvas)

        #::---------------------------------------
        # Graphic 2 : Accuracy vs. K Value
        #::---------------------------------------

        self.fig2 = Figure()
        self.ax2 = self.fig2.add_subplot(111)
        self.axes2 = [self.ax2]
        self.canvas2 = FigureCanvas(self.fig2)

        self.canvas2.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.canvas2.updateGeometry()

        self.groupBoxG2 = QGroupBox('Accuracy vs. K Value')
        self.groupBoxG2Layout = QVBoxLayout()
        self.groupBoxG2.setLayout(self.groupBoxG2Layout)

        self.groupBoxG2Layout.addWidget(self.canvas2)

        #::-------------------------------------------
        # Graphic 3 : Cross Validation Score
        #::-------------------------------------------

        self.fig3 = Figure()
        self.ax3 = self.fig3.add_subplot(111)
        self.axes3 = [self.ax3]
        self.canvas3 = FigureCanvas(self.fig3)

        self.canvas3.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.canvas3.updateGeometry()

        self.groupBoxG3 = QGroupBox('Cross Validation Score')
        self.groupBoxG3Layout = QVBoxLayout()
        self.groupBoxG3.setLayout(self.groupBoxG3Layout)
        self.groupBoxG3Layout.addWidget(self.canvas3)

        #::--------------------------------------------
        # Graphic 4 : ROC Curve by class
        #::--------------------------------------------

        self.fig4 = Figure()
        self.ax4 = self.fig4.add_subplot(111)
        self.axes4 = [self.ax4]
        self.canvas4 = FigureCanvas(self.fig4)

        self.canvas4.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.canvas4.updateGeometry()

        self.groupBoxG4 = QGroupBox('ROC Curve by Class')
        self.groupBoxG4Layout = QVBoxLayout()
        self.groupBoxG4.setLayout(self.groupBoxG4Layout)
        self.groupBoxG4Layout.addWidget(self.canvas4)

        #::-------------------------------------------------
        # End of graphs
        #::-------------------------------------------------

        self.layout.addWidget(self.groupBox1,0,0,2,1)
        self.layout.addWidget(self.groupBoxG1,0,1,1,1)
        self.layout.addWidget(self.groupBoxG3,0,2,1,1)
        self.layout.addWidget(self.groupBox2,0,3,1,1)
        self.layout.addWidget(self.groupBoxG2,1,1,1,1)
        self.layout.addWidget(self.groupBoxG4,1,2,1,1)
        self.layout.addWidget(self.groupBox3,1,3,1,1)

        self.setCentralWidget(self.main_widget)
        self.resize(1800, 900)
        self.show()

    def update(self):
        self.list_corr_features = pd.DataFrame([])
        # self.featureCount=0

        for i in range(len(featuresList)):
            s = "f" + str(i) + "CheckBox"
            if (self.featureDict[s].isChecked()):
                if len(self.list_corr_features) == 0:
                    self.list_corr_features = data[featuresList[i]]
                else:
                    self.list_corr_features = pd.concat([self.list_corr_features, data[featuresList[i]]], axis=1)

        try:
            vtest_per = float(self.txtPercentTest.text())
            if(vtest_per<100 and vtest_per>0):
                pass
            else:
                vtest_per = 20
                self.txtPercentTest.setText(str(vtest_per))
        except:
            vtest_per=20
            self.txtPercentTest.setText(str(vtest_per))

        try:
            neighbour_input = round(float(self.txtNeighbourCount.text()))
            if (neighbour_input < 100 and neighbour_input > 0):
                pass
            else:
                neighbour_input = 9
                self.txtNeighbourCount.setText(str(neighbour_input))
        except:
            neighbour_input=9
            self.txtNeighbourCount.setText(str(neighbour_input))

        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()
        self.ax4.clear()
        self.txtResults.clear()
        self.txtResults.setUndoRedoEnabled(False)

        vtest_per = vtest_per / 100

        X_dt = self.list_corr_features
        y_dt = data[targetVariable]
        X_columns = X_dt.columns.tolist()
        labelencoder_columns = list(set(X_columns) & set(ordinalFeatures))
        one_hot_encoder_columns = list(set(X_columns) & set(nominalFeatures))

        class_le = LabelEncoder()
        class_ohe = OneHotEncoder()

        temp = X_columns.copy()
        for ohe_val in one_hot_encoder_columns:
            temp.remove(ohe_val)
        temp_X_dt = X_dt[temp]
        for le_val in labelencoder_columns:
            temp_X_dt[le_val] = class_le.fit_transform(temp_X_dt[le_val])
        X_dt = pd.concat((temp_X_dt, pd.get_dummies(X_dt[one_hot_encoder_columns])), 1)
        print("Y_DT(Prev): Values:\n", y_dt, "\n\n")
        y_dt = class_le.fit_transform(y_dt)
        print("Y_DT(After): Values:\n", y_dt, "\n\n")
        X_train, X_test, y_train, y_test = train_test_split(X_dt, y_dt, test_size=vtest_per, random_state=500)

        # specify knn classifier
        self.clf_knn = KNeighborsClassifier(n_neighbors=neighbour_input)

        # perform training
        self.clf_knn.fit(X_train, y_train)

        #-----------------------------------------------------------------------

        # prediction on test using all features
        y_pred = self.clf_knn.predict(X_test)
        y_pred_score = self.clf_knn.predict_proba(X_test)

        # confusion matrix for RandomForest
        conf_matrix = confusion_matrix(y_test, y_pred)

        # clasification report

        self.ff_class_rep = classification_report(y_test, y_pred)
        self.txtResults.appendPlainText(self.ff_class_rep)

        # accuracy score

        self.ff_accuracy_score = accuracy_score(y_test, y_pred) * 100
        self.txtCurrentAccuracy.setText(str(self.ff_accuracy_score))

        # precision score

        self.ff_precision_score = precision_score(y_test, y_pred,average='micro')*100
        self.txtCurrentPrecision.setText(str(self.ff_precision_score))

        # recall score

        self.ff_recall_score = recall_score(y_test, y_pred,average='micro') * 100
        self.txtCurrentRecall.setText(str(self.ff_recall_score))

        # f1_score

        self.ff_f1_score = f1_score(y_test, y_pred,average='micro')
        self.txtCurrentF1score.setText(str(self.ff_f1_score))

        #::------------------------------------
        ##  Graph1 :
        ##  Confusion Matrix
        #::------------------------------------
        class_names1 = list([" "])
        class_names1 += list(targetClasses)
        class_names1 = ['\n'.join(wrap(l, 12)) for l in class_names1]

        self.ax1.matshow(conf_matrix, cmap=plt.cm.get_cmap('Blues', 14))
        self.ax1.set_yticklabels(class_names1, fontsize=8)
        self.ax1.set_xticklabels(class_names1, rotation=90, fontsize=8)
        self.ax1.set_xlabel('Predicted label')
        self.ax1.set_ylabel('True label')

        for i in range(len(targetClasses)):
            for j in range(len(targetClasses)):
                self.ax1.text(j, i, str(conf_matrix[i][j]))

        self.fig.tight_layout()
        self.fig.canvas.draw_idle()

        #::----------------------------------------
        ## Graph 2 - Accuracy vs K Values
        #::----------------------------------------

        accuracy_test = []
        accuracy_train = []

        for i in range(1, 20,2):
            self.knn_graph = KNeighborsClassifier(n_neighbors=i)
            self.knn_graph.fit(X_train, y_train)
            pred_i = self.knn_graph.predict(X_train)
            accuracy_train.append(accuracy_score(y_train, pred_i))
            pred_i = self.knn_graph.predict(X_test)
            accuracy_test.append(accuracy_score(y_test, pred_i))

        self.ax2.plot(range(1, 20,2),accuracy_train , color='blue', marker='o',
                 markerfacecolor='orange', markersize=5,label="Train Accuracy")
        self.ax2.plot(range(1, 20,2), accuracy_test, color='red', marker='o',
                 markerfacecolor='orange', markersize=5, label="Test Accuracy")
        self.ax2.set_xlabel('K')
        self.ax2.axes.set_xticks(np.arange(1,20,2))
        self.ax2.set_xlabel('K')
        self.ax2.set_ylabel('Accuracy')
        self.ax2.legend(loc="lower right")

        self.fig2.tight_layout()
        self.fig2.canvas.draw_idle()

        #::----------------------------------------
        ## Graph 3 - Cross Validation Score
        #::----------------------------------------

        scores_arr = []

        for val in (X_train.columns):
            cvs_X = X_train[val].values.reshape(-1, 1)
            scores = cross_val_score(self.clf_knn, cvs_X, y_train, cv=10)
            scores_arr.append(scores.mean())

        f_importances = pd.Series(scores_arr, X_train.columns)

        f_importances.sort_values(ascending=False, inplace=True)
        f_importances=f_importances[0:20]
        X_Features = f_importances.index
        y_Importance = list(f_importances)
        X_Features = ['\n'.join(wrap(l, 20)) for l in X_Features]

        max_value = f_importances.max()
        min_value = f_importances.min()
        self.ax3.barh(X_Features, y_Importance)
        self.ax3.set_xlim(min_value - (min_value * 0.05), max_value + (max_value * 0.05))
        self.ax3.tick_params(labelsize=8)

        # show the plot
        self.fig3.tight_layout()
        self.fig3.canvas.draw_idle()

        #::-----------------------------------------------------
        # Graph 4 - ROC Curve by Class
        #::-----------------------------------------------------

        if len(targetClasses) > 2:
            y_test_bin = label_binarize(y_dt, classes=range(len(targetClasses)))
            n_classes = y_test_bin.shape[1]
            print(n_classes)
            X_train_temp, X_test_temp, y_train_temp, y_test_temp = train_test_split(X_dt, y_test_bin,
                                                                                    test_size=vtest_per,
                                                                                    random_state=500)

            classifier = OneVsRestClassifier(self.clf_knn)
            y_score = classifier.fit(X_train_temp, y_train_temp).predict_proba(X_test_temp)
            # str_classes = targetClasses
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            for i in range(n_classes):
                fpr[i], tpr[i], _ = roc_curve(y_test_temp[:, i], y_score[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])

            for i in range(n_classes):
                self.ax4.plot(fpr[i], tpr[i],
                              label='{0} (area = {1:0.2f})'
                                    ''.format(targetClasses[i], roc_auc[i]))
        else:
            y_test_bin = pd.get_dummies(y_test).to_numpy()
            n_classes = y_test_bin.shape[1]

            # From the sckict learn site
            # https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            for i in range(n_classes):
                fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_score[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])
            # print(pd.get_dummies(y_test).to_numpy().ravel())

            # print("\n\n********************************\n\n")
            # print(y_pred_score.ravel())
            # Compute micro-average ROC curve and ROC area
            fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_pred_score.ravel())

            roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
            lw = 2
            str_classes = ['No', 'Yes']
            colors = cycle(['magenta', 'darkorange'])
            for i, color in zip(range(n_classes), colors):
                self.ax4.plot(fpr[i], tpr[i], color=color, lw=lw,
                              label='{0} (area = {1:0.2f})'
                                    ''.format(targetClasses[i], roc_auc[i]))

        self.ax4.plot([0, 1], [0, 1], 'k--')
        self.ax4.set_xlim([0.0, 1.0])
        self.ax4.set_ylim([0.0, 1.05])
        self.ax4.set_xlabel('False Positive Rate')
        self.ax4.set_ylabel('True Positive Rate')
        self.ax4.legend(loc="lower right")

        # show the plot
        self.fig4.tight_layout()
        self.fig4.canvas.draw_idle()

        #::-----------------------------------------------------
        # Other Models Comparison
        #::-----------------------------------------------------

        self.other_clf_rf=RandomForestClassifier(n_estimators=100, random_state=500)
        self.other_clf_rf.fit(X_train, y_train)
        y_pred_rf = self.other_clf_rf.predict(X_test)
        self.accuracy_rf=accuracy_score(y_test, y_pred_rf) *100
        self.txtAccuracy_rf.setText(str(self.accuracy_rf))

        self.other_clf_dt = DecisionTreeClassifier(criterion="gini")
        self.other_clf_dt.fit(X_train, y_train)
        y_pred_dt = self.other_clf_dt.predict(X_test)
        self.accuracy_dt = accuracy_score(y_test, y_pred_dt) * 100
        self.txtAccuracy_dt.setText(str(self.accuracy_dt))

        self.other_clf_lr = LogisticRegression(random_state=500)
        self.other_clf_lr.fit(X_train, y_train)
        y_pred_lr = self.other_clf_lr.predict(X_test)
        self.accuracy_lr = accuracy_score(y_test, y_pred_lr) * 100
        self.txtAccuracy_lr.setText(str(self.accuracy_lr))

        self.btnSave.setEnabled(True)

    def saveModel(self):
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getSaveFileName(self, "QFileDialog.getSaveFileName()", "",
                                                  "SAV Files (*.sav)", options=options)
        if fileName:
            # print(fileName)
            # fileName += '.out'
            pickle.dump(self.clf_rf, open(fileName, 'wb'))

    def multiclass_roc_auc_score(self, y_test, y_pred, average="micro"):
        lb = LabelBinarizer()
        lb.fit(y_test)
        y_test = lb.transform(y_test)
        y_pred = lb.transform(y_pred)
        return roc_auc_score(y_test, y_pred, average=average)

    def multiclass_roc_curve(self, y_test, y_pred, average="micro"):
        lb = LabelBinarizer()
        lb.fit(y_test)
        y_test = lb.transform(y_test)
        y_pred = lb.transform(y_pred)
        return roc_curve(y_test, y_pred, average=average)


class PlotCanvas(FigureCanvas):

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)

        FigureCanvas.__init__(self, fig)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self,
                                   QSizePolicy.Expanding,
                                   QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

    def plot(self):
        self.ax = self.figure.add_subplot(111)


class CanvasWindow(QMainWindow):

    def __init__(self, parent=None):
        super(CanvasWindow, self).__init__(parent)

        self.left = 200
        self.top = 200
        self.Title = 'Distribution'
        self.width = 500
        self.height = 500
        self.initUI()

    def initUI(self):

        self.setWindowTitle(self.Title)
        self.setStyleSheet(font_size_window)

        self.setGeometry(self.left, self.top, self.width, self.height)

        self.m = PlotCanvas(self, width=5, height=4)
        self.m.move(0, 30)


class App(QMainWindow):

    def __init__(self):
        super().__init__()
        self.left = 200
        self.top = 200
        self.width = 1000
        self.height = 500
        self.Title = 'Employee Attrition Evaluator & Predictor '
        label = QLabel(self, alignment=Qt.AlignCenter)
        pixmap = QPixmap('Background Image.png')
        label.setPixmap(pixmap)
        self.setCentralWidget(label)
        self.resize(pixmap.width(), pixmap.height())
        self.initUI()

    def initUI(self):
        #::-------------------------------------------------
        # Creates the menu and the items
        #::-------------------------------------------------
        self.setWindowTitle(self.Title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        #::-----------------------------
        # Create the menu bar
        # and three items for the menu, File, EDA Analysis and ML Models
        #::-----------------------------
        mainMenu = self.menuBar()
        mainMenu.setStyleSheet('background-color: lightblue')

        fileMenu = mainMenu.addMenu('File')
        EDAMenu = mainMenu.addMenu('EDA Analysis')
        MLModelMenu = mainMenu.addMenu('ML Models')


        #::--------------------------------------
        # Exit application
        # Creates the actions for the fileMenu item
        #::--------------------------------------

        exitButton = QAction(QIcon('enter.png'), 'Exit', self)
        exitButton.setShortcut('Ctrl+Q')
        exitButton.setStatusTip('Exit application')
        exitButton.triggered.connect(self.close)

        #::----------------------------------------
        # Open and Save File
        #::----------------------------------------

        file1Button = QAction(QIcon('analysis.png'), 'Open Data', self)
        file1Button.setStatusTip('Opens the data')
        file1Button.triggered.connect(self.show_table)
        fileMenu.addAction(file1Button)

        file2Button = QAction(QIcon('analysis.png'), 'Explore Variables', self)
        file2Button.setStatusTip('Variables Information')
        file2Button.triggered.connect(self.variables_info)
        fileMenu.addAction(file2Button)

        file3Button = QAction(QIcon('analysis.png'), 'Open Previous File', self)
        file3Button.setStatusTip('Opens the file')
        file3Button.triggered.connect(self.open_previous)
        fileMenu.addAction(file3Button)


        fileMenu.addAction(exitButton)

        #::----------------------------------------
        # EDA analysis
        # Creates the actions for the EDA Analysis item
        # Variable Distribution : Distribution of Continuous Variables
        # Variable Relation : Shows the relation between continuous - continuous variables (scatter plot) and categorical - continuous variables (box plot)
        # Attrition Relation : Compares the variables on the basis of Attrition Yes and Attrition No
        #::----------------------------------------

        EDA1Button = QAction(QIcon('analysis.png'),'Variable Distribution', self)
        EDA1Button.setStatusTip('Presents the variable distribution')
        EDA1Button.triggered.connect(self.EDA1)
        EDAMenu.addAction(EDA1Button)

        EDA2Button = QAction(QIcon('analysis.png'), 'Variable Relation', self)
        EDA2Button.setStatusTip('Presents the relationship between variables')
        EDA2Button.triggered.connect(self.EDA2)
        EDAMenu.addAction(EDA2Button)

        EDA4Button = QAction(QIcon('analysis.png'), 'Attrition Relation', self)
        EDA4Button.setStatusTip('Compares the variables with respect to Attrition')
        EDA4Button.triggered.connect(self.EDA4)
        EDAMenu.addAction(EDA4Button)

        #::--------------------------------------------------
        # ML Models for prediction
        # There are four models
        #       Decision Tree
        #       Random Forest
        #       Logistic Regression
        #       KNN

        #::--------------------------------------------------
        # Decision Tree Classifier
        #::--------------------------------------------------
        MLModel1Button =  QAction(QIcon(), 'Decision Tree', self)
        MLModel1Button.setStatusTip('ML algorithm ')
        MLModel1Button.triggered.connect(self.MLDT)

        #::------------------------------------------------------
        # Random Forest Classifier
        #::------------------------------------------------------
        MLModel2Button = QAction(QIcon(), 'Random Forest Classifier', self)
        MLModel2Button.setStatusTip('Random Forest Classifier ')
        MLModel2Button.triggered.connect(self.MLRF)

        #::--------------------------------------------------
        # Logistic Regression Classifier
        #::--------------------------------------------------
        MLModel3Button = QAction(QIcon(), 'Logistic Regression', self)
        MLModel3Button.setStatusTip('Logistic Regression')
        MLModel3Button.triggered.connect(self.MLLR)

        #::--------------------------------------------------
        # KNN Classifier
        #::--------------------------------------------------
        MLModel4Button = QAction(QIcon(), 'K- Nearest Neigbor', self)
        MLModel4Button.setStatusTip('K- Nearest Neigbor')
        MLModel4Button.triggered.connect(self.MLKNN)

        MLModelMenu.addAction(MLModel1Button)
        MLModelMenu.addAction(MLModel2Button)
        MLModelMenu.addAction(MLModel3Button)
        MLModelMenu.addAction(MLModel4Button)

        self.dialogs = list()

    def file1(self):
        #def openFileNameDialog(self):
        options = QFileDialog.Options()
        # options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self, "Open File", "",
                                                  "CSV (*.csv);; MS Excel (*.xlsx *xls);;All Files (*)",
                                                  options=options)
        if fileName:
            print(fileName)
            self.show_table()

    def show_table(self):
        dialog = showTable()
        self.dialogs.append(dialog)
        dialog.show()
        # self.pathLE.setText(fileName)
        # df = pd.read_csv(fileName)
        # model = PandasModel(df)
        # self.pandasTv.setModel(model)

    def variables_info(self):
        dialog = VariableInformation()
        self.dialogs.append(dialog)
        dialog.show()
        # self.pathLE.setText(fileName)
        # df = pd.read_csv(fileName)
        # model = PandasModel(df)
        # self.pandasTv.setModel(model)

    def open_previous(self):
        fileName, _ = QFileDialog.getOpenFileName(self, "Open File", "", "Shelve Files (*.out.db)");
        fileName=fileName[:-3]
        my_shelf = shelve.open(fileName)
        for key in my_shelf:
            try:
                globals()[key] = my_shelf[key]
            except:
                i=1
        my_shelf.close()

    def EDA1(self):
        dialog = VariableDistribution()
        self.dialogs.append(dialog)
        dialog.show()

    def EDA2(self):
        dialog = VariableRelation()
        self.dialogs.append(dialog)
        dialog.show()

    def EDA4(self):
        dialog = TargetRelation()
        self.dialogs.append(dialog)
        dialog.show()

    def MLDT(self):
        dialog = DecisionTree()
        self.dialogs.append(dialog)
        dialog.show()

    def MLRF(self):
        dialog = RandomForest()
        self.dialogs.append(dialog)
        dialog.show()

    def MLLR(self):
        dialog = LogisticRegressionClassifier()
        self.dialogs.append(dialog)
        dialog.show()

    def MLKNN(self):
        dialog = KNNClassifier()
        self.dialogs.append(dialog)
        dialog.show()

def main():
    #::-------------------------------------------------
    # Initiates the application
    #::-------------------------------------------------
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    ex = App()
    ex.show()
    ex.showMaximized()
    sys.exit(app.exec_())


def attrition_data():
    #::--------------------------------------------------
    # Loads the dataset HR_Employee-Attrition.csv
    # Specifies columns into several buckets such as continuous_features, categorical_features
    # personal_features, organisation_features, commution_features, satisfaction_features
    #::--------------------------------------------------
    #global data

    global features_list
    #global class_names
    #global target_variable
    #global personal_features
    #global organisation_features
    #global commution_features
    #global satisfaction_features
    global ratio_features
    global ordinal_features
    global nominal_features

    #data = pd.read_csv('HR-Employee-Attrition.csv')
    all_columns = data.columns.tolist()

    #all_columns.remove("Attrition")
    features_list=all_columns.copy()

    #target_variable="Attrition"
    #class_names = ['No', 'Yes']
    label_encoder_variables =["Education","JobLevel"]
    hot_encoder_variables=["BusinessTravel","Department","EducationField","Gender","JobRole","MaritalStatus","OverTime","StockOptionLevel"]
    personal_features=["Age","Education","EducationField","MaritalStatus","Gender"]
    organisation_features=["DailyRate","Department","HourlyRate","JobInvolvement","JobLevel","JobRole","MonthlyIncome","MonthlyRate","NumCompaniesWorked","OverTime","PercentSalaryHike","PerformanceRating","StockOptionLevel","TotalWorkingYears","TrainingTimesLastYear","YearsAtCompany","YearsInCurrentRole","YearsSinceLastPromotion","YearsWithCurrManager"]
    commution_features=["BusinessTravel","DistanceFromHome"]
    satisfaction_features=["EnvironmentSatisfaction","JobSatisfaction","RelationshipSatisfaction","WorkLifeBalance"]
    continuous_features=["Age","DistanceFromHome","DailyRate","HourlyRate","MonthlyIncome","MonthlyRate","NumCompaniesWorked","PercentSalaryHike","TotalWorkingYears","TrainingTimesLastYear","YearsAtCompany","YearsInCurrentRole","YearsSinceLastPromotion","YearsWithCurrManager"]
    categorical_features=list(set(features_list) - set(continuous_features))

if __name__ == '__main__':
    #::------------------------------------
    # First reads the data then calls for the application
    #::------------------------------------
    #attrition_data()
    main()
