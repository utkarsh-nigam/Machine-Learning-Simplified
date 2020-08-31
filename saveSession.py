from PyQt5.QtWidgets import (QMainWindow, QApplication, QWidget, QPushButton, QAction, QComboBox, QLabel,
                             QGridLayout, QCheckBox, QGroupBox, QVBoxLayout, QHBoxLayout, QLineEdit, QPlainTextEdit,
                             QInputDialog, QFileDialog, QTableView, QFormLayout, QScrollArea, QSpinBox)

from PyQt5.QtGui import QIcon
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtCore import Qt, QAbstractTableModel, QVariant, QModelIndex
import sys
import shelve


import pandas as pd
df=pd.read_csv("HR-Employee-Attrition.csv")
print(df.columns)
list=df.dtypes.tolist()


import shelve
import pandas as pd
df=pd.read_csv("HR-Employee-Attrition.csv")
print(df.columns)
newList=df.dtypes.tolist()
print(newList)
filename='shelveNew.out'
my_shelf = shelve.open(filename,'n') # 'n' for new
print(globals())
print("\n\n")
print(globals().keys())
print(dir())
globalKeys=list(globals().keys())
for key in globalKeys:
    print(key)
    try:
        my_shelf[key] = globals()[key]
    except TypeError:
        #
        # __builtins__, my_shelf, and imported modules can not be shelved.
        #
        print('ERROR shelving: {0}'.format(key))
my_shelf.close()







import shelve



filename='temp.out'
my_shelf = shelve.open(filename)
for key in my_shelf:
    print(key)
    try:
        globals()[key]=my_shelf[key]
    except:
        i=1
my_shelf.close()

print(df.head())
print(newList)





class Main(QMainWindow):
    def __init__(self):
        super(Main, self).__init__()
        self.Title = "File Browser"
        self.initUi()

    def initUi(self):
        global df
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getSaveFileName(self,"QFileDialog.getSaveFileName()", "",
                                                  "Shelve Files (*.out)", options=options)
        if fileName:
            print(fileName)
            fileName += '.out'
            my_shelf = shelve.open(fileName, 'n')  # 'n' for new

            for key in dir():
                try:
                    my_shelf[key] = globals()[key]
                except TypeError:
                    #
                    # __builtins__, my_shelf, and imported modules can not be shelved.
                    #
                    print('ERROR shelving: {0}'.format(key))
            my_shelf.close()



    #     self.i=0
    #     self.catDict=dict()
    #     self.setWindowTitle(self.Title)
    #     self.main_widget = QWidget(self)
    #     self.layout = QGridLayout(self.main_widget)
    #
    #     # main button
    #     # self.addButton = QPushButton('button to add other widgets')
    #     # self.addButton.clicked.connect(self.addWidget)
    #     # self.delButton = QPushButton('button to del other widgets')
    #     # self.delButton.clicked.connect(self.delWidget)
    #
    #     # scroll area widget contents - layout
    #     # self.scrollLayout = QFormLayout()
    #     #
    #     # # scroll area widget contents
    #     # self.scrollWidget = QWidget()
    #     # self.scrollWidget.setLayout(self.scrollLayout)
    #     #
    #     # # scroll area
    #     # self.scrollArea = QScrollArea()
    #     # self.scrollArea.setWidgetResizable(True)
    #     # self.scrollArea.setWidget(self.scrollWidget)
    #
    #     # main layout
    #     # self.mainLayout = QGridLayout()
    #
    #
    #
    #     self.catCurrentCount = 0
    #     self.catDict = dict()
    #     self.catCount = QSpinBox()
    #     self.catCount.setRange(2, 5)
    #     self.catCount.setValue(2)
    #     self.catCount.valueChanged.connect(self.catCountUpdate)
    #
    #     self.catScrollLayout = QFormLayout()
    #
    #     # scroll area widget contents
    #     self.catScrollWidget = QWidget()
    #     self.catScrollWidget.setLayout(self.catScrollLayout)
    #
    #     # scroll area
    #     self.catScrollArea = QScrollArea()
    #     self.catScrollArea.setWidgetResizable(True)
    #     self.catScrollArea.setWidget(self.catScrollWidget)
    #
    #     # add all main to the main vLayout
    #     self.layout.addWidget(QLabel("Choose number of categories:"), 0, 0, 1, 1)
    #     self.layout.addWidget(self.catCount, 0, 1, 1, 1)
    #     self.layout.addWidget(self.catScrollArea, 1, 0, 7, 2)
    #
    #
    #     # central widget
    #     # self.centralWidget = QWidget()
    #     # self.centralWidget.setLayout(self.mainLayout)
    #
    #     # set central widget
    #     self.setCentralWidget(self.main_widget)
    #     self.resize(1800, 900)
    #     self.show()
    #     self.addWidget()
    #     self.addWidget()
    #
    # def catCountUpdate(self):
    #     if self.catCount.value() > self.catCurrentCount:
    #         while (self.catCount.value() > self.catCurrentCount):
    #             #self.catCurrentCount += 1
    #             self.addWidget()
    #     elif self.catCount.value() < self.catCurrentCount:
    #         while (self.catCount.value() < self.catCurrentCount):
    #             self.delWidget()
    #             #self.catCurrentCount -= 1
    #
    #
    # def addWidget(self):
    #     self.catCurrentCount += 1
    #     s = "s" + str(self.catCurrentCount) +"Layout"
    #     self.catDict[s] = QLineEdit("Enter Category " + str(self.catCurrentCount))
    #     self.catDict[s].setEnabled(True)
    #     # self.s=QLineEdit('I am in Test widget')
    #     self.catScrollLayout.addRow(self.catDict[s])
    #
    #     # self.i+=1
    #     # self.txtEstimatorCount.setText("35")
    #
    #
    # def delWidget(self):
    #     t = "s" + str(self.catCurrentCount)
    #     print(self.catDict[t+"Layout"].text())
    #     self.catDict[t+"Layout"].deleteLater()
    #     self.catCurrentCount -= 1
    #     # to_delete = self.scrollLayout.takeAt(self.scrollLayout.count() - 1)
    #     # if to_delete is not None:
    #     #     while to_delete.count():
    #     #         item = to_delete.takeAt(0)
    #     #         widget = item.widget()
    #     #         if widget is not None:
    #     #             widget.deleteLater()
    #     #         else:
    #     #             pass



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



global i

def t1():
    global i
    i=2

def t2():
    global i
    i+=2

def t3():
    global i
    print(i)

t1()
t2()
t3()


i=[1,2,3,4,5]
j=[2,3,4,5,6]

print(i+j)

import textwrap as wrap
import matplotlib.pyplot as plt
import numpy as np
a = np.random.rand(4)
b = np.random.rand(4)
c = np.random.rand(4)
x10 = ["wdfvweilyfbeifbjcb.kjec","ewdeyulc3vcyecbe3ocu3jcl","cju3vc3iekhc31bkcj","iewiebcliecewbcliwe"]
plt.barh(x10, a, color = 'b', label="A" )
plt.barh(x10, b, color = 'g', left = a, label="B" )
plt.barh(x10, c, color = 'r', left = a + b, label="C" )
plt.yticks(['\n'.join(wrap(l, 12)) for l in x10])
plt.tick_params(labelsize=8)
#plt.ylabel(fontdict=)
plt.legend()
plt.show()
print('#',50*"-")


a=['Yes', 'No']
print(sorted(a))







# stacked bar plot
import numpy as np
import matplotlib.pyplot as plt

# Get values from the group and categories
quarter = ["Q1", "Q2", "Q3", "Q4"]
jeans = [100, 75, 5, 133]
tshirt = [44, 120, 150, 33]
formal_shirt = [70, 90, 111, 80]

# add colors
#colors = ['#FF9999', '#00BFFF', '#C1FFC1', '#CAE1FF', '#FFDEAD']
# The position of the bars on the x-axis
r = range(len(quarter))
barWidth = 0.8
# plot bars
plt.figure(figsize=(10, 7))
ax1 = plt.bar(r, jeans, edgecolor='white', width=barWidth, label="jeans")
ax2 = plt.bar(r, tshirt, bottom=np.array(jeans), edgecolor='white', width=barWidth, label='tshirt')
ax3 = plt.bar(r, formal_shirt, bottom=np.array(jeans) + np.array(tshirt), edgecolor='white',
              width=barWidth, label='formal shirt')
plt.legend()
# Custom X axis
plt.xticks(r, quarter, fontweight='bold')
plt.ylabel("sales")
print(ax1)
for r1, r2, r3 in zip(ax1, ax2, ax3):
    h1 = r1.get_height()
    h2 = r2.get_height()
    h3 = r3.get_height()
    print (r1,h1)
    plt.text(r1.get_x() + r1.get_width() / 2., h1 / 2., str(h1)+"k", ha="center", va="center", color="white", fontsize=16,
             fontweight="bold")
    plt.text(r2.get_x() + r2.get_width() / 2., h1 + h2 / 2., "%d" % h2, ha="center", va="center", color="white",
             fontsize=16, fontweight="bold")
    plt.text(r3.get_x() + r3.get_width() / 2., h1 + h2 + h3 / 2., "%d" % h3, ha="center", va="center", color="white",
             fontsize=16, fontweight="bold")

plt.show()
# You can replace "%d" % h1 with "{}".format(h1)

for r1, r2, r3 in zip(ax1, ax2, ax3):
    h1 = r1.get_height()
    h2 = r2.get_height()
    h3 = r3.get_height()
    print (r1,h1)
    print(r2, h2)
    print(r3, h3)



import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier

iris = datasets.load_iris()
print(iris)
X = iris.data
y = iris.target
print (y)

# Binarize the output
y = label_binarize(y, classes=[0,1,2])
print(y)
n_classes = y.shape[1]
print(y)
print(n_classes)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5, random_state=0)

classifier = OneVsRestClassifier(RandomForestClassifier(n_estimators=100))
y_score = classifier.fit(X_train, y_train).predict_proba(X_test)
print(y_score)
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

for i in range(n_classes):
    plt.plot(fpr[i], tpr[i],
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([-0.05, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic for multi-class data')
plt.legend(loc="lower right")
plt.show()

s=["H" "E"]
t=[""]
t+=s
print(t)


from textwrap import wrap
labels=["efwefwfwqkfjeqwnf1ejfnrocjqerljc", "awejvdfwehfvefhwekfwhfjwf"]
print(labels)
labels = ['\n'.join(wrap(l, 20)) for l in labels]
print(labels)