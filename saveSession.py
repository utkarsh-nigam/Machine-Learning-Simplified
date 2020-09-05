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









import matplotlib.pyplot as plt

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_blobs
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss

np.random.seed(0)

# Generate data
X, y = make_blobs(n_samples=1000, random_state=42, cluster_std=5.0)
X_train, y_train = X[:600], y[:600]
X_valid, y_valid = X[600:800], y[600:800]
X_train_valid, y_train_valid = X[:800], y[:800]
X_test, y_test = X[800:], y[800:]

# Train uncalibrated random forest classifier on whole train and validation
# data and evaluate on test data
clf = LogisticRegression()
clf.fit(X_train_valid, y_train_valid)
clf_probs = clf.predict_proba(X_test)
score = log_loss(y_test, clf_probs)

# Train random forest classifier, calibrate on validation data and evaluate
# on test data
clf = LogisticRegression()
clf.fit(X_train, y_train)
clf_probs = clf.predict_proba(X_test)
sig_clf = CalibratedClassifierCV(clf, method="sigmoid", cv="prefit")
sig_clf.fit(X_valid, y_valid)
sig_clf_probs = sig_clf.predict_proba(X_test)
sig_score = log_loss(y_test, sig_clf_probs)

# Plot changes in predicted probabilities via arrows
plt.figure()
colors = ["r", "g", "b"]
for i in range(clf_probs.shape[0]):
    plt.arrow(clf_probs[i, 0], clf_probs[i, 1],
              sig_clf_probs[i, 0] - clf_probs[i, 0],
              sig_clf_probs[i, 1] - clf_probs[i, 1],
              color=colors[y_test[i]], head_width=1e-2)

# Plot perfect predictions
plt.plot([1.0], [0.0], 'ro', ms=20, label="Class 1")
plt.plot([0.0], [1.0], 'go', ms=20, label="Class 2")
plt.plot([0.0], [0.0], 'bo', ms=20, label="Class 3")

# Plot boundaries of unit simplex
plt.plot([0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], 'k', label="Simplex")

# Annotate points on the simplex
plt.annotate(r'($\frac{1}{3}$, $\frac{1}{3}$, $\frac{1}{3}$)',
             xy=(1.0/3, 1.0/3), xytext=(1.0/3, .23), xycoords='data',
             arrowprops=dict(facecolor='black', shrink=0.05),
             horizontalalignment='center', verticalalignment='center')
plt.plot([1.0/3], [1.0/3], 'ko', ms=5)
plt.annotate(r'($\frac{1}{2}$, $0$, $\frac{1}{2}$)',
             xy=(.5, .0), xytext=(.5, .1), xycoords='data',
             arrowprops=dict(facecolor='black', shrink=0.05),
             horizontalalignment='center', verticalalignment='center')
plt.annotate(r'($0$, $\frac{1}{2}$, $\frac{1}{2}$)',
             xy=(.0, .5), xytext=(.1, .5), xycoords='data',
             arrowprops=dict(facecolor='black', shrink=0.05),
             horizontalalignment='center', verticalalignment='center')
plt.annotate(r'($\frac{1}{2}$, $\frac{1}{2}$, $0$)',
             xy=(.5, .5), xytext=(.6, .6), xycoords='data',
             arrowprops=dict(facecolor='black', shrink=0.05),
             horizontalalignment='center', verticalalignment='center')
plt.annotate(r'($0$, $0$, $1$)',
             xy=(0, 0), xytext=(.1, .1), xycoords='data',
             arrowprops=dict(facecolor='black', shrink=0.05),
             horizontalalignment='center', verticalalignment='center')
plt.annotate(r'($1$, $0$, $0$)',
             xy=(1, 0), xytext=(1, .1), xycoords='data',
             arrowprops=dict(facecolor='black', shrink=0.05),
             horizontalalignment='center', verticalalignment='center')
plt.annotate(r'($0$, $1$, $0$)',
             xy=(0, 1), xytext=(.1, 1), xycoords='data',
             arrowprops=dict(facecolor='black', shrink=0.05),
             horizontalalignment='center', verticalalignment='center')
# Add grid
plt.grid(False)
for x in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    plt.plot([0, x], [x, 0], 'k', alpha=0.2)
    plt.plot([0, 0 + (1-x)/2], [x, x + (1-x)/2], 'k', alpha=0.2)
    plt.plot([x, x + (1-x)/2], [0, 0 + (1-x)/2], 'k', alpha=0.2)

plt.title("Change of predicted probabilities after sigmoid calibration")
plt.xlabel("Probability class 1")
plt.ylabel("Probability class 2")
plt.xlim(-0.05, 1.05)
plt.ylim(-0.05, 1.05)
plt.legend(loc="best")

print("Log-loss of")
print(" * uncalibrated classifier trained on 800 datapoints: %.3f "
      % score)
print(" * classifier trained on 600 datapoints and calibrated on "
      "200 datapoint: %.3f" % sig_score)

# Illustrate calibrator
plt.figure()
# generate grid over 2-simplex
p1d = np.linspace(0, 1, 20)
p0, p1 = np.meshgrid(p1d, p1d)
p2 = 1 - p0 - p1
p = np.c_[p0.ravel(), p1.ravel(), p2.ravel()]
p = p[p[:, 2] >= 0]

calibrated_classifier = sig_clf.calibrated_classifiers_[0]
prediction = np.vstack([calibrator.predict(this_p)
                        for calibrator, this_p in
                        zip(calibrated_classifier.calibrators_, p.T)]).T
prediction /= prediction.sum(axis=1)[:, None]

# Plot modifications of calibrator
for i in range(prediction.shape[0]):
    plt.arrow(p[i, 0], p[i, 1],
              prediction[i, 0] - p[i, 0], prediction[i, 1] - p[i, 1],
              head_width=1e-2, color=colors[np.argmax(p[i])])
# Plot boundaries of unit simplex
plt.plot([0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], 'k', label="Simplex")

plt.grid(False)
for x in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    plt.plot([0, x], [x, 0], 'k', alpha=0.2)
    plt.plot([0, 0 + (1-x)/2], [x, x + (1-x)/2], 'k', alpha=0.2)
    plt.plot([x, x + (1-x)/2], [0, 0 + (1-x)/2], 'k', alpha=0.2)

plt.title("Illustration of sigmoid calibrator")
plt.xlabel("Probability class 1")
plt.ylabel("Probability class 2")
plt.xlim(-0.05, 1.05)
plt.ylim(-0.05, 1.05)

plt.show()


from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pickle
# Load and split data
data = load_iris()
Xtrain, Xtest, Ytrain, Ytest = train_test_split(data.data, data.target, test_size=0.3, random_state=4)
# Create a model
model = LogisticRegression(C=0.1,
                           max_iter=20,
                           fit_intercept=True,
                           n_jobs=3,
                           solver='liblinear')
model.fit(Xtrain, Ytrain)

score = model.score(Xtest, Ytest)
print("Test score: {0:.2f} %".format(100 * score))

tuple_objects = {"model":model, "score":score}
pickle.dump(tuple_objects, open("tuple_model.pkl", 'wb'))






pickled_file = pickle.load(open("tuple_model.pkl", 'rb'))
pickled_model=pickled_file["model"]
pickled_score=pickled_file["score"]
print(pickled_score)
score = pickled_model.score(Xtest, Ytest)
print(score)




from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import datasets
import pandas as pd
import numpy as np
data=pd.DataFrame()
data["C1"]=np.arange(0,5)
data["C2"]=np.arange(20,25)
data["C3"]=np.array([0,1,0,2,1])
data["C4"]=np.array([3,1,0,2,3])
data["C5"]=np.array(["C1","C3","C3","C2","C2"])
data["C6"]=np.array(["P2","P1","P1","P2","P3"])
data["Target"]=np.array(["T2","T3","T3","T2","T1"])
col1=["C1","C2"]
col2=["C3","C4"]
col3=["C5","C6"]
targetVariable="Target"
print(data)
# temp_X_dt =pd.get_dummies(data[["C5","C6"]])
X_dt = pd.concat((data[col1+col2], pd.get_dummies(data[col3])), 1)
print(X_dt)
y_dt=data[targetVariable]

X_train, X_test, y_train, y_test = train_test_split(X_dt, y_dt, test_size=0.2, random_state=500)

ss=StandardScaler()
oe=OrdinalEncoder()

ss.fit(X_train[col1])
X_train[col1]=ss.transform(X_train[col1])
X_test[col1]=ss.transform(X_test[col1])

oe.fit(X_train[col2])
X_train[col2]=oe.transform(X_train[col2])
X_test[col2]=oe.transform(X_test[col2])

print(X_train)
print(X_test)

print(X_train[col2].dtypes)



import sys
from PyQt5.QtWidgets import QWidget, QProgressBar, QPushButton, QApplication
from PyQt5.QtCore import QBasicTimer

class ProgressBarDemo(QWidget):
	def __init__(self):
		super().__init__()

		self.progressBar = QProgressBar(self)
		self.progressBar.setGeometry(30, 40, 200, 25)

		self.btnStart = QPushButton('Start', self)
		self.btnStart.move(30, 80)
		self.btnStart.clicked.connect(self.startProgress)

		self.btnReset = QPushButton('Reset', self)
		self.btnReset.move(120, 80)
		self.btnReset.clicked.connect(self.resetBar)

		self.timer = QBasicTimer()
		self.step = 0

	def resetBar(self):
		self.step = 0
		self.progressBar.setValue(0)

	def startProgress(self):
		if self.timer.isActive():
			self.timer.stop()
			self.btnStart.setText('Start')
		else:
			self.timer.start(100, self)
			self.btnStart.setText('Stop')

	def timerEvent(self, event):
		if self.step >= 100:
			self.timer.stop()
			self.btnStart.setText('Start')
			return

		self.step +=1
		self.progressBar.setValue(self.step)

if __name__=='__main__':
	app = QApplication(sys.argv)

	demo = ProgressBarDemo()
	demo.show()

	sys.exit(app.exec_())












import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets
import pandas as pd
import numpy as np


df=pd.read_csv("HR-Employee-Attrition.csv")
print(df.columns)
colList=df.columns.tolist()
colList.remove("Department")
colList.remove("JobRole")

X_data=df[colList]
y_data=df["Department"]
le=LabelEncoder()
y_data=le.fit_transform(y_data)

print(y_data)

continuous_features=["Age","DistanceFromHome","DailyRate","HourlyRate","MonthlyIncome","MonthlyRate","NumCompaniesWorked","PercentSalaryHike","TotalWorkingYears","TrainingTimesLastYear","YearsAtCompany","YearsInCurrentRole","YearsSinceLastPromotion","YearsWithCurrManager"]
categorical_features=list(set(colList) - set(continuous_features))

X_data=pd.concat((X_data[continuous_features],pd.get_dummies(X_data[categorical_features])),1)

X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=500)

X_test.to_csv("file1.csv", index=False)
X_train.to_csv("file2.csv", index=False)

model=RandomForestClassifier()
model.fit(X_train,y_train)

X_test.drop(columns=["Age","DistanceFromHome","DailyRate","HourlyRate","MonthlyIncome"],inplace=True)

y_predict=model.predict(X_test)
print(y_predict)


y_predictprob=model.predict_proba(X_test)
print(y_predictprob)
y_predict=le.inverse_transform(y_predict)
# print(le.inverse_transform(y_predict))

predictedData=pd.DataFrame(data=y_predict,columns=["PredictedClass"])


predictedDataNew=pd.DataFrame(data=y_predict,columns=["PredictedClass"])
predictedDataNew["Class1"]=""
predictedDataNew["Class2"]=""
predictedDataNew["Class3"]=""
predictedDataNew[["Class1","Class2","Class3"]]=model.predict_proba(X_test)


le.classes_






