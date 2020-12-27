# MACHINE LEARNING SIMPLIFIED

This tool provides end - to - end deployment of a Machine Learning model, from Data Mining, Pre Processing, Exploratory Data Analysis to Feature Selection and Machine Learning model building and its application. Currently, this tool supports only classification.

### Overall application can be segmented into three major parts:

1. **Data Loading/Features Setup**: This is done to load the data going to be used and setup the target and input features.
 
2. **EDA Analysis**: This is essentially used to evaluate features w.r.t entire dataset and the target classes.
    1. **Variable Distribution**: Showcases the distribution of continuous features along with option to choose data filter
    2. **Variable Relation**: Showcases the relationship between two features along with option to choose data filter. based on the type of features, the graph changes i.e., in case both the features are continuous, then a scatter plot is poulated, while in case one feature is categorical and the other is continuous, then a box plot is populated.
    3. **Target Class Relation**: Showcases the comparison of a feature between Target Class subsets. For continuous features, it compares the Minimum, Median, Mean and Maximum Values. For categorical features, it compares the count of values for each distinct element of the feature.

3. **Train ML Models**: This is used to see how the features along with the type of model perform to predict the Target Class. User can:
    * Input the value for Test Percentage (set by default to 20)
    * Select the features from the list (by default all are selected)
    * View the performance of the model in terms of Accuracy, Precision, Recall and F1 Score
    * Compare the performance of the current model with the other available models with same configuration (features and test percentage)
    * Save the final model
    
    Following are the models used for classifying the Atrition of an employee as Yes or No:
    1. **Decision Trees**: Showcases the results in the form of Confusion Matrix, Decision Tree Graph, Importance of Features and ROC Curve by Class.
    2. **Random Forest**: Showcases the results in the form of Confusion Matrix, AUC Score vs Number of Trees, Importance of Features and ROC Curve by Class. User can input the number of trees (set by default to 35).
    3. **Logistic Regression**: Showcases the results in the form of Confusion Matrix, Calibration Curve, Cross Validation Score, and ROC Curve by Class.
    4. **K Nearest Neighbours**: Showcases the results in the form of Confusion Matrix, Accuracy vs K Value, Cross Validation Score, and ROC Curve by Class. User can input the number of neighbours (set by default to 9).

3. **Test ML Models**: This enables the user to load the pre trained models, view their cofiguration and then use them to make predictions on new/test data set.

## Run the app
To start the tool, run the "code.py".  
(Mac Users: When the Application starts, unfocus the app and then focus the app again to activate the menu items).  
Note: In case if you wish to skip the installation of the required packages (such as PyQt5, sklearn etc), comment the lines 2-9.

