from __future__ import print_function
from sklearn.model_selection import train_test_split
from sklearn import linear_model
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpat
import csv

# Function to load CSV file
def loadCSV(filename):
    file = open(filename, 'r', encoding="utf8")     # Open file
    csv_file = csv.reader(file)                     # Create CSV reader
    data = list()                                   # Create empty Data list
    target = list()                                 # Create empty target list
    
    for row in csv_file:                            # Rows contain target split into two
        if (len(row) == 0):                         # Skip empty rows
            continue
        data.append(row[:len(row) - 1])
        if (row[len(row) - 1] == 'cat'):
            target.append(1)
        else:
            target.append(-1)
        #target.append(row[len(row) - 1])
    
    return np.asarray(data, dtype=np.float64), np.asarray(target, dtype=np.int64)    # Return tuple containing data array and target array. Set data type to float64 or else it'll crash when predicting


# Load data from CSV file. Edit this to point to the features file
data, target = loadCSV("D:\\School\\CS 599 - Machine Learning\\ArrayCreator\\features.csv")

# Split the data into two parts: training data and testing data
train_data,test_data,train_target,test_target = train_test_split(data,(target[:, np.newaxis]), test_size=0.2, random_state=42)
train_target = np.ravel(train_target)


log_reg = linear_model.LogisticRegression()  # Get LogisticalRegression model
log_reg.fit(train_data, train_target)      # Fit the model to the train data and target (label)
test_predict = log_reg.predict(test_data)
print("Training Accuracy:",log_reg.score(train_data,train_target), "\nTest Accuracy:", log_reg.score(test_data, test_target)) # Compute accuracy

# Show a quick plot
fig = plt.figure()
axis = fig.add_subplot(111)
colors = ['red' if target == 1 else 'blue' for target in test_target]
axis.scatter(range(0, len(test_data)), test_predict, color=colors)   #Print predicted points
legend_red = mpat.Patch(color='red', label='Cat')
legend_blue = mpat.Patch(color='blue', label='Dog')
#axis.scatter(range(0, len(test_data)), test_predict, color='blue', label='Predicted')   #Print predicted points
#axis.scatter(range(0, len(test_data)), test_target, color='red', label='Target')        #Print target points
#for i in range(0, len(test_target)):\
#    axis.plot([i,i],[test_target[i],test_predict[i]], 'k-')   #Print a line between the predicted and target
axis.legend(handles=[legend_red, legend_blue], loc='lower center')
plt.show()