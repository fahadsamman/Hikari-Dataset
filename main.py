# import required libraries:
import pandas as pd
from sklearn import metrics
from sklearn import preprocessing
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# set default display to show all rows and columns:
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 1000)


# import dataset:
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
data = pd.read_csv("/content/drive/MyDrive/CPCS331 Project/PROJECT 2 - Machine Learning/HIKARI2021.csv")
print(data.columns)




###################################################################################################
# ------------------- PREPROCESSING (CLEANING) -----------------
###################################################################################################

# 1. check for null values:
#print(data.isnull().sum())

# 2 .drop useless columns:
data = data.drop('Unnamed: 0.1', 1)
data = data.drop('uid', 1)
data = data.drop('Unnamed: 0', 1)
data = data.drop('originh', 1)
data = data.drop('originp', 1)
data = data.drop('responh', 1)
data = data.drop('responp', 1)

# 3. combine benign data with background data:
data['traffic_category'] = data['traffic_category'].replace(['Benign'],'Background')

# 4. remove duplicates:
sam = data.query("traffic_category == 'Background'").sample(n=510000)
data = data.drop_duplicates()
#print("Data after removing duplicates: \n" ,data['traffic_category'].value_counts())

# 5. sample from background (to balance data):
data = pd.concat([data, sam, sam]).drop_duplicates(keep=False)
#print("Data after sampling background: \n" ,data['traffic_category'].value_counts())

# 6. sample from probing (to balance data):
probing = data.query("traffic_category == 'Probing'").sample(n=17000)
data = pd.concat([data, probing, probing]).drop_duplicates(keep=False)
#print("******** Data after sampling probing: \n" ,data['traffic_category'].value_counts())

# 7. normalize:
data[['flow_duration', 'fwd_pkts_tot', 'bwd_pkts_tot', 'fwd_data_pkts_tot', 'bwd_data_pkts_tot', 'fwd_pkts_per_sec', 'bwd_pkts_per_sec', 'flow_pkts_per_sec', 'down_up_ratio', 'fwd_header_size_tot', 'fwd_header_size_min', 'fwd_header_size_max', 'bwd_header_size_tot', 'bwd_header_size_min', 'bwd_header_size_max', 'flow_FIN_flag_count', 'flow_SYN_flag_count', 'flow_RST_flag_count', 'fwd_PSH_flag_count', 'bwd_PSH_flag_count', 'flow_ACK_flag_count', 'fwd_URG_flag_count', 'bwd_URG_flag_count', 'flow_CWR_flag_count', 'flow_ECE_flag_count', 'fwd_pkts_payload.min', 'fwd_pkts_payload.max', 'fwd_pkts_payload.tot', 'fwd_pkts_payload.avg', 'fwd_pkts_payload.std', 'bwd_pkts_payload.min', 'bwd_pkts_payload.max', 'bwd_pkts_payload.tot', 'bwd_pkts_payload.avg', 'bwd_pkts_payload.std', 'flow_pkts_payload.min', 'flow_pkts_payload.max', 'flow_pkts_payload.tot', 'flow_pkts_payload.avg', 'flow_pkts_payload.std', 'fwd_iat.min', 'fwd_iat.max', 'fwd_iat.tot', 'fwd_iat.avg', 'fwd_iat.std', 'bwd_iat.min', 'bwd_iat.max', 'bwd_iat.tot', 'bwd_iat.avg', 'bwd_iat.std', 'flow_iat.min', 'flow_iat.max', 'flow_iat.tot', 'flow_iat.avg', 'flow_iat.std', 'payload_bytes_per_second', 'fwd_subflow_pkts', 'bwd_subflow_pkts', 'fwd_subflow_bytes', 'bwd_subflow_bytes', 'fwd_bulk_bytes', 'bwd_bulk_bytes', 'fwd_bulk_packets', 'bwd_bulk_packets', 'fwd_bulk_rate', 'bwd_bulk_rate', 'active.min', 'active.max', 'active.tot', 'active.avg', 'active.std', 'idle.min', 'idle.max', 'idle.tot', 'idle.avg', 'idle.std', 'fwd_init_window_size', 'bwd_init_window_size', 'fwd_last_window_size']] = preprocessing.normalize(data[['flow_duration', 'fwd_pkts_tot', 'bwd_pkts_tot', 'fwd_data_pkts_tot', 'bwd_data_pkts_tot', 'fwd_pkts_per_sec', 'bwd_pkts_per_sec', 'flow_pkts_per_sec', 'down_up_ratio', 'fwd_header_size_tot', 'fwd_header_size_min', 'fwd_header_size_max', 'bwd_header_size_tot', 'bwd_header_size_min', 'bwd_header_size_max', 'flow_FIN_flag_count', 'flow_SYN_flag_count', 'flow_RST_flag_count', 'fwd_PSH_flag_count', 'bwd_PSH_flag_count', 'flow_ACK_flag_count', 'fwd_URG_flag_count', 'bwd_URG_flag_count', 'flow_CWR_flag_count', 'flow_ECE_flag_count', 'fwd_pkts_payload.min', 'fwd_pkts_payload.max', 'fwd_pkts_payload.tot', 'fwd_pkts_payload.avg', 'fwd_pkts_payload.std', 'bwd_pkts_payload.min', 'bwd_pkts_payload.max', 'bwd_pkts_payload.tot', 'bwd_pkts_payload.avg', 'bwd_pkts_payload.std', 'flow_pkts_payload.min', 'flow_pkts_payload.max', 'flow_pkts_payload.tot', 'flow_pkts_payload.avg', 'flow_pkts_payload.std', 'fwd_iat.min', 'fwd_iat.max', 'fwd_iat.tot', 'fwd_iat.avg', 'fwd_iat.std', 'bwd_iat.min', 'bwd_iat.max', 'bwd_iat.tot', 'bwd_iat.avg', 'bwd_iat.std', 'flow_iat.min', 'flow_iat.max', 'flow_iat.tot', 'flow_iat.avg', 'flow_iat.std', 'payload_bytes_per_second', 'fwd_subflow_pkts', 'bwd_subflow_pkts', 'fwd_subflow_bytes', 'bwd_subflow_bytes', 'fwd_bulk_bytes', 'bwd_bulk_bytes', 'fwd_bulk_packets', 'bwd_bulk_packets', 'fwd_bulk_rate', 'bwd_bulk_rate', 'active.min', 'active.max', 'active.tot', 'active.avg', 'active.std', 'idle.min', 'idle.max', 'idle.tot', 'idle.avg', 'idle.std', 'fwd_init_window_size', 'bwd_init_window_size', 'fwd_last_window_size']])
#print("** Data after normalize: \n" ,data['traffic_category'].value_counts())
#print(data.sample(n=1)) # check a sample

# 8. adjust labels:

data['Label'] = data['traffic_category']

data['Label'] = data['Label'].replace(['Background'],0)
data['Label'] = data['Label'].replace(['Bruteforce'],1)
data['Label'] = data['Label'].replace(['Bruteforce-XML'],2)
data['Label'] = data['Label'].replace(['Probing'],3)
data['Label'] = data['Label'].replace(['XMRIGCC CryptoMiner'],4)

data['Label'] = data['Label'].astype(str).astype(float)

###################################################################################################


###################################################################################################
## ----------------------------- DECISION TREE ---------------------------------
###################################################################################################
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import tree
from sklearn import metrics
import seaborn as sn
import matplotlib.pyplot as plt


# Select features:
x = data[['fwd_iat.min','fwd_iat.avg','flow_iat.min','bwd_bulk_rate','flow_iat.avg']]
# Select Target:
y = data['traffic_category']

# Split Data into training set and testing set:
xTrain, xTest, yTrain, yTest = train_test_split(x,y,test_size=0.20,random_state=0)

# Create Decision Tree object
treeModel = tree.DecisionTreeClassifier()

# Train Decision Tree Classifer
treeModel = treeModel.fit(xTrain,yTrain)

# Predict the response for test dataset
yPred = treeModel.predict(xTest)

# Confusion Matrix:
confusion_matrix = pd.crosstab(yTest, yPred, rownames=['Actual'], colnames=['Predicted'])
sn.heatmap(confusion_matrix, annot=True)
plt.show()

from sklearn.metrics import classification_report
print(classification_report(yTest, yPred))
print("Accuracy:",metrics.accuracy_score(yTest, yPred))



###################################################################################################
## ----------------------------- KNN ---------------------------------
###################################################################################################

# select features:
x = data[['fwd_iat.min','fwd_iat.avg','flow_iat.min','bwd_bulk_rate','flow_iat.avg']]

# Select Target:
y = data['traffic_category']
#y = data['Label']

# Split Data into training set and testing set:
xTrain, xTest, yTrain, yTest = train_test_split(x,y,test_size=0.20,random_state=0)


# Create KNN object
from sklearn.neighbors import KNeighborsClassifier
knn_model = KNeighborsClassifier(n_neighbors=5)

# Train KNN Clustering
knn_model.fit(xTrain, yTrain)

# Test :
yPred = knn_model.predict(xTest)

# Confusion Matrix:
confusion_matrix = pd.crosstab(yTest, yPred, rownames=['Actual'], colnames=['Predicted'])
sn.heatmap(confusion_matrix, annot=True)
plt.show()

# Analysis:
from sklearn.metrics import classification_report
print(classification_report(yTest, yPred))
print("Accuracy:",metrics.accuracy_score(yTest, yPred))



###################################################################################################
## ----------------------------- KNN improved ---------------------------------
###################################################################################################


from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsClassifier

# select features:
X = data[['fwd_iat.min', 'fwd_iat.avg', 'flow_iat.min', 'bwd_bulk_rate', 'flow_iat.avg']]

# select class:
Y = data['traffic_category']

# create KNN model:
model = KNeighborsClassifier(n_neighbors=5)

# create k fold:
k = 10
kv = KFold(n_splits=k, shuffle=True, random_state=None)

acc_score = []  # to save all accuracy scores

for train_index, test_index in kv.split(X):
    # print("Train Index: ", train_index, "\n")
    # print("Test Index: ", test_index)
    # print("***********")

    # split into training and testing:
    X_train, X_test, Y_train, Y_test = X.iloc[train_index, :], X.iloc[test_index, :], Y.iloc[train_index], Y.iloc[
        test_index]

    # Train (Fit) model
    model.fit(X_train, Y_train)

    # Test model:
    Y_pred = model.predict(X_test)

    # save accuracy score:
    acc = accuracy_score(Y_pred, Y_test)
    acc_score.append(acc)

avg_acc_score = sum(acc_score) / k

# print('accuracy of each fold - {}'.format(acc_score))

# Confusion Matrix:
confusion_matrix = pd.crosstab(Y_test, Y_pred, rownames=['Actual'], colnames=['Predicted'])
sn.heatmap(confusion_matrix, annot=True)
plt.show()

# Analysis:
from sklearn.metrics import classification_report

print(classification_report(Y_test, Y_pred))
print('Avg accuracy : {}'.format(avg_acc_score))

