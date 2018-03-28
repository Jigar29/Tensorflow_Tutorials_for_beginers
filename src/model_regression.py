import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

class DataPreparation:
    def __init__(self, filepath):
        self.filepath = filepath
        return

    #This fucntion returns the train input and train labels from the stored training dataset as a numpy array
    #input : label_str ---> This fucntion takes label string as an input
    #########In case of null string it will print an error
    #Tobe fixed for validation on filepath name and label_str string
    def readFromCSV(self, label_str):
        dataframe = pd.read_csv(self.filepath)
        self.inputs = dataframe.drop(label_str, axis= 1).values
        self.labels = dataframe[[label_str]].values
        return self.inputs, self.labels

train_data = DataPreparation("..\Data\sales_data_training.csv")
test_data  = DataPreparation("..\Data\sales_data_test.csv")

X_train, y_train = train_data.readFromCSV("unit_price")
X_test, y_test   = test_data.readFromCSV("unit_price")

#the data is not scaled for all the features. Uses different scales for all the features will give you wrong result
#Let's go ahead and create the scalers
features_scaler = MinMaxScaler(feature_range=(0,1))
labels_scaler = MinMaxScaler(feature_range=(0,1))

#We have created separate scalers for features and target variables because the dimension of the matrix does not match
X_train = features_scaler.fit_transform(X_train)
y_train = labels_scaler.fit_transform(y_train)

X_test = features_scaler.transform(X_test)
y_test = labels_scaler.transform(y_test)

print(y_test)