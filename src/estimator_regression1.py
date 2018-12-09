import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

my_data = pd.read_csv("../Data/cal_housing_clean.csv")
print(my_data.columns)

scaler = MinMaxScaler();

df_cols = ['housingMedianAge', 'totalRooms', 'totalBedrooms', 'population', 'households', 'medianIncome'];
my_data.dropna(axis=0);

x_train, x_test, y_train, y_test = train_test_split(my_data[df_cols], my_data['medianHouseValue'], train_size= 0.7, random_state= 101)

scaler.fit(x_train)

x_train = pd.DataFrame(data= scaler.transform(x_train), columns=x_train.columns, index= x_train.index)
x_test  = pd.DataFrame(data= scaler.transform(x_test), columns=x_test.columns, index= x_test.index)

median_age = tf.feature_column.numeric_column('housingMedianAge');
total_rooms = tf.feature_column.numeric_column('totalRooms');
total_bedrooms = tf.feature_column.numeric_column('population');
house_holds = tf.feature_column.numeric_column('households');
median_income = tf.feature_column.numeric_column('medianIncome');

feature_cols = [median_age, total_rooms, total_bedrooms, house_holds, median_income];

# Building a model
model = tf.estimator.DNNRegressor(hidden_units=[6,6,6], feature_columns= feature_cols);

train_input_func = tf.estimator.inputs.pandas_input_fn(x= x_train, y=y_train, batch_size= 10, num_epochs= 1000, shuffle= True)
eval_input_func  = tf.estimator.inputs.pandas_input_fn(x= x_test, y=y_test, batch_size= 10, num_epochs= 1000, shuffle= False)
pred_input_func  = tf.estimator.inputs.pandas_input_fn(x = x_test, batch_size= 10, num_epochs= 1, shuffle= False);

training = model.train(input_fn= train_input_func, steps= 20000)
#eval = model.evaluate(input_fn=eval_input_func, steps= 1000)

predictions = []
for pre in model.predict(input_fn=pred_input_func):
    predictions.append(pre['predictions'])

error = mean_squared_error(y_test, predictions)**0.5    #Root mean square error 

print(error)
print(training)
print(eval)