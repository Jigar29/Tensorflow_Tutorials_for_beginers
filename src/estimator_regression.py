import tensorflow as tf
from tensorflow._api.v1 import feature_column
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

##using estimator API steps
# 1. generate feture colums
# 2. generate the estimator or model
# 3. generate the input function
# 4. train the model
# 5. evaluate or predict the model

num_examples = 1000;

#Dataset creation
x_data = np.linspace(0, 10, num_examples)
noise  = np.random.randn(len(x_data));
y_data = (0.5*x_data) + 5 + noise;

x_df = pd.DataFrame(data=x_data, columns=['X_data']);
y_df = pd.DataFrame(data=y_data, columns=['Y_data']);
my_data = pd.concat([x_df, y_df], axis=1);

x_train , x_test, y_train, y_test = train_test_split(x_data, y_data, train_size=0.7, random_state=42);

#model creation
features_col = [tf.feature_column.numeric_column("X_data", shape=[1])]

estimator = tf.estimator.LinearRegressor(feature_columns=features_col);

input_train_func = tf.estimator.inputs.numpy_input_fn({'X_data':x_train}, y_train, batch_size= 10, num_epochs= 200, shuffle=True)
input_test_func = tf.estimator.inputs.numpy_input_fn({'X_data':x_test}, y_test, batch_size= 10, num_epochs= 200, shuffle=True)
input_pred_func = tf.estimator.inputs.numpy_input_fn({'X_data':x_test},shuffle=False)

train_matrix = estimator.train(input_fn= input_train_func)

prediction = [];
for pred in estimator.predict(input_fn= input_pred_func):
    prediction.append(pred['predictions']);

my_data.sample(250).plot(kind='scatter', x = 'X_data', y = 'Y_data')
plt.plot(x_test, prediction, 'r')
plt.show()