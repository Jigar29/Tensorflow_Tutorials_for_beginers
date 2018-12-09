import tensorflow as tf
from tensorflow._api.v1 import feature_column
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

##using estimator API steps
# 1. generate feture colums
# 2. generate the estimator or model
# 3. generate the input function
# 4. train the model
# 5. evaluate or predict the model

my_data = pd.read_csv("../Data/census_data.csv")
my_data = my_data.dropna(axis=0)

encoder = LabelEncoder()

my_data['income_bracket'] = encoder.fit_transform(my_data['income_bracket']);
age = tf.feature_column.numeric_column("age")
workclass = tf.feature_column.categorical_column_with_hash_bucket("workclass", hash_bucket_size= 100);
education = tf.feature_column.categorical_column_with_hash_bucket("education", hash_bucket_size= 100);
education_num = tf.feature_column.numeric_column("education_num");
marital_status = tf.feature_column.categorical_column_with_hash_bucket("marital_status", hash_bucket_size= 100);
occupation = tf.feature_column.categorical_column_with_hash_bucket("occupation", hash_bucket_size= 100);
relationship = tf.feature_column.categorical_column_with_hash_bucket("relationship", hash_bucket_size= 100);
race = tf.feature_column.categorical_column_with_hash_bucket("race", hash_bucket_size= 100);
gender = tf.feature_column.categorical_column_with_hash_bucket("gender", hash_bucket_size= 100);
capital_gain = tf.feature_column.numeric_column("capital_gain")
capital_loss = tf.feature_column.numeric_column("capital_loss")
hours_per_week = tf.feature_column.numeric_column("hours_per_week")
native_country = tf.feature_column.categorical_column_with_hash_bucket("native_country", hash_bucket_size= 100);

feature_column = [age, workclass, education, education_num, marital_status, occupation, relationship, race, gender, capital_gain, capital_loss, hours_per_week, native_country]

x_train, x_test, y_train, y_test = train_test_split(my_data.drop('income_bracket', axis= 1), my_data['income_bracket'], train_size= 0.7, random_state= 101)

model = tf.estimator.LinearClassifier(feature_columns=feature_column, n_classes=2)

train_input_fun = tf.estimator.inputs.pandas_input_fn(x = x_train, y = y_train, batch_size= 10, num_epochs= 1000, shuffle= True)
eval_input_fun = tf.estimator.inputs.pandas_input_fn(x = x_test, y = y_test, batch_size= 10, num_epochs= 1000, shuffle= False)
pred_input_fun = tf.estimator.inputs.pandas_input_fn(x = x_test, batch_size= 10, num_epochs= 1, shuffle= False)

training = model.train(input_fn=train_input_fun, steps= 10000)
predict = model.predict(input_fn= pred_input_fun)

predictions = []
for pred in predict:
    predictions.append(pred['class_ids'][0])

print(classification_report(y_test, predictions))