import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler as scaler
from sklearn.model_selection import train_test_split

##using estimator API steps
# 1. generate feture colums
# 2. generate the estimator or model
# 3. generate the input function
# 4. train the model
# 5. evaluate or predict the model

my_data = pd.read_csv("../Data/pima-indians-diabetes.csv");

scaling_cols = ['Number_pregnant', 'Glucose_concentration', 'Blood_pressure', 'Triceps',
       'Insulin', 'BMI', 'Pedigree']

my_data[scaling_cols] = my_data[scaling_cols].apply(lambda x: ((x - x.min())/ (x.max() - x.min())))

num_preg = tf.feature_column.numeric_column('Number_pregnant')
glucose_cons = tf.feature_column.numeric_column('Glucose_concentration')
blood_pr = tf.feature_column.numeric_column('Blood_pressure')
triceps = tf.feature_column.numeric_column('Triceps')
insulin = tf.feature_column.numeric_column('Insulin')
bmi = tf.feature_column.numeric_column('BMI')
pedigree = tf.feature_column.numeric_column('Pedigree')
age = tf.feature_column.numeric_column('Age')

##catogorical data, label encoding
group = tf.feature_column.categorical_column_with_hash_bucket('Group', hash_bucket_size=4);
age_bucket = tf.feature_column.bucketized_column(age, [10, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90])

feature_col = [num_preg, glucose_cons, blood_pr, triceps, insulin, bmi, pedigree, age_bucket, group]

x_train, x_test, y_train, y_test = train_test_split(my_data.drop('Class', axis=1), my_data['Class'], train_size=0.8, random_state=101)

model = tf.estimator.LinearClassifier(feature_columns= feature_col, n_classes=2)

train_func = tf.estimator.inputs.pandas_input_fn(x=x_train, y=y_train, batch_size=10, num_epochs= 1, shuffle= True);
eval_func = tf.estimator.inputs.pandas_input_fn(x=x_test, y=y_test, batch_size=10, num_epochs= 1, shuffle= False);

training = model.train(input_fn= train_func)

eval = model.evaluate(input_fn=eval_func)

print(eval)
