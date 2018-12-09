import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

num_features = 10;
num_dense_neurons = 3;
num_epochs = 200;
num_examples = 100000;
batch_size = 10;

#Dataset creation
x_data = np.linspace(0, 10, num_examples)
noise  = np.random.randn(len(x_data));
y_data = (0.5*x_data) + 5 + noise;

#Actual ML creation
x_df = pd.DataFrame(data=x_data, columns=['X_data']);
y_df = pd.DataFrame(data=y_data, columns=['Y_data']);
my_data = pd.concat([x_df, y_df], axis=1);

x_train = tf.placeholder(tf.float32, shape=[batch_size])
y_train = tf.placeholder(tf.float32, shape=[batch_size])

weights = tf.Variable(dtype= tf.float32, initial_value=0.81);
bias = tf.Variable(dtype= tf.float32, initial_value= 1);

y_pred = tf.add(tf.multiply(x_train, weights), bias);

loss = tf.reduce_sum(tf.square(y_train - y_pred));
optimiser = tf.train.GradientDescentOptimizer(learning_rate=0.001);

train = optimiser.minimize(loss);

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init);

    for i in range(num_examples):
        rand = np.random.randint(0, len(x_data), batch_size);
        sess.run(train, feed_dict= {x_train: x_data[rand], y_train: y_data[rand]})

    slope, intercept = sess.run((weights, bias))
    print(slope, intercept)

y_test = slope * x_data + intercept;

my_data.sample(250).plot(kind= "scatter", x = 'X_data', y= 'Y_data')
plt.plot(x_data, y_test, "r")
plt.show()