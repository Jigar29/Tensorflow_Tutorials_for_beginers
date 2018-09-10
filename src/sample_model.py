import tensorflow as tf
import numpy as np

##This neural network has one hidden layer and there are 3 neurons in a single hidden layer
##No_of_features are 4
no_of_features = 4
no_of_dense_neurons = 3

x = tf.placeholder(dtype= tf.float32, shape= [None, no_of_features], name= "input_placeholder")
W = tf.Variable(initial_value=tf.random_normal(shape= [no_of_features, no_of_dense_neurons], dtype= tf.float32),name= "Weights_tensor")
b = tf.Variable(initial_value= tf.ones(shape= [no_of_dense_neurons]), dtype= tf.float32, name= "Bias_tensor")

multiplication = tf.matmul(x, W, name="Multiplication_operation")
y = tf.add(multiplication, b, name= "addition_operation")
activation_output = tf.sigmoid(y, name="Activation_operation")
init = tf.global_variables_initializer()

with tf.Session() as sess:
    tensorboard = tf.summary.FileWriter(logdir="../Log/", graph=sess.graph)
    sess.run(init)
    result_tensor = sess.run(activation_output, feed_dict={x:np.random.random([30, no_of_features])})
    print(result_tensor)
    tensorboard.close()


