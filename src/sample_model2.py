import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

no_of_epochs = 200
no_of_examples = 100

np.random.seed(1)
tf.set_random_seed(1)

X = np.linspace(-10, 10, no_of_examples) + np.random.normal(0, 1, no_of_examples)
y = np.linspace(-1, 1, no_of_examples) + np.random.normal(0, 1, no_of_examples)

#Plotting the original dataset
plt.scatter(X, y)
plt.show()

## TensorFlow implementation

#Place Holders
x_train = tf.placeholder(tf.float32, shape= None, name = 'Input_tensor')
y_train = tf.placeholder(tf.float32, shape= None, name = 'Label_tensor')

#Variables Declaration
slope = tf.Variable(initial_value= 1, dtype= tf.float32)
bias  = tf.Variable(initial_value= 0.04, dtype= tf.float32)

#Operations
mult = tf.multiply(slope, x_train, name= "Multiplicarion_operation")
predict = tf.add(mult, bias)

loss = tf.losses.mean_squared_error(predict, y_train)
optimizer = tf.train.GradientDescentOptimizer(learning_rate= 0.01, name="Optimizer")

model_output = optimizer.minimize(loss)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for i in range(no_of_epochs):
        prediction, output , cost = sess.run([predict, model_output, loss], feed_dict={x_train: X, y_train: y})

        print(str(i) + 'Ephochs Passed')

        if(i % 5 == 0):
            plt.scatter(X, y)
            plt.plot(X, prediction)
            plt.text(0.8, 1, "Loss = %.4f" %cost, fontdict={'size': 27, 'color': 'red'})
            plt._show()
    print("The model has been trained successfully ")