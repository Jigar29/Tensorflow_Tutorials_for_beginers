import tensorflow as tf
import numpy as np
import matplotlib.pyplot as mplt

#Creating a linear numspace for activation function range
numspace = np.linspace(-5, 5, 100)

#some popular activation function implementation using tensorflow
out_relu = tf.nn.relu(numspace, name="RELU")
out_sigmoid = tf.nn.sigmoid(numspace, "Sigmoid")
out_softplus = tf.nn.softplus(numspace, "Softplus")
out_tanh = tf.nn.tanh(numspace, "Signum")

#Let's run the implemented activation functions
with tf.Session() as sess:
    tensorboard = tf.summary.FileWriter("..\Log", sess.graph)
    out_relu = sess.run(out_relu)
    out_sigmoid = sess.run(out_sigmoid)
    out_softplus = sess.run(out_softplus)
    out_tanh = sess.run(out_tanh)
    tensorboard.close()

mplt.subplot(221)
mplt.plot(numspace, out_relu, label = 'Relu Function')
mplt.legend(loc = "best")

mplt.subplot(222)
mplt.plot(numspace, out_tanh, label = 'Signum Function')
mplt.legend(loc = "best")

mplt.subplot(223)
mplt.plot(numspace, out_softplus, label = 'Softplus Function')
mplt.legend(loc = "best")

mplt.subplot(224)
mplt.plot(numspace, out_sigmoid)
mplt.legend(loc = "best")
mplt.show()