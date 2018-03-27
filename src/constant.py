import tensorflow as tf

# Tutorial for constants. Constants are just the scalar tensor. Scalar tensor means an array which can be
# multidimensional as tensor and it is scalar means its value can not be changed ever during the run-time of the
# program

# Here, the shape of the tensor is chosen to be None. That means it is a one dimentional tensor.
# General form is [no_of_rows, no_of_cols], name is just a name assigned to the tensor, leaving the gap in string of the name
# will not work. name is used for tensorboard naming of the module
x = tf.constant(value= 3, dtype= tf.float32, shape=None, name = 'First_1D_Tensor')
y = tf.constant(value= 4, dtype=tf.float32, shape= None, name = 'Second_1D_Tensor')

#Matrix tensor examples
a = tf.constant(value= [[2], [3]], dtype=tf.float32, shape = [2, 1])
b = tf.constant(value= [[4], [2]], dtype=tf.float32, shape = [2, 1])

#Multiplicatin operation on the tensors
out = tf.multiply(x, y)
out_scalar = tf.scalar_mul(2, a)
out_mat = tf.matmul(a,b,transpose_b=True)

#Using with..as deletes the context automaticalyy for Session
with tf.Session() as sess:
    tensorboard = tf.summary.FileWriter("..\Log", sess.graph)
    print('1D Tensor---> ' + str(sess.run(out)))
    print('ND Tensor---> ' + str(sess.run(out_mat)))
    print('Scalar Tensor---> ' + str(sess.run(out_scalar)))
    tensorboard.close()