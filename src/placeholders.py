import tensorflow as tf

# So, the placeholder is just like the variables but the purpose of usage is different
# Vaiables are supposed to be used for trainiable quantities only i.e weights and biases whereas the placeholders are used for
# the quantities which can be fed at the runtime i.e. dataset

#Let's declare a few 1-D place holders
x = tf.placeholder(dtype= tf.float32, shape=None)
y = tf.placeholder(dtype= tf.float32, shape=None)

#Declaring N-D place holders
a = tf.placeholder(dtype=tf.float32, shape=(2,1))
b = tf.placeholder(dtype=tf.float32, shape= (2,1))

#Implementing some operations
out = tf.multiply(x,y,name='1-D_Mult')
out_mat = tf.matmul(a,b,transpose_b=True, name='N-D_Mult')

#let's run this operations
with tf.Session() as sess:
    tensorboard = tf.summary.FileWriter("..\Log", sess.graph)
    print(sess.run(out, feed_dict={x:4, y:5}))
    print(sess.run(out_mat, feed_dict={a:[[2], [4]], b:[[4], [5]]}))
    tensorboard.close()

