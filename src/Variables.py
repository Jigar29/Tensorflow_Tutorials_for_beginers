import tensorflow as tf

#this is the implementation of the tf variables with 1-D of type float(32 bit)
x = tf.Variable(initial_value= 0, trainable= True, dtype= tf.float32)
y = tf.Variable(initial_value= 1, trainable= True, dtype= tf.float32)
#This is the implementation of the tf variables with N-D of type float(32 bit)
a = tf.Variable(tf.zeros(dtype=tf.float32, shape=(2,2)), dtype= tf.float32, trainable= True)
b = tf.Variable(tf.ones(dtype=tf.float32, shape=(2,2)), dtype= tf.float32, trainable= True)

# This is the operation to be performed on the declared tf variables
out = tf.add(x,y, name="1-D_Sum")
out_mat = tf.add(a, b, name= "MAT_Sum")

#This is necessary to initialize the variables defined above
init = tf.global_variables_initializer()

#Let's Run the operations defined above
with tf.Session() as sess:
    tensorboard = tf.summary.FileWriter("..\Log", sess.graph)
    sess.run(init)
    print(sess.run(out_mat))
    print(sess.run(out))
    tensorboard.close()