import tensorflow as tf

#gives you a default graph
print(tf.get_default_graph())

#Creates  a new graph
g1 = tf.Graph()
with g1.as_default():
    a = tf.constant(8)
    b = tf.constant(2)
    op = tf.add(a, b, name='Graph2_operation')

g2 = tf.Graph()
with g2.as_default():
    a = tf.constant(2)
    b = tf.constant(2)
    op2 = tf.add(a,b, name='Graph2_operation')

with tf.Session(graph=g1) as sess1:
    print(sess1.run(op))

with tf.Session(graph=g2) as sess2:
    print(sess2.run(op2))