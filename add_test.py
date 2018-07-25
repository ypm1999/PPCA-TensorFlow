""" import your model here """
import tensorflow as tf

import numpy as np

sess = tf.Session()

# adder
a = tf.placeholder(tf.int64)
b = tf.placeholder(tf.int64)
adder_node = a + b;

ans = sess.run(adder_node, {a: 3, b: 4})
print(ans)
assert np.equal(ans, 7)

ans = sess.run(adder_node, {a: [1, 3], b: [2, 3]})
print(ans)
assert np.array_equal(ans, [3, 6])
ans = sess.run(adder_node, {a: [[1, 3],[1, 3]], b: [[2, 3]]})
print(ans)