import numpy as np
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

###############################################################################
# Does a placeholder need to be filled even if it's not along critical path   #
# to an evaluated operator?                                                   #
###############################################################################


def model(d):
  with tf.variable_scope("a"):
    x_op = tf.placeholder(tf.float32, shape=(), name="x_ph")
    tf.placeholder(tf.float32, shape=(), name="y_ph")
    z_op = tf.add(x_op, d, name="z")


def main(argv=None):
  with tf.Graph().as_default() as G:
    model(1)
    with tf.Session() as sess:
      x_op = G.get_tensor_by_name("a/x_ph:0")
      z_op = G.get_tensor_by_name("a/z:0")
      x_value, z_value = sess.run([x_op, z_op], feed_dict={"a/x_ph:0": 10})
      print("%.2f + 1 = %.2f" % (x_value, z_value))


if __name__ == "__main__":
  tf.app.run()
