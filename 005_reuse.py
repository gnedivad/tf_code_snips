import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from scipy.ndimage import imread
from scipy.ndimage.filters import gaussian_filter

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
  "image_filename", "data/lenna.png", """Filename for image.""")
tf.app.flags.DEFINE_boolean(
  "show", True, """Whether to show or not.""")

###############################################################################
# If variables are shared and multiple inputs lead to different outputs,      #
# which one is used?                                                          #
###############################################################################


def add_two_numbers(input_op):
  delta_op = tf.get_variable(initializer=10, name="delta", dtype=tf.int32)
  G = tf.get_default_graph()
  input_plus_delta_op = tf.add(input_op, G.get_tensor_by_name("noob/delta:0"), name="sum")
  input_plus_2delta_op = tf.add(input_plus_delta_op, G.get_tensor_by_name("noob/delta:0"), name="sum2")

def main(argv=None):
  one_op = tf.constant(1, name="one")
  two_op = tf.constant(2, name="two")

  with tf.variable_scope("noob") as scope:
    add_two_numbers(one_op)
    scope.reuse_variables()
    add_two_numbers(two_op)

  G = tf.get_default_graph()
  with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    print([n.name for n in tf.get_default_graph().as_graph_def().node])
    print(sess.run(G.get_tensor_by_name("noob/sum2:0")))


if __name__ == "__main__":
  tf.app.run()
