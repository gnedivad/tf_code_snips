import numpy as np
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
  "checkpoint_dir", "checkpoints/initialize_or_restore", """Directory for checkpoints.""")

###############################################################################
# Does variable initialization overwrite variables read from checkpoints?     #
###############################################################################


def save_first_model(sess):
  saver = tf.train.Saver()  # defaults to list of all saveable objects
  saver.save(sess, "%s/model.ckpt" % FLAGS.checkpoint_dir)


def restore_first_model(sess, var_list):
  is_restored = False
  saver = tf.train.Saver(var_list)
  checkpoint_state = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
  if checkpoint_state and checkpoint_state.model_checkpoint_path:
    saver.restore(sess, checkpoint_state.model_checkpoint_path)
    is_restored = True
  return is_restored


def model(x, y):
  with tf.variable_scope("a"):
    x_op = tf.get_variable("x", [1], initializer=tf.constant_initializer(x))
    y_op = tf.get_variable("y", [1], initializer=tf.constant_initializer(y))
    z_op = tf.add(x_op, y_op, "z")


def first_model():
  """
  Initialized variables x and y and bound them to tf.constant_intializers(1)
  and tf.constant_initializer(2) in line (a), initialized them to 1 and 2 in
  line (b), saved these variables to a checkpoint file in line (c), yielding
  values 1 and 2 in line (d).
  """
  with tf.Graph().as_default() as G1:
    model(1, 2)  # (a)
    with tf.Session() as sess:
      z_op = G1.get_tensor_by_name("a/z:0")
      var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="a")
      sess.run(tf.variables_initializer(var_list=var_list))  # (b)
      save_first_model(sess)  # (c)
      x_value, y_value, z_value = sess.run(var_list + [z_op])  # (d)
      print("%.2f + %.2f = %.2f" % (x_value, y_value, z_value))
      # yields "1.00 + 2.00 = 3.00"


def second_model():
  """
  Initialized variables x and y and bound them to tf.constant_initializer(3)
  and tf.constant_initializer(4) in line (a), restored them to 1 and 2 in line
  (b), did not execute line (c), yielding values 1 and 2 in line (d).
  """
  with tf.Graph().as_default() as G2:
    model(3, 4)  # (a)
    with tf.Session() as sess:
      z_op = G2.get_tensor_by_name("a/z:0")
      var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="a")
      is_restored = restore_first_model(sess, var_list)  # (b)
      if not is_restored:
        sess.run(tf.variables_initializer(var_list=var_list))  # (c)
      x_value, y_value, z_value = sess.run(var_list + [z_op])  # (d)
      print("%.2f + %.2f = %.2f" % (x_value, y_value, z_value))
      # yields "1.00 + 2.00 = 3.00"


def third_model():
  """
  Initialized variables x and y and bound them to tf.constant_intializer(3)
  and tf.constant_initializer(4) in line (a), restored them to 1 and 2 in line
  (b), initialized them to 3 and 4 in line (c), yielding values 3 and 4 in
  line (d).
  """
  with tf.Graph().as_default() as G3:
    model(3, 4)  # (a)
    with tf.Session() as sess:
      z_op = G3.get_tensor_by_name("a/z:0")
      var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="a")
      restore_first_model(sess, var_list)  # (b)
      sess.run(tf.variables_initializer(var_list=var_list))  # (c)
      x_value, y_value, z_value = sess.run(var_list + [z_op])  # (d)
      print("%.2f + %.2f = %.2f" % (x_value, y_value, z_value))
      # yields "3.00 + 4.00 = 7.00"


def fourth_model():
  """
  Initialized variables x and y and bound them to tf.constant_initializer(3)
  and tf.constant_initializer(4) in line (a), initialized them to 3 and 4 in
  line (b), restored them to 1 and 2 in line (c), yielding values 1 and 2 in
  line (d).
  """
  with tf.Graph().as_default() as G4:
    model(3, 4)
    with tf.Session() as sess:
      z_op = G4.get_tensor_by_name("a/z:0")
      var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="a")
      sess.run(tf.variables_initializer(var_list=var_list))  # (b)
      restore_first_model(sess, var_list)  # (c)
      x_value, y_value, z_value = sess.run(var_list + [z_op])  # (d)
      print("%.2f + %.2f = %.2f" % (x_value, y_value, z_value))
      # yields "1.00 + 2.00 = 3.00"


def fifth_model():
  """
  Initialized variables x and y and bound them to tf.constant_initializer(3)
  and tf.constant_initializer(4) in line (a), initialized x to 3 in line (b),
  restored y to 2 in line (c), yielding values 3 and 2 in line (d).
  """
  with tf.Graph().as_default() as G5:
    model(3, 4)  # (a)
    with tf.Session() as sess:
      z_op = G5.get_tensor_by_name("a/z:0")
      var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="a/x")
      sess.run(tf.variables_initializer(var_list=var_list))  # line (b)
      var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="a/y")
      restore_first_model(sess, var_list)  # line (c)
      var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="a")
      x_value, y_value, z_value = sess.run(var_list + [z_op])  # line (d)
      print("%.2f + %.2f = %.2f" % (x_value, y_value, z_value))
      # yields "3.00 + 2.00 = 5.00"


def main(argv=None):
  first_model()
  second_model()
  third_model()
  fourth_model()
  fifth_model()


if __name__ == "__main__":
  tf.app.run()
