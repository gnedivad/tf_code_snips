import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
  "checkpoint_dir", "checkpoints/saver_hook", """Directory for checkpoints.""")

###############################################################################
# Response to https://github.com/tensorflow/tensorflow/issues/13265           #
###############################################################################


def main(argv=None):
  graph = tf.Graph()
  with graph.as_default():
    v = tf.get_variable("x", shape=(100, 100), dtype=tf.float32)
    save = tf.train.CheckpointSaverHook(FLAGS.checkpoint_dir, 10)

  with graph.as_default():
    tf.train.create_global_step()
    a = tf.constant(1)
    with tf.train.MonitoredSession(hooks=[save]) as sess:
      sess.run(a)

  # prints [('x', [100, 100])]
  print(tf.contrib.framework.list_variables("checkpoints/saver_hook"))


if __name__ == "__main__":
  tf.app.run()
