import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from scipy.ndimage import imread
from scipy.ndimage.filters import gaussian_filter

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
  "image_filename", "data/lenna.png", """Filename for image.""")
tf.app.flags.DEFINE_boolean(
  "show", False, """Whether to show or not.""")

###############################################################################
# Tensorflow implementation of scipy.ndimage.filters.gaussian_filter for      #
# tensors.                                                                    #
###############################################################################


def expanded_gaussian_kernel(pixels, sigma):

  def gaussian_kernel(pixels, sigma):
    ax = np.arange(-pixels // 2 + 1., pixels // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    return kernel / np.sum(kernel)
  
  return np.expand_dims(
    np.expand_dims(gaussian_kernel(pixels, sigma), axis=2),
    axis=3
  )


def main(argv=None):
  image = imread(FLAGS.image_filename, flatten=True)
  blurred_image = gaussian_filter(image, sigma=2.0, order=0).astype(np.uint8)
  
  image_placeholder_op = tf.placeholder(
    tf.float32, shape=(512, 512), name="image")
  expanded_image_placeholder_op = tf.expand_dims(
    tf.expand_dims(image_placeholder_op, axis=0),
    axis=3
  )
  # Use tf.nn.conv2d instead of tf.layers.conv2d since layers must be explicit
  blurred_image_op = tf.cast(
    tf.nn.conv2d(
      expanded_image_placeholder_op,  # (1, 512, 512, 1)
      expanded_gaussian_kernel(13, 2.0),  # (512, 512, 1, 1)
      [1, 1, 1, 1],
      padding="SAME"
    ),  # (1, 512, 512, 1)
    tf.uint8
  )
  squeezed_blurred_image_op = tf.squeeze(blurred_image_op)  # (512, 512)
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    result = sess.run(
      fetches={"blurred_image": squeezed_blurred_image_op},
      feed_dict={"image:0": image}
    )
  
  if FLAGS.show:
    plt.imshow(blurred_image - result["blurred_image"], cmap="gray")
    plt.show()


if __name__ == "__main__":
  tf.app.run()
