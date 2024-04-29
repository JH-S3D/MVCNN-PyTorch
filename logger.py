import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io

class Logger(object):
    def __init__(self, log_dir):
        """Create a summary writer logging to log_dir."""
        self.writer = tf.summary.create_file_writer(log_dir)

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        with self.writer.as_default():
            tf.summary.scalar(tag, value, step=step)
            self.writer.flush()

    def image_summary(self, tag, images, step):
        """Log a list of images."""
        with self.writer.as_default():
            for i, img in enumerate(images):
                # Convert numpy array to PIL Image object and then to image tensor
                if img.dtype == np.uint8:
                    img = Image.fromarray(img)
                else:
                    img = Image.fromarray((img * 255).astype(np.uint8))
                buffer = io.BytesIO()
                img.save(buffer, format='PNG')
                buffer.seek(0)
                image_tensor = tf.io.decode_png(buffer.getvalue(), channels=4)
                tf.summary.image(f"{tag}/{i}", tf.expand_dims(image_tensor, 0), step=step)
            self.writer.flush()

    def histo_summary(self, tag, values, step, bins=1000):
        """Log a histogram of the tensor of values."""
        with self.writer.as_default():
            tf.summary.histogram(tag, values, buckets=bins, step=step)
            self.writer.flush()
