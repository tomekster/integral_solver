import glob
import numpy as np
import cv2
import random 
import csv 
import tensorflow as tf
import os

from object_detection.util import dataset_util

HEIGHT = 400
WIDTH = 400
EXAMPLE_SIZE = 32
STRIDE = 3
TOTAL = EXAMPLE_SIZE + 2 * STRIDE

HASY_LABELS = 'hasy-data-labels.csv'
HASY_SYMBOLS = 'symbols.csv'
IMAGES_DIR = 'train_images/'
IMAGES_COUNT = 0

flags = tf.app.flags
flags.DEFINE_string('hasy_dir', '', 'Path to HASYv2 directory')
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
FLAGS = flags.FLAGS

def chunks(l,n=100):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

def create_tf_example(example_paths):
    canva = np.full((HEIGHT, WIDTH), 255, np.uint8)
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes = []
    classes_text = []

    # Fill canva to create image detection example
    for i in range(HEIGHT // TOTAL):
        for j in range(WIDTH // TOTAL):

            if not example_paths:
                continue

            offsets = [-2, -1, 0, 1, 2]
            dx = random.choice(offsets)
            dy = random.choice(offsets)
            ypos = i*TOTAL + STRIDE + dy
            xpos = j*TOTAL + STRIDE + dx
            
            example_path = example_paths.pop()
            example = cv2.imread(example_path, cv2.IMREAD_GRAYSCALE)
            
            canva[  ypos : ypos + EXAMPLE_SIZE, 
                    xpos : xpos + EXAMPLE_SIZE] = example

            success, encoded_img = cv2.imencode('.png', canva)
            encoded_image_data = encoded_img.tobytes()

            # Save information about objects in the image
            xmins.append(xpos / WIDTH)
            xmaxs.append( (xpos + EXAMPLE_SIZE) / WIDTH )
            ymins.append(ypos / HEIGHT)
            ymaxs.append( (ypos + EXAMPLE_SIZE) / HEIGHT )
            filename = example_path.remove(FLAGS.hasy_dir)
            class_id = labelmap[filename]
            classes.append(class_id)
            classes_text.append(symbolmap(class_id))
    # Create tf.Example object
    tf_example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': dataset_util.int64_feature(HEIGHT),
      'image/width': dataset_util.int64_feature(WIDTH),
      'image/filename': dataset_util.bytes_feature(filename),
      'image/source_id': dataset_util.bytes_feature(filename),
      'image/encoded': dataset_util.bytes_feature(encoded_image_data),
      'image/format': dataset_util.bytes_feature(b'png'),
      'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
      'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
      'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
      'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
      'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
      'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return canva, tf_example

def main(_):
    with open(FLAGS.hasy_dir + HASY_LABELS) as hasy_labels_file:
        reader = csv.reader(hasy_labels_file)
        labelmap = { rows[0] : int(rows[1]) for rows in reader }

    with open(FLAGS.hasy_dir + HASY_SYMBOLS) as hasy_symbols_file:
        reader = csv.reader(hasy_symbols_file)
        symbolmap = { int(rows[0]) : rows[1] for rows in reader }

    hasy_examples = glob.glob(FLAGS.hasy_dir + 'hasy-data/*.png')
    random.shuffle(hasy_examples)

    writer = tf.python_io.TFRecordWriter(FLAGS.output_path)
 
    if not os.path.isdir(IMAGES_DIR):
        os.mkdir(IMAGES_DIR)

    for chunk in chunks(hasy_examples[:5]):
        image, tf_example = create_tf_example(chunk)
        writer.write(tf_example.SerializeToString())
        image_name = str(IMAGES_COUNT).zfill(5)
        IMAGES_COUNT += 1
        cv2.imwrite(IMAGES_DIR + image_name + '.png', canva)
        cv2.imshow('blank', image)
        cv2.waitKey(0)
    writer.close()

if __name__ == '__main__':
    tf.app.run()

