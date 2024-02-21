
import os
import sys
import time
from glob import glob
from os.path import join
from time import time
# to make run from console for module import
sys.path.append(os.path.abspath(".."))
import tensorflow as tf

import scipy.io as sio
import cv2
import numpy as np
from main.config import Config
config = Config()
config.save_config()
config.display()
data_dir = config.DATA_DIR
datasets = config.DATASETS
tf_record_dirs = [join(data_dir, dataset, '*.tfrecord') for dataset in datasets]
tf_records = [tf_record for tf_records in sorted([glob(f) for f in tf_record_dirs]) for tf_record in tf_records]


def int64_feature(value):
    """Wrapper for inserting int64 features into Example proto."""
    if not isinstance(value, list) and not isinstance(value, np.ndarray):
        value = [value]
    if isinstance(value, np.ndarray):
        value = np.reshape(value, -1)
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def float_feature(value):
    """Wrapper for inserting float features into Example proto."""
    if not isinstance(value, list) and not isinstance(value, np.ndarray):
        value = [value]
    if isinstance(value, np.ndarray):
        value = np.reshape(value, -1)
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def bytes_feature(value):
    """Wrapper for inserting bytes features into Example proto."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

class ExceptionHandlingIterator:
    """This class was introduced to avoid tensorflow.python.framework.errors_impl.InvalidArgumentError
        thrown while iterating over the zipped datasets.

        One assumption is that the tf records contain one wrongly generated set due to following error message:
            Expected begin[1] in [0, 462], but got -11 [[{{node Slice}}]] [Op:IteratorGetNextSync]
    """

    def __init__(self, iterable):
        self._iter = iter(iterable)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            return self._iter.__next__()
        except StopIteration as e:
            raise e
        except Exception as e:
            print(e)
            return self.__next__()

def parse(example_proto):
    feature_map = {
        'image_raw': tf.io.FixedLenFeature([], dtype=tf.string, default_value=''),
        'keypoints_2d': tf.io.VarLenFeature(dtype=tf.float32),
        'keypoints_3d': tf.io.VarLenFeature(dtype=tf.float32),
        'has_3d': tf.io.FixedLenFeature([], dtype=tf.int64),
    }
    features = tf.io.parse_single_example(example_proto, feature_map)
    image_data = features['image_raw']
    kp2d = tf.sparse.to_dense(features['keypoints_2d'])
    kp2d = tf.reshape(kp2d, (19, 3))

    # indices=tf.constant(np.arange(self.config.NUM_KP2D))
    # kp2d=tf.gather(kp2d,indices)
    # kp2d = tf.reshape(tf.sparse.to_dense(features['keypoints_2d']), (self.config.NUM_KP2D, 3))
    # kp3d=tf.sparse.to_dense(features['keypoints_3d'])
    # kp3d = tf.reshape(kp3d, (14, 3))
    # kp3d=kp3d[:self.config.NUM_KP3D,:]
    kp3d = tf.reshape(tf.sparse.to_dense(features['keypoints_3d']), (config.NUM_KP3D, 3))
    has_3d = features['has_3d']

    return image_data, kp2d, kp3d, has_3d

def parse_new(example_proto):
    feature_map = {
        'image_raw_path': tf.io.FixedLenFeature([], dtype=tf.string),
        'keypoints_2d': tf.io.VarLenFeature(dtype=tf.float32),
        'keypoints_3d': tf.io.VarLenFeature(dtype=tf.float32),
        'has_3d': tf.io.FixedLenFeature([], dtype=tf.int64),

    }
    features = tf.io.parse_single_example(example_proto, feature_map)
    image_data_path = features['image_raw_path']

    def _preprocess(filename):
        raw = tf.io.read_file(filename)
        image = tf.image.decode_png(raw, channels=3)
        return image

    image_data_path = tf.strings.regex_replace(image_data_path, '/opt/project/tf_data', data_dir)
    image_data = tf.py_function(_preprocess,[image_data_path],[tf.uint8])


    kp2d = tf.sparse.to_dense(features['keypoints_2d'])
    kp2d = tf.reshape(kp2d, (19, 3))
    kp2d = kp2d[:config.NUM_KP2D, :]
    # indices=tf.constant(np.arange(self.config.NUM_KP2D))
    # kp2d=tf.gather(kp2d,indices)
    # kp2d = tf.reshape(tf.sparse.to_dense(features['keypoints_2d']), (self.config.NUM_KP2D, 3))
    # kp3d=tf.sparse.to_dense(features['keypoints_3d'])
    # kp3d = tf.reshape(kp3d, (14, 3))
    # kp3d=kp3d[:self.config.NUM_KP3D,:]
    kp3d = tf.reshape(tf.sparse.to_dense(features['keypoints_3d']), (config.NUM_KP3D, 3))
    has_3d = features['has_3d']



    return image_data, kp2d, kp3d, has_3d




# def readmat(image_data_path, kp2d, kp3d, has_3d):
#     def _preprocess(filename):
#         filename=filename.numpy()
#         filename=filename[0]
#         data = sio.loadmat(filename)
#         data = tf.dtypes.cast(data['image'], dtype=tf.float32)
#         return data
#
#     image_data_path = tf.strings.regex_replace(image_data_path, '/opt/project/tf_data', data_dir)
#     image_data = tf.py_function(_preprocess,[image_data_path],[tf.float32])
#
#     return image_data, kp2d, kp3d, has_3d

def _bytes_feature(value):
    if isinstance(value, type(tf.constant(0))):
        value=value.numpy()
        return tf.train.Feature(bytes_list=tf.train.BytesList(valye=[value]))

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

for tf_record in tf_records:
    print(tf_record)
    dataset = tf.data.TFRecordDataset(tf_record)
    dataset = dataset.map(parse, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset=dataset.batch(1)
    # a=list(dataset.as_numpy_iterator())
    dataset_train = ExceptionHandlingIterator(dataset)
    new_tfrecord_file = tf_record.split('.')[0]+'_new.'+tf_record.split('.')[1]

    with tf.io.TFRecordWriter(new_tfrecord_file) as writer:
        image_folder=new_tfrecord_file.split('.')[0]
        if not os.path.exists(image_folder):
            os.makedirs(image_folder)
        for i, example in enumerate(dataset_train):
            image_data, kp2d, kp3d, has_3d=example
            image_data=image_data[0]
            kp2d=kp2d[0]
            kp3d=kp3d[0]
            has_3d=has_3d[0]
            image_path=image_folder+'/%06d.png'%(i)
            image_raw=tf.image.decode_jpeg(image_data, channels=3)

            image_numpy=image_raw.numpy()

            cv2.imwrite(image_path, cv2.cvtColor(image_numpy,cv2.COLOR_RGB2BGR))
            raw = tf.io.read_file(image_path)
            image_re = tf.image.decode_png(raw, channels=3)
            # sio.savemat(, {'image':image_numpy})
            kp2d=kp2d.numpy()
            kp3d=kp3d.numpy()
            has_3d=has_3d.numpy()



            feat_dict = {
                'image_raw_path': bytes_feature(tf.compat.as_bytes(image_path)),
                'keypoints_2d': float_feature(kp2d),
                'keypoints_3d': float_feature(kp3d),
                'has_3d': int64_feature(has_3d),

            }
            example_write=tf.train.Example(features=tf.train.Features(feature=feat_dict))
            writer.write(example_write.SerializeToString())

    # dataset = tf.data.TFRecordDataset(new_tfrecord_file)
    # dataset = dataset.map(parse_new, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # # dataset=dataset.map(readmat)
    # dataset=dataset.batch(1,drop_remainder=True)
    # dataset_train = ExceptionHandlingIterator(dataset)
    # for i, example in enumerate(dataset_train):
    #     image_data_path, kp2d, kp3d, has_3d = example




