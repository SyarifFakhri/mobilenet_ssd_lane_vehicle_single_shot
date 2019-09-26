import uff
import tensorrt as trt
from tensorrt.parsers import uffparser
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
from random import randint # generate a random test case
from PIL import Image
from matplotlib.pyplot import imshow #to show test case
import time #import system tools
import os
import tensorflow as tf


pb_name = 'D:/models-master/research/object_detection/haq_master/e_ssd_inception_aug_dataset/frozen_inference_graph.pb'
# pb_name = './haq_master/e_f_dst_all_malaysia_and_culane_dataset_aug_comb_v2_point_seg_normalized_ssd/frozen_inference_graph.pb'

with tf.gfile.FastGFile(pb_name, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

with tf.Session() as sess:
    sess.graph.as_default()
    tf.import_graph_def(graph_def, name='')

    graph = tf.get_default_graph()


		uff_model = uff.from_tensorflow(graph, ["num"])






