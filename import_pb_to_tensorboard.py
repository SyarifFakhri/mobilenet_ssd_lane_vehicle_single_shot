# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ================================
"""Imports a protobuf model as a graph in Tensorboard."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

import tensorflow as tf

from tensorflow.core.framework import graph_pb2
from tensorflow.python.client import session
from tensorflow.python.framework import importer
from tensorflow.python.framework import ops
from tensorflow.python.platform import app
from tensorflow.python.platform import gfile
from tensorflow.python.summary import summary
# D:\models-master\research\object_detection\ssd_mobilenet_v2_coco_2018_03_29 model_dir =
# "D:/models-master/research/object_detection/haq_master
# /e_f_dst_all_malaysia_and_culane_dataset_aug_comb_v2_point_seg_normalized/frozen_inference_graph.pb"
model_dir = "D:/models-master/research/object_detection/ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.pb"
# log_dir = "D:/models-master/research/object_detection/haq_master/e_f_dst_all_malaysia_and_culane_dataset_aug_comb_v2_point_seg_normalized/pb_vis"
log_dir = "D:/models-master/research/object_detection/ssd_mobilenet_v2_coco_2018_03_29/pb_vis"

def import_to_tensorboard(model_dir, log_dir):
  """View an imported protobuf model (`.pb` file) as a graph in Tensorboard.
  Args:
    model_dir: The location of the protobuf (`pb`) model to visualize
    log_dir: The location for the Tensorboard log to begin visualization from.
  Usage:
    Call this function with your model location and desired log directory.
    Launch Tensorboard by pointing it to the log directory.
    View your imported `.pb` model as a graph.
  """
  with session.Session(graph=ops.Graph()) as sess:
    with gfile.FastGFile(model_dir, "rb") as f:
      graph_def = graph_pb2.GraphDef()
      graph_def.ParseFromString(f.read())
      importer.import_graph_def(graph_def)

    pb_visual_writer = summary.FileWriter(log_dir)
    pb_visual_writer.add_graph(sess.graph)

    graph = tf.get_default_graph()
    tensors_per_node = [node.values() for node in graph.get_operations()]
    tensor_names = [tensor.name for tensors in tensors_per_node for tensor in tensors]
    for i in range(len(tensor_names)):
      print(tensor_names[i])

    print("Model Imported. Visualize by running: "
          "tensorboard --logdir={}".format(log_dir))


def main(unused_args):
  import_to_tensorboard(model_dir, log_dir)

if __name__ == "__main__":
  import_to_tensorboard(model_dir, log_dir)