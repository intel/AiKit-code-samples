#!/usr/bin/env python
# coding: utf-8

# #  Getting Started with TensorFlow low-precision int8 inference
# 
# This code sample will serve as a sample use case to perform low precision int8 inference on a synthetic data implementing a ResNet50 pre-trained model. The pre-trained model published as part of Intel Model Zoo will be used in this sample. 

# In[1]:



# Import statements
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import time

import tensorflow as tf
import os


# In[2]:


#download Intel's pretrained resnet50 model
try:
    get_ipython().system('wget https://storage.googleapis.com/intel-optimized-tensorflow/models/resnet50_int8_pretrained_model.pb')
except:
    import urllib.request
    urllib.request.urlretrieve('https://storage.googleapis.com/intel-optimized-tensorflow/models/resnet50_int8_pretrained_model.pb', 'resnet50_int8_pretrained_model.pb')


# We will be using a synthetic dataset of size 244x244.
# It is important to set optimial batch_size, MKL run-time settings, TensorFlow's inter-intra number of threads to enable compute and data layer optimizations. We have identified  optimial settings for popular topologies including ResNet50 to maximize CPU utlization. For more details on Run-time settings refer to blogs [maximize CPU performance](https://software.intel.com/en-us/articles/maximize-tensorflow-performance-on-cpu-considerations-and-recommendations-for-inference), [Intel Model Zoo tutorials](https://github.com/IntelAI/models/tree/master/docs). 
#  

# In[3]:


try:
    physical_cores = get_ipython().getoutput("lscpu -b -p=Core,Socket | grep -v '^#' | sort -u | wc -l")
except:
    physical_cores = [str(os.cpu_count())]


# In[4]:


model_file = "resnet50_int8_pretrained_model.pb"
input_height = 224
input_width = 224
batch_size = 64
input_layer = "input" # input tensor name from the stored graph
output_layer = "predict"# input tensor name to be computed
warmup_steps = 10
steps = 50

os.environ["KMP_BLOCKTIME"] = "1"
os.environ["KMP_SETTINGS"] = "1"
os.environ["KMP_AFFINITY"]= "granularity=fine,verbose,compact,1,0"
os.environ["OMP_NUM_THREADS"]= physical_cores[0]
num_inter_threads = 2
num_intra_threads = int(physical_cores[0])
data_config = tf.ConfigProto()
data_config.intra_op_parallelism_threads = 16 
data_config.inter_op_parallelism_threads = 14 
data_config.use_per_session_threads = 1

infer_config = tf.ConfigProto()
infer_config.intra_op_parallelism_threads = num_intra_threads
infer_config.inter_op_parallelism_threads = num_inter_threads
infer_config.use_per_session_threads = 1


# Create data graph, and infer graph from pre-trained int8 resnet50 model

# In[5]:


data_graph = tf.Graph()
with data_graph.as_default():
    input_shape = [batch_size, input_height, input_width, 3]
    images = tf.random.uniform(input_shape, 0.0, 255.0, dtype=tf.float32, name='synthetic_images')

infer_graph = tf.Graph()
with infer_graph.as_default():
    graph_def = tf.GraphDef()
    with open(model_file, "rb") as f:
      graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')


# Create data and infer sessions for optimized data access and graph computation configured with best thread settings for Resnet50 and run warm-up steps
# 

# In[6]:


input_tensor = infer_graph.get_tensor_by_name(input_layer + ":0")
output_tensor = infer_graph.get_tensor_by_name(output_layer + ":0")
tf.global_variables_initializer()

data_sess = tf.Session(graph=data_graph, config=data_config)
infer_sess = tf.Session(graph=infer_graph, config=infer_config)

print("[Running warmup steps...]")
step_total_time = 0
step_total_images = 0

for t in range(warmup_steps):
    data_start_time = time.time()
    image_data = data_sess.run(images)
    data_load_time = time.time() - data_start_time

    start_time = time.time()
    infer_sess.run(output_tensor, {input_tensor: image_data})
    elapsed_time = time.time() - start_time

    step_total_time += elapsed_time
    step_total_images += batch_size

    if ((t + 1) % 10 == 0):
      print("steps = {0}, {1} images/sec"
            "".format(t + 1, step_total_images / step_total_time))
      step_total_time = 0
      step_total_images = 0

print("[Running benchmark steps...]")
total_time = 0
total_images = 0

step_total_time = 0
step_total_images = 0


# Run training steps with batch size 64 to measure average throughput

# In[7]:


for t in range(steps):
    try:
      data_start_time = time.time()
      image_data = data_sess.run(images)
      data_load_time = time.time() - data_start_time

      start_time = time.time()
      infer_sess.run(output_tensor, {input_tensor: image_data})
      elapsed_time = time.time() - start_time


      total_time += elapsed_time
      total_images += batch_size

      step_total_time += elapsed_time
      step_total_images += batch_size

      if ((t + 1) % 10 == 0):
        print("steps = {0}, {1} images/sec"
              "".format(t + 1, step_total_images / step_total_time))
        step_total_time = 0
        step_total_images = 0

    except tf.errors.OutOfRangeError:
      print("Running out of images from dataset.")
      break

print("Average throughput for batch size {0}: {1} images/sec".format(batch_size, total_images / total_time))


# In[8]:


print('[CODE_SAMPLE_COMPLETED_SUCCESFULLY]')


# In[ ]:




