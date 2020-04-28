import numpy as np
import sys
import time
# import nms
import copy
import threading as td
import multiprocessing as mp
from queue import Queue
from random import randint
import pickle
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
#from tensorflow.examples.tutorials.mnist import input_data
import input_data

w=tf.Variable(np.arange(6).reshape(2,3),dtype=tf.float32,name='weights')
b=tf.Variable(np.arange(3).reshape(1,3),dtype=tf.float32,name='biases')
saver=tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess,"200428/trainsave.ckpt")
    print('weights',sess.run(w))
    print('biases',sess.run(b))
