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


with tf.variable_scope('a_variable_scope') as scope:
    initializer=tf.constant_initializer(value=3)
    var3=tf.get_variable(name='var3',shape=[1],dtype=tf.float32,initializer=initializer)
    scope.reuse_variables()
    var3_reuse=tf.get_variable(name='var3')
    var4=tf.Variable(name='var4',initial_value=[4],dtype=tf.float32)
    var4_reuse=tf.Variable(name='var4',initial_value=[4],dtype=tf.float32)

with tf.Session() as sess:
    init=tf.global_variables_initializer()
    sess.run(init)
    print(var3.name)
    print(sess.run(var3))
    print(var3_reuse.name)
    print(sess.run(var3_reuse))
    print(var4.name)
    print(sess.run(var4.name))
    print(var4_reuse.name)
    print(sess.run(var4_reuse))




