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





# mnist分类例子，注意需要input_data.py下载数据集
def add_layer(inputs,in_size,out_size,n_layer,activation_function=None):
    layer_name='layer%s'%n_layer
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            Weights=tf.Variable(tf.random_normal([in_size,out_size]),name='w')
            tf.summary.histogram(layer_name+'/weights',Weights)
        with tf.name_scope('biases'):
            biases=tf.Variable(tf.zeros([1,out_size])+0.1,name='b')
            tf.summary.histogram(layer_name+'/biases',biases)
        with tf.name_scope('w_with_b'):
            W_with_b=tf.add(tf.matmul(inputs,Weights),biases)
        if activation_function is None:
            outputs=W_with_b
        else:
            outputs=activation_function(W_with_b)
        tf.summary.histogram(layer_name+'/outputs',outputs)
    return outputs


def compute_accuracy(x_vs,y_vs):
    global prediction
    y_pre=sess.run(prediction,feed_dict={xs:x_vs})
    correct_prediction=tf.equal(tf.argmax(y_pre,1),tf.argmax(y_vs,1))
    accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    result=sess.run(accuracy,feed_dict={xs:x_vs,ys:y_vs})
    return result

mnist=input_data.read_data_sets('MNIST_data/',one_hot=True)

with tf.name_scope('inputs'):
    xs=tf.placeholder(tf.float32,[None,28*28],name='x_input')
    ys=tf.placeholder(tf.float32,[None,10],name='y_input')

prediction=add_layer(xs,28*28,10,n_layer=1,activation_function=tf.nn.softmax)
cross_entropy=tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction),reduction_indices=[1]))
train_step=tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
init=tf.global_variables_initializer()
sess=tf.Session()
sess.run(init)

for i in range(1000):
    batch_xs,batch_ys=mnist.train.next_batch(100)
    sess.run(train_step,feed_dict={xs:batch_xs,ys:batch_ys})
    if i%50==0:
        print(compute_accuracy(mnist.test.images,mnist.test.labels))