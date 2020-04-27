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

if __name__=='__main__':
    # 拟合一条曲线例子
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

    x_data=np.linspace(-1,1,300,dtype=np.float32)[:,np.newaxis]
    noise=np.random.normal(0,0.05,x_data.shape).astype(np.float32)
    y_data=np.square(x_data)-0.5+noise

    with tf.name_scope('inputs'):
        xs=tf.placeholder(tf.float32,[None,1],name='x_input')
        ys=tf.placeholder(tf.float32,[None,1],name='y_input')

    l1=add_layer(xs,1,10,n_layer=1,activation_function=tf.nn.relu)
    prediction=add_layer(l1,10,1,n_layer=2,activation_function=None)

    with tf.name_scope('loss'):
        loss=tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction),reduction_indices=[1]))
        tf.summary.scalar('loss',loss)

    with tf.name_scope('train'):
        train_step=tf.train.GradientDescentOptimizer(0.1).minimize(loss)

    init=tf.global_variables_initializer()
    sess=tf.Session()

    merged=tf.summary.merge_all()
    sess.run(init)
    writer=tf.summary.FileWriter('log0426/seetrain',sess.graph)

    for i in range(1000):
        sess.run(train_step,feed_dict={xs:x_data,ys:y_data})
        if i%50==0:
            rs=sess.run(merged,feed_dict={xs:x_data,ys:y_data})
            writer.add_summary(rs,i)


    #动态画图
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.scatter(x_data, y_data)
    plt.ion()#本次运行请注释，全局运行不要注释

    for i in range(1000):
        # training
        sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
        if i % 50 == 0:
            # to visualize the result and improvement
            try:
                ax.lines.remove(lines[0])
            except Exception:
                pass
            prediction_value = sess.run(prediction, feed_dict={xs: x_data})
            # plot the prediction
            lines = ax.plot(x_data, prediction_value, 'r-', lw=5)
            plt.pause(0.1)

    plt.ioff()
    plt.show()