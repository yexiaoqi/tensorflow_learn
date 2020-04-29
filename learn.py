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
# from tensorflow.examples.tutorials.mnist import input_data
import input_data

on_train = True
ACTIVATION=tf.nn.relu
N_LAYERS=7
N_HIDDEN_UNIT=30

def fix_seed(seed=1):
    np.random.seed(seed=seed)
    tf.set_random_seed(seed)

def built_net(xs, ys, norm=False):
    # 比如inputs是100个大小为64的数据，in_size=64,输出比如说要求为50
    # 那么weights是64*50的，biases是1*50的，相加是每一行的weights都加上biases
    def add_layer(inputs, in_size, out_size, activation_function=None):
        weights = tf.Variable(tf.random_normal([in_size, out_size]), tf.float32)
        biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
        wx_plus_b = tf.matmul(inputs, weights) + biases

        if norm:
            fc_mean, fc_var = tf.nn.moments(wx_plus_b, axes=[0])
            scale = tf.Variable(tf.ones[out_size])
            shift = tf.Variable(tf.zeros[out_size])
            epsilon = 0.001

            ema = tf.train.ExponentialMovingAverage(decay=0.5)

            def mean_var_with_update():
                ema_apply_op = ema.apply([fc_mean, fc_var])
                with tf.control_dependencies([ema_apply_op]):
                    return tf.identity(fc_mean), tf.identity(fc_var)

            on_train_bool = tf.constant(on_train, tf.bool)
            mean_var = tf.cond(on_train_bool, mean_var_with_update, lambda: (ema.average(fc_mean), ema.average(fc_var)))

            wx_plus_b = tf.nn.batch_normalization(wx_plus_b, mean_var, shift, scale, epsilon)

            if activation_function is None:
                outputs = wx_plus_b
            else:
                outputs = activation_function(wx_plus_b)

            return outputs

    fix_seed(1)

    if norm:
        fc_mean,fc_var=tf.nn.moments(axes=[0])
        shift=tf.Variable(tf.zeros([1]))
        scale=tf.Variable(tf.zeros([1]))
        epsilon=0.001

        ema=tf.train.ExponentialMovingAverage(decay=0.5)
        def mean_var_with_update():
            ema_apply_op=ema.apply([fc_mean,fc_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(fc_mean),tf.identity(fc_var)

        on_train_bool=tf.constant(on_train,tf.bool)
        xs=tf.nn.batch_normalization(xs,fc_mean,fc_var,shift,scale,epsilon)

    layer_inputs=[xs]

    for l_n in range(N_LAYERS):
        layer_input=layer_inputs[l_n]
        in_size=layer_inputs[l_n].get_shape()[1].value

        output=add_layer(layer_input,in_size,N_HIDDEN_UNIT,ACTIVATION)
        layer_inputs.append(output)

    prediction=add_layer(layer_inputs[-1],30,1,activation_function=None)
    cost=tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction),reduction_indices=[1]))
    train_op=tf.train.GradientDescentOptimizer(0.001).minimize(cost)
    return [train_op,cost,layer_inputs]


fix_seed(1)
x_data=np.linspace(-7,10,2500)[:,np.newaxis]
noise=np.random.normal(0,8,x_data.shape)
y_data=np.square(x_data)-5+noise

plt.scatter(x_data,y_data)
plt.show()

xs=tf.placeholder(tf.float32,[None,1])
ys=tf.placeholder(tf.float32,[None,1])

train_op,cost,layer_inputs=built_net(xs,ys,norm=False)
train_op_norm,cost_norm,layer_inputs_norm=built_net(xs,ys,norm=True)

init=tf.global_variables_initializer()
sess=tf.Session()
sess.run(init)

cost_his=[]
cost_his_norm=[]
record_strp=5

plt.ion()
plt.figure(figsize=(7,3))



