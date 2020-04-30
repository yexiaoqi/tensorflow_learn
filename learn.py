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

N_LAYERS = 7
ACTIVATION = tf.nn.relu
N_HIDDEN_UNITS = 30

on_train = True


def fix_seed(seed):
    np.random.seed(seed=seed)
    tf.set_random_seed(seed)


def plot_his(all_inputs,all_inputs_norm):
    for j,all_input in enumerate([all_inputs,all_inputs_norm]):
        for i,input in enumerate(all_input):
            plt.subplot(2,len(all_inputs),j*len(all_input)+(i+1))
            plt.cla()
            if i==0:
                the_range=(-7,10)
            else:
                the_range=(-1,1)
            plt.hist(input.ravel(),bins=15,range=the_range,color='r')
            plt.yticks(())
            if j==1:
                plt.xticks(())
            else:
                plt.xticks(the_range)
            ax=plt.gca()
            ax.spines['right'].set_color('none')
            ax.spines['top'].set_color('none')
        plt.title("%s normalizing" %("with" if j==0 else "with"))
        plt.draw()
        plt.pause(0.01)



def built_net(xs, ys, norm=False):
    def add_layer(inputs, in_size, out_size, activation_function=None, norm=False):
        weights = tf.Variable(tf.random_normal([in_size, out_size]), tf.float32)
        biases = tf.Variable(tf.zeros([1, out_size]) + 0.1, tf.float32)
        wx_plus_b = tf.matmul(inputs, weights) + biases

        if norm:
            fc_mean, fc_var = tf.nn.moments(wx_plus_b, axes=[0])
            scale = tf.Variable(tf.ones([out_size]))
            shift = tf.Variable(tf.zeros([out_size]))
            episilon = 0.001

            ema = tf.train.ExponentialMovingAverage(decay=0.5)

            def mean_var_with_update():
                ema_apply_op = ema.apply([fc_mean, fc_var])
                with tf.control_dependencies([ema_apply_op]):
                    return tf.identity(fc_mean), tf.identity(fc_var)

            on_train_bool = tf.constant(on_train, tf.bool)
            mean, var = tf.cond(on_train_bool, mean_var_with_update,
                                lambda: (ema.average(fc_mean), ema.average(fc_var)))

            wx_plus_b = tf.nn.batch_normalization(wx_plus_b, mean, var, shift, scale, episilon)

        if activation_function is None:
            outputs = wx_plus_b
        else:
            outputs = activation_function(wx_plus_b)
        return outputs

    fix_seed(1)

    if norm:
        fc_mean, fc_var = tf.nn.moments(xs, axes=[0])
        scale = tf.Variable(tf.ones([1]))
        shift = tf.Variable(tf.zeros([1]))
        episilon = 0.001

        ema = tf.train.ExponentialMovingAverage(decay=0.5)

        def mean_var_with_update():
            ema_apply_op = ema.apply([fc_mean, fc_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(fc_mean), tf.identity(fc_var)

        on_train_bool = tf.constant(on_train, tf.bool)
        mean, var = tf.cond(on_train_bool, mean_var_with_update, lambda: (ema.average(fc_mean), ema.average(fc_var)))

        xs = tf.nn.batch_normalization(xs, mean, var, shift, scale, episilon)

    layers_inputs = [xs]

    for i in range(N_LAYERS):
        layer_input = layers_inputs[i]
        in_size = layer_input.get_shape()[1].value

        output = add_layer(layer_input, in_size, N_HIDDEN_UNITS, ACTIVATION, norm)
        layers_inputs.append(output)

    prediction = add_layer(layers_inputs[-1], 30, 1, None)
    cost = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1]))
    train_op = tf.train.GradientDescentOptimizer(0.001).minimize(cost)
    return [train_op, cost, layers_inputs]


fix_seed(1)

xs_data=np.linspace(-7,10,2500)[:,np.newaxis]
np.random.shuffle(xs_data)
noise=np.random.normal(0,8,xs_data.shape)
ys_data=np.square(xs_data)-5+noise

plt.scatter(xs_data,ys_data)
plt.show()

xs=tf.placeholder(tf.float32,[None,1])
ys=tf.placeholder(tf.float32,[None,1])

train_op,cost,layer_inputs=built_net(xs,ys,norm=False)
train_op_norm,cost_norm,layer_inputs_norm=built_net(xs,ys,norm=True)

cost_his=[]
cost_his_norm=[]
record_step=5


init=tf.global_variables_initializer()
sess=tf.Session()
sess.run(init)

plt.ion()
plt.figure(figsize=(7,3))
for i in range(250):
    if i%50==0:
        all_inputs,all_inputs_norm=sess.run([layer_inputs,layer_inputs_norm],feed_dict={xs:xs_data,ys:ys_data})
        plot_his(all_inputs,all_inputs_norm)
    sess.run([train_op,train_op_norm],feed_dict={xs:xs_data[i*10:i*10+10],ys:ys_data[i*10:i*10+10]})
    if i%record_step==0:
        cost_his.append(sess.run(cost,feed_dict={xs:xs_data,ys:ys_data}))
        cost_his_norm.append(sess.run(cost_norm,feed_dict={xs:xs_data,ys:ys_data}))

plt.ioff()
plt.figure()
plt.plot(np.arange(len(cost_his))*record_step,np.array(cost_his),label='no bn')
plt.plot(np.arange(len(cost_his_norm))*record_step,np.array(cost_his_norm),label='bn')
plt.legend()
plt.show()