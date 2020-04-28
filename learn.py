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

mnist=input_data.read_data_sets('MNIST_data',one_hot=False)

learning_rate=0.01
batch_size=256
display_fre=1
training_epoch=5
examples_to_show=10

n_input=784

x=tf.placeholder(tf.float32,[None,n_input])

n_hidden1=128
n_hidden2=64
n_hidden3=10
n_hidden4=2
weights={
    'encoder_h1':tf.Variable(tf.random_normal([n_input,n_hidden1])),
    'encoder_h2':tf.Variable(tf.random_normal([n_hidden1,n_hidden2])),
    'encoder_h3':tf.Variable(tf.random_normal([n_hidden2,n_hidden3])),
    'encoder_h4':tf.Variable(tf.random_normal([n_hidden3,n_hidden4])),

    'decoder_h1':tf.Variable(tf.random_normal([n_hidden4,n_hidden3])),
    'decoder_h2':tf.Variable(tf.random_normal([n_hidden3,n_hidden2])),
    'decoder_h3':tf.Variable(tf.random_normal([n_hidden2,n_hidden1])),
    'decoder_h4':tf.Variable(tf.random_normal([n_hidden1,n_input]))
}

biases={
    'encoder_b1':tf.Variable(tf.random_normal([n_hidden1])),
    'encoder_b2':tf.Variable(tf.random_normal([n_hidden2])),
    'encoder_b3': tf.Variable(tf.random_normal([n_hidden3])),
    'encoder_b4': tf.Variable(tf.random_normal([n_hidden4])),

    'decoder_b1':tf.Variable(tf.random_normal([n_hidden3])),
    'decoder_b2':tf.Variable(tf.random_normal([n_hidden2])),
    'decoder_b3':tf.Variable(tf.random_normal([n_hidden1])),
    'decoder_b4':tf.Variable(tf.random_normal([n_input]))
}

def encoder(x):
    layer_1=tf.nn.sigmoid(tf.matmul(x,weights['encoder_h1'])+biases['encoder_b1'])
    layer_2=tf.nn.sigmoid(tf.matmul(layer_1,weights['encoder_h2'])+biases['encoder_b2'])
    layer_3 = tf.nn.sigmoid(tf.matmul(layer_2, weights['encoder_h3']) + biases['encoder_b3'])
    layer_4 = tf.matmul(layer_3, weights['encoder_h4']) + biases['encoder_b4']
    return layer_4

def decoder(x):
    layer_1 = tf.nn.sigmoid(tf.matmul(x, weights['decoder_h1']) + biases['decoder_b1'])
    layer_2 = tf.nn.sigmoid(tf.matmul(layer_1, weights['decoder_h2']) + biases['decoder_b2'])
    layer_3 = tf.nn.sigmoid(tf.matmul(layer_2, weights['decoder_h3']) + biases['decoder_b3'])
    layer_4 = tf.nn.sigmoid(tf.matmul(layer_3, weights['decoder_h4']) + biases['decoder_b4'])
    return layer_4

encoder_op=encoder(x)
decoder_op=decoder(encoder_op)

y_pre=decoder_op
y_true=x

cost=tf.reduce_mean(tf.pow(y_pre-y_true,2))
train_step=tf.train.AdamOptimizer(learning_rate).minimize(cost)

with tf.Session() as sess:
    init=tf.global_variables_initializer()
    sess.run(init)
    total_batch=int(mnist.train.num_examples/batch_size)
    for step in range(training_epoch):
        for i in range(total_batch):
            batch_xs,batch_ys=mnist.train.next_batch(batch_size)
            _,c=sess.run([train_step,cost],feed_dict={x:batch_xs})
        if step%display_fre==0:
            print("epoch:","%4d"%step,"cost=","{:.9f}".format(c))

    print("Optimization finished!")

    # encode_decode=sess.run(y_pre,feed_dict={x:mnist.test.images[:examples_to_show]})
    # fig,ax=plt.subplots(2,10,figsize=[10,2])
    # for i in range(examples_to_show):
    #     ax[0][i].imshow(np.reshape(mnist.test.images[i],(28,28)))
    #     ax[1][i].imshow(np.reshape(encode_decode[i],(28,28)))
    #plt.show()

    encoder_result=sess.run(encoder_op,feed_dict={x:mnist.test.images})
    plt.scatter(encoder_result[:,0],encoder_result[:,1],c=mnist.test.labels)
    plt.colorbar()
    plt.show()



