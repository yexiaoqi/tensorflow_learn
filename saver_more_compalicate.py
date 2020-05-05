# import nms
import tensorflow as tf
# from tensorflow.examples.tutorials.mnist import input_data
import input_data
import numpy as np


class Train_Mnist:
    def __init__(self):
        self.xs = tf.placeholder(tf.float32, [None, 28 * 28])
        self.ys = tf.placeholder(tf.float32, [None, 10])

        self.prediction = self.add_layer(self.xs, 28 * 28, 10, n_layer=1, activation_function=tf.nn.softmax)

        self.sess = tf.Session()

        self.cross_entropy = tf.reduce_mean(-tf.reduce_sum(self.ys * tf.log(self.prediction), reduction_indices=[1]))
        self.train_step = tf.train.GradientDescentOptimizer(0.4).minimize(self.cross_entropy)

        # tensorflow默认只会保存最近的5个模型文件，如果你希望保存更多，可以通过max_to_keep来指定
        self.saver = tf.train.Saver(max_to_keep=20)
        # saver = tf.train.Saver( keep_checkpoint_every_n_hours=2)#希望每2个小时保存一次
        # saver = tf.train.Saver()
        '''如果没有在tf.train.Saver()
        指定任何参数，这样表示默认保存所有变量。如果我们不希望保存所有变量，而只是其中的一部分，此时我们可以指点要保存的变量或者集合：我们只需在创建tf.train.Saver的时候把一个列表或者要保存变量的字典作为参数传进去。'''
        # saver = tf.train.Saver([weights])

        init = tf.global_variables_initializer()
        self.sess.run(init)

    def add_layer(self, inputs, in_size, out_size, n_layer, activation_function=None):
        layer_name = 'layer%s' % n_layer
        with tf.name_scope(layer_name):
            with tf.name_scope('weights'):
                Weights = tf.Variable(tf.random_normal([in_size, out_size]), name='w')
                tf.summary.histogram(layer_name + '/weights', Weights)
            with tf.name_scope('biases'):
                biases = tf.Variable(tf.zeros([1, out_size]) + 0.1, name='b')
                tf.summary.histogram(layer_name + '/biases', biases)
            with tf.name_scope('w_with_b'):
                W_with_b = tf.add(tf.matmul(inputs, Weights), biases)
            if activation_function is None:
                outputs = W_with_b
            else:
                outputs = activation_function(W_with_b)
            tf.summary.histogram(layer_name + '/outputs', outputs)
        return outputs

    def compute_accuracy(self, v_xs, v_ys):
        y_pre = self.sess.run(self.prediction, feed_dict={self.xs: v_xs})
        correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        result = self.sess.run(accuracy, feed_dict={self.xs: v_xs, self.ys: v_ys})
        return result

    def train(self, xs, ys):
        loss, _ = self.sess.run([self.cross_entropy, self.train_step], feed_dict={self.xs: xs, self.ys: ys})
        return loss

    def save(self, i):
        # global_step是保存第几次的模型
        # 网络结构没有变化，没必要重复保存.meta文件
        if i == 0:
            self.saver.save(self.sess, "./models200504/model.ckpt", global_step=i)
        else:
            self.saver.save(self.sess, "./models200504/model.ckpt", global_step=i, write_meta_graph=False)

    def predict(self,x):
        self.saver=tf.train.import_meta_graph('./models200504/model.ckpt-0.meta')
        # self.saver.restore(self.sess,tf.train.latest_checkpoint('./models200504/'))
        self.saver.restore(self.sess,'./models200504/model.ckpt-0')
        #print(self.sess.run('Weights:0'))
        result=self.sess.run(self.prediction,{self.xs:x})
        return result

def train():
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    # mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)
    train200504 = Train_Mnist()
    for i in range(2000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        train_loss = train200504.train(batch_xs, batch_ys)
        if i % 100 == 0:
            print("loss=",train_loss,"acc=",train200504.compute_accuracy(mnist.test.images, mnist.test.labels))
            train200504.save(i)

def predict(data,i):
    result=np.argmax(train200504.predict(data.test.images[i,:][np.newaxis,:]),1)
    label=np.argmax(mnist.test.labels[i,:][np.newaxis,:],1)
    print('prediciton:',result,'label:',label)


if __name__ == '__main__':
    # train()

    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    train200504 = Train_Mnist()
    for i in range(10):
        predict(mnist,i)
