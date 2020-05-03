from urllib.request import urlretrieve
import os
import numpy as np
import tensorflow as tf
import skimage.io
import skimage.transform
import matplotlib.pyplot as plt


def download():     # download tiger and kittycat image
    categories = ['tiger', 'kittycat']
    for category in categories:
        os.makedirs('./for_transfer_learning/data/%s' % category, exist_ok=True)
        with open('./for_transfer_learning/imagenet_%s.txt' % category, 'r') as file:
            urls = file.readlines()
            n_urls = len(urls)
            for i, url in enumerate(urls):
                try:
                    urlretrieve(url.strip(), './for_transfer_learning/data/%s/%s' % (category, url.strip().split('/')[-1]))
                    print('%s %i/%i' % (category, i, n_urls))
                except:
                    print('%s %i/%i' % (category, i, n_urls), 'no image')


def load_img(path):
    img = skimage.io.imread(path)
    img = img / 255.0
    # print "Original Image Shape: ", img.shape
    # we crop image from center
    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy: yy + short_edge, xx: xx + short_edge]
    # resize to 224, 224
    resized_img = skimage.transform.resize(crop_img, (224, 224))[None, :, :, :]   # shape [1, 224, 224, 3]
    return resized_img


def load_data():
    imgs = {'tiger': [], 'kittycat': []}
    for k in imgs.keys():
        dir = './for_transfer_learning/data/' + k
        for file in os.listdir(dir):
            if not file.lower().endswith('.jpg'):
                continue
            try:
                resized_img = load_img(os.path.join(dir, file))
            except OSError:
                continue
            imgs[k].append(resized_img)    # [1, height, width, depth] * n
            if len(imgs[k]) == 400:        # only use 400 imgs to reduce my memory load
                break
    # fake length data for tiger and cat
    tigers_y = np.maximum(20, np.random.randn(len(imgs['tiger']), 1) * 30 + 100)
    cat_y = np.maximum(10, np.random.randn(len(imgs['kittycat']), 1) * 8 + 40)
    return imgs['tiger'], imgs['kittycat'], tigers_y, cat_y


class Vgg16:
    VGG_MEAN = [103.939, 116.779, 123.68]

    def __init__(self, vgg_npy_path=None,restore_from=None):
        try:
            self.data_dict = np.load(vgg_npy_path, encoding='latin1',
                                     allow_pickle=True).item()  # 由于numpy版本较低，allow_pickle在低版本中默认为False

        except FileNotFoundError:
            print("Please download VGG16 parameters")

        self.tfx=tf.placeholder(tf.float32,[None,224,224,3])
        self.tfy=tf.placeholder(tf.float32,[None,1])

        red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=self.tfx*255.0)
        assert red.get_shape().as_list()[1:] == [224, 224, 1]
        assert green.get_shape().as_list()[1:] == [224, 224, 1]
        assert blue.get_shape().as_list()[1:] == [224, 224, 1]
        bgr = tf.concat(axis=3, values=[
            blue - self.VGG_MEAN[0],
            green - self.VGG_MEAN[1],
            red - self.VGG_MEAN[2]])
        assert bgr.get_shape().as_list()[1:] == [224, 224, 3]

        conv1_1 = self.conv_layer(bgr, "conv1_1")
        conv1_2 = self.conv_layer(conv1_1, "conv1_2")
        pool1 = self.max_pool(conv1_2, 'pool1')

        conv2_1 = self.conv_layer(pool1, "conv2_1")
        conv2_2 = self.conv_layer(conv2_1, "conv2_2")
        pool2 = self.max_pool(conv2_2, 'pool2')

        conv3_1 = self.conv_layer(pool2, "conv3_1")
        conv3_2 = self.conv_layer(conv3_1, "conv3_2")
        conv3_3 = self.conv_layer(conv3_2, "conv3_3")
        pool3 = self.max_pool(conv3_3, 'pool3')

        conv4_1 = self.conv_layer(pool3, "conv4_1")
        conv4_2 = self.conv_layer(conv4_1, "conv4_2")
        conv4_3 = self.conv_layer(conv4_2, "conv4_3")
        pool4 = self.max_pool(conv4_3, 'pool4')

        conv5_1 = self.conv_layer(pool4, "conv5_1")
        conv5_2 = self.conv_layer(conv5_1, "conv5_2")
        conv5_3 = self.conv_layer(conv5_2, "conv5_3")
        pool5 = self.max_pool(conv5_3, 'pool5')

        self.flatten=tf.reshape(pool5,[-1,7*7*512])
        self.fc6=tf.layers.dense(self.flatten,256,tf.nn.relu,name='fc6')
        self.out=tf.layers.dense(self.fc6,1,name='out')

        self.sess=tf.Session()
        if restore_from:
            saver=tf.train.Saver()
            saver.restore(self.sess,restore_from)
        else:
            self.loss=tf.losses.mean_squared_error(labels=self.tfy,predictions=self.out)
            self.train_op=tf.train.RMSPropOptimizer(0.001).minimize(self.loss)
            self.sess.run(tf.global_variables_initializer())

    def get_conv_filter(self, name):
        return tf.constant(self.data_dict[name][0], name="filter")

    def get_bias(self, name):
        return tf.constant(self.data_dict[name][1], name="biases")

    def get_fc_weight(self, name):
        return tf.constant(self.data_dict[name][0], name="weights")

    def conv_layer(self, bottom, name):
        with tf.variable_scope(name):
            filt = self.get_conv_filter(name)
            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')
            conv_biases = self.get_bias(name)
            bias = tf.nn.bias_add(conv, conv_biases)
            relu = tf.nn.relu(bias)
            return relu

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def train(self,x,y):
        loss,_=self.sess.run([self.loss,self.train_op],{self.tfx:x,self.tfy:y})
        return  loss

    def predict(self,paths):
        fig,axs=plt.subplots(1,2)
        for i,path in enumerate(paths):
            x=load_img(path)
            length=self.sess.run(self.out,{self.tfx:x})
            axs[i].imshow(x[0])
            axs[i].set_title('len:%.1f cm'%length)
            axs[i].set_xticks(())
            axs[i].set_ytick(())
        plt.show()

    def save(self,path='./for_transfer_learning/models/transfer_learn'):
        saver=tf.train.Saver()
        saver.save(self.sess,path,write_meta_graph=False)

def train():
    tigers_x,cats_x,tigers_y,cats_y=load_data()

    plt.hist(tigers_y,bins=20,label='tigers')
    plt.hist(cats_y,bins=10,label='cats')
    plt.legend()
    plt.xlabel('length')
    plt.show()

    xs=np.concatenate(tigers_x+cats_x,axis=0)
    ys=np.concatenate((tigers_y,cats_y),axis=0)

    vgg=Vgg16(vgg_npy_path='./for_transfer_learning/models/vgg16.npy')
    print('net built')
    for i in range(100):
        b_idx=np.random.randint(0,len(xs),6)
        train_loss=vgg.train(xs[b_idx],ys[b_idx])
        print(i,'train loss: ',train_loss)

    vgg.save('./for_transfer_learning/models/transfer_learn')

def eval():
    vgg=Vgg16(vgg_npy_path='./models/vgg16.npy',restore_from='./models/200503')
    vgg.predict(['./for_transfer_learning/data/kittycat/000129037.jpg', './for_transfer_learning/data/tiger/391412.jpg'])



if __name__=='__main__':
    # download()
    train()
    # eval()