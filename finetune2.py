import tensorflow as tf
import vgg
import cv2
import numpy as np


# 1.查看模型参数
def look_model_params():
    graph=tf.Graph
    inputs=tf.placeholder(dtype=tf.float32,shape=[None,224,224,3],name='inputs')
    net,end_points=vgg.vgg_16(inputs,num_classes=1000)

    saver=tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess,'models/vgg_16.ckpt')
        tvs=[v for v in tf.trainable_variables()]
        print('获得所有可训练变量的权重:')
        for v in tvs:
            print(v.name)
            print(sess.run(v))
        gv=[v for v in tf.global_variables()]
        print('获得所有变量:')
        for v in gv:
            print(v.name,'\n')
        ops=[o for o in sess.graph.get_operations()]
        print('获得所有operations相关的tensor:')
        for o in ops:
            print(o.name,'\n')

#直接使用原始模型进行测试
def test_with_origin():
    image=cv2.imread('./imgdata/cat.18.jpg')
    print(image.shape)
    res=cv2.resize(image,(224,224))
    res_image=np.expand_dims(res,0)
    print(res_image.shape,type(res_image))

    graph = tf.Graph
    inputs = tf.placeholder(dtype=tf.float32, shape=[None, 224, 224, 3], name='inputs')
    net, end_points = vgg.vgg_16(inputs, num_classes=1000)
    print(end_points)
    saver=tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess,'./models/vgg_16.ckpt')
        input=sess.graph.get_tensor_by_name('inputs:0')
        output=sess.graph.get_tensor_by_name('vgg_16/fc8/squeezed:0')
        pred=sess.run(output,feed_dict={input:res_image})
        print(np.argmax(pred,1))

def extend_origin_model():
    image=cv2.imread('./imgdata/cat.18.jpg')
    print(image.shape)
    res=cv2.resize(image,(224,224))
    res_image=np.expand_dims(res,0)
    print(res_image.shape,type(res_image))

    graph = tf.Graph
    inputs = tf.placeholder(dtype=tf.float32, shape=[None, 224, 224, 3], name='inputs')
    net, end_points = vgg.vgg_16(inputs, num_classes=1000)
    print(end_points)
    saver=tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess,'./models/vgg_16.ckpt')
        input=sess.graph.get_tensor_by_name('inputs:0')
        output=sess.graph.get_tensor_by_name('vgg_16/fc8/squeezed:0')
        pred=tf.argmax(output,1)
        pred=sess.run(pred,feed_dict={input:res_image})
        print(pred)


#微调
def finetune():
    image=cv2.imread('./imgdata/cat.18.jpg')
    print(image.shape)
    res_image=cv2.resize(image,(224,224),interpolation=cv2.INTER_CUBIC)
    print(res_image.shape)
    res_image=np.expand_dims(res_image,axis=0)
    print(res_image.shape)
    labels=[[1,0]]

    graph=tf.get_default_graph()

    input=tf.placeholder(dtype=tf.float32,shape=[None,224,224,3],name='inputs')
    y_=tf.placeholder(dtype=tf.float32,shape=[None,2],name='labels')

    net,end_points=vgg.vgg_16(input,num_classes=2)
    print(net,end_points)

    y=tf.nn.softmax(net)
    cross_entropy=tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y),reduction_indices=[1]))
    output_vars=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='vgg_16/fc8')
    print(output_vars)

    train_op=tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy,var_list=output_vars)

    var=tf.global_variables()
    print(var)

    var_to_restore=[val for val in var if 'fc8' not in val.name]
    print(var_to_restore)

    saver=tf.train.Saver(var_to_restore)
    with tf.Session() as sess:
        saver.restore(sess,'./models/vgg_16.ckpt')
        var_to_init=[val for val in var if 'fc8' in val.name]
        sess.run(tf.variables_initializer(var_to_init))

        w1=sess.graph.get_tensor_by_name('vgg_16/conv1/conv1_1/weights:0')
        print(sess.run(w1,feed_dict={input:res_image}))

        w8=sess.graph.get_tensor_by_name('vgg_16/fc8/weights:0')
        print('w8',sess.run(w8,feed_dict={input:res_image}))

        sess.run(train_op,feed_dict={input:res_image,y_:labels})

if __name__=='__main__':
    #look_model_params()
    #test_with_origin()
    #extend_origin_model()
    finetune()