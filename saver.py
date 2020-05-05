import tensorflow as tf
import numpy as np

if __name__ == '__main__':
    # 保存
    w = tf.Variable([[1, 2, 3], [3, 4, 5]], dtype=tf.float32, name='weights')
    b = tf.Variable([[1, 2, 3]], dtype=tf.float32, name='biases')

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    #没有其他设置的保存
    with tf.Session() as sess:
        sess.run(init)
        save_path = saver.save(sess, 'net200427/200427net.ckpt')
        print("save to path:", save_path)

    # 提取
    # w = tf.Variable(np.arange(6).reshape(2, 3), dtype=tf.float32, name='weights')
    # b = tf.Variable(np.arange(3).reshape(1, 3), dtype=tf.float32, name='biases')
    #
    # saver = tf.train.Saver()
    # with tf.Session() as sess:
    #     saver.restore(sess, 'net200427/200427net.ckpt')
    #     print('weights:', sess.run(w))
    #     print('biases', sess.run(b))
