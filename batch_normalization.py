import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

ACTIVATION = tf.nn.relu
N_LAYERS = 7
N_HIDDEN_UNITS = 30


def fix_seed(seed=1):
    np.random.seed(seed)
    tf.set_random_seed(seed)


def plot_his(inputs, inputs_norm):
    # j=0对应inputs，j=1对应inputs_norm
    for j, all_inputs in enumerate([inputs, inputs_norm]):
        for i, input in enumerate(all_inputs):
            plt.subplot(2, len(all_inputs), j * len(all_inputs) + (i + 1))
            plt.cla()
            if i == 0:
                the_range = (-7, 10)
            else:
                the_range = (-1, 1)
            plt.hist(input.ravel(), bins=15, range=the_range, color='#FF5733')
            plt.yticks(())
            if j == 1:
                plt.xticks(the_range)
            else:
                plt.xticks(())
            ax = plt.gca()
            ax.spines['right'].set_color('none')
            ax.spines['top'].set_color('none')
        plt.title("%s normalizing" % ("without" if j == 0 else "with"))
        plt.draw()
        plt.pause(0.01)


def built_net(xs, ys, norm):
    # 比如inputs是100个大小为64*1的数据，in_size=64,输出比如说要求为50
    # 那么weights是64*50的，biases是1*50的，相加是每一行的weights都加上biases
    def add_layer(inputs, in_size, out_size, activation_function=None, norm=False):
        weights = tf.Variable(tf.random_normal([in_size, out_size]))
        biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)

        wx_plus_b = tf.matmul(inputs, weights) + biases

        if norm:
            fc_mean, fc_var = tf.nn.moments(wx_plus_b, axes=[0])# axes=[0]表示按列计算
            scale = tf.Variable(tf.ones([out_size]))
            shift = tf.Variable(tf.zeros([out_size]))
            epsilon = 0.001

            ema = tf.train.ExponentialMovingAverage(decay=0.5)

            def mean_var_with_update():
                ema_apply_op = ema.apply([fc_mean, fc_var])
                with tf.control_dependencies([ema_apply_op]):
                    return tf.identity(fc_mean), tf.identity(fc_var)

            # mean,var = mean_var_with_update()
            on_train_bool=tf.constant(on_train,tf.bool)
            mean, var = tf.cond(on_train_bool, mean_var_with_update, lambda: (ema.average(fc_mean), ema.average(fc_var)))

            # wx_plus_b = tf.nn.batch_normalization(wx_plus_b, fc_mean, fc_var, shift, scale, epsilon)
            wx_plus_b = tf.nn.batch_normalization(wx_plus_b, mean, var, shift, scale, epsilon)

        if activation_function is None:
            outputs = wx_plus_b
        else:
            outputs = activation_function(wx_plus_b)

        return outputs

    fix_seed(1)

    if norm:
        fc_mean, fc_var = tf.nn.moments(xs, axes=[0])
        scale = tf.Variable(tf.ones([1])) #一开始的输入是64*1，列数是1
        shift = tf.Variable(tf.zeros([1]))
        epsilon = 0.001

        ema = tf.train.ExponentialMovingAverage(decay=0.5)
        def mean_var_with_update():
            ema_apply_op = ema.apply([fc_mean, fc_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(fc_mean), tf.identity(fc_var)

        # mean,var = mean_var_with_update()
        on_train_bool = tf.constant(on_train, tf.bool)
        mean, var = tf.cond(on_train_bool, mean_var_with_update, lambda: (ema.average(fc_mean), ema.average(fc_var)))
        xs = tf.nn.batch_normalization(xs, fc_mean, fc_var, shift, scale, epsilon)

    layer_inputs = [xs]

    for l_n in range(N_LAYERS):
        layer_input = layer_inputs[l_n]
        in_size = layer_inputs[l_n].get_shape()[1].value  # 上一层的输出[in_size,out_size]中的out_size作为这一层的输入，value是get_shape()[1]的大小

        output = add_layer(layer_input, in_size, N_HIDDEN_UNITS, ACTIVATION, norm)
        layer_inputs.append(output)

    prediction = add_layer(layer_inputs[-1], 30, 1, activation_function=None)
    cost = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1]))
    train_op = tf.train.GradientDescentOptimizer(0.001).minimize(cost)
    return [train_op, cost, layer_inputs]

fix_seed(1)
x_data = np.linspace(-7, 10, 2500)[:, np.newaxis]
np.random.shuffle(x_data)
noise = np.random.normal(0, 8, x_data.shape)
y_data = np.square(x_data) - 5 + noise

on_train = True

plt.scatter(x_data, y_data)
plt.show()

xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])

train_op, cost, layers_inputs = built_net(xs, ys, norm=False)
train_op_norm, cost_norm, layers_inputs_norm = built_net(xs, ys, norm=True)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

cost_his = []
cost_his_norm = []
record_step = 5

plt.ion()
plt.figure(figsize=(7, 3))
for i in range(250):
    if i % 50 == 0:
        all_inputs, all_inputs_norm = sess.run([layers_inputs, layers_inputs_norm], feed_dict={xs: x_data, ys: y_data})
        plot_his(all_inputs, all_inputs_norm)

    # sess.run(train_op, feed_dict={xs: x_data, ys: y_data})
    # sess.run(train_op_norm, feed_dict={xs: x_data, ys: y_data})
    sess.run([train_op,train_op_norm], feed_dict={xs: x_data[i*10:i*10+10], ys: y_data[i*10:i*10+10]})

    if i % record_step == 0:
        cost_his.append(sess.run(cost, feed_dict={xs: x_data, ys: y_data}))
        cost_his_norm.append(sess.run(cost_norm, feed_dict={xs: x_data, ys: y_data}))

plt.ioff()
plt.figure()
plt.plot(np.arange(len(cost_his))*record_step, np.array(cost_his), label='no BN')
plt.plot(np.arange(len(cost_his))*record_step, np.array(cost_his_norm), label='BN')
plt.legend()
plt.show()




