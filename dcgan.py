# Chandler Supple, October 24, 2018

import numpy as np
import tensorflow as tf
from keras.datasets import cifar10
from keras.preprocessing import image
sess = tf.InteractiveSession()

def ret_data(batch_size, noise, class_num= False):
    data = []
    (cf_data, cf_labels), _ = cifar10.load_data()
    
    if class_num:
        data = []
        for imag in range (len(cf_labels)):
            if cf_labels[imag] == class_num:
                data.append(cf_data[imag])
    else:
        data = cf_data
    
    batches = []
    for batch_iter in range (len(data) // batch_size):
        b_iw = data[(batch_iter * batch_size): (batch_iter * batch_size) + batch_size]
        resh_b_iw = np.reshape(b_iw, [batch_size, 32, 32, 3])
        batches.append(resh_b_iw)
        
    batches = batches + (noise * np.random.normal(size= np.shape(batches)))
        
    return batches
                    
def cnv_op(x, f, kernel, padding, stride= (2, 2)):
    o = tf.layers.conv2d(inputs= x, filters= f, kernel_size= kernel, strides= stride, padding= padding)
    
    return o 

def cnv_op_t(x, f, kernel, padding, stride= (2, 2)):
    o = tf.layers.conv2d_transpose(inputs= x, filters= f, kernel_size= kernel, strides= stride, padding= padding)
    
    return o 

def generator(z, tr):
    with tf.variable_scope('g', reuse= tf.AUTO_REUSE):
        
        if tr == 1:
            t = True
        else:
            t = False
        
        l_one = tf.layers.dense(z, 8192)
        l_one = tf.layers.batch_normalization(l_one, training= t)
        
        resh_z = tf.reshape(l_one, [-1, 4, 4, 512])
        
        conv_one = cnv_op_t(resh_z, 256, (5, 5), padding= 'same')
        conv_one = tf.layers.batch_normalization(conv_one, training= t)
        conv_one = tf.nn.leaky_relu(conv_one)
        
        conv_two = cnv_op_t(conv_one, 128, (5, 5), padding= 'same')
        conv_two = tf.layers.batch_normalization(conv_two, training= t)
        conv_two = tf.nn.leaky_relu(conv_two)
        
        conv_three = cnv_op_t(conv_two, 64, (5, 5), stride= (1, 1), padding= 'same')
        conv_three = tf.layers.batch_normalization(conv_three, training= t)
        conv_three = tf.nn.leaky_relu(conv_three)
        
        conv_four = cnv_op_t(conv_three, 32, (5, 5), stride= (1, 1), padding= 'same')
        conv_four = tf.layers.batch_normalization(conv_four, training= t)
        conv_four = tf.nn.leaky_relu(conv_four)
        
        conv_five = cnv_op_t(conv_four, 3, (5, 5), padding= 'same')
        act_conv_five = tf.nn.tanh(conv_five)
        resh_conv_five = tf.reshape(act_conv_five, [-1, 32, 32, 3])
        
    return resh_conv_five
    
def discriminator(x, tr):
    with tf.variable_scope('d', reuse= tf.AUTO_REUSE):
        
        if tr == 1:
            t = True
        else:
            t = False
        
        conv_one = cnv_op(x, 256, (4, 4), padding= 'same')
        conv_one = tf.layers.batch_normalization(conv_one, training= t)
        conv_one = tf.nn.leaky_relu(conv_one)
        
        conv_two = cnv_op(conv_one, 128, (4, 4), padding= 'same')
        conv_two = tf.layers.batch_normalization(conv_two, training= t)
        conv_two = tf.nn.leaky_relu(conv_two)
        
        conv_three = cnv_op(conv_two, 64, (4, 4), padding= 'same')
        conv_three = tf.layers.batch_normalization(conv_three, training= t)
        conv_three = tf.nn.leaky_relu(conv_three)
        
        l_one = tf.layers.dense(conv_three, 1)
        
    return l_one

noise_num = 100

x = tf.placeholder(tf.float32, shape= [None, 32, 32, 3])
z = tf.placeholder(tf.float32, shape= [None, noise_num])
t = tf.placeholder(tf.float32, shape= [])

g = generator(z, t)
dr = discriminator(x, t)
dg = discriminator(g, t)


g_loss = -1 * tf.reduce_mean(dg)
dr_loss = tf.reduce_mean(dr)
dg_loss = tf.reduce_mean(dg)
    
d_loss = dg_loss - dr_loss

t_vars = tf.trainable_variables()
g_vars = [variable for variable in t_vars if 'g' in variable.name]
d_vars = [variable for variable in t_vars if 'd' in variable.name]

g_opt = tf.train.RMSPropOptimizer(2e-4).minimize(g_loss, var_list= g_vars)
d_opt = tf.train.RMSPropOptimizer(2e-4).minimize(d_loss, var_list= d_vars)

d_clip = [v.assign(tf.clip_by_value(v, -0.01, 0.01)) for v in d_vars]

sess.run(tf.global_variables_initializer())
    
g_iters = 1
d_iters = 5
images = []
epoch = 128
batch_size = 32
batches = ret_data(batch_size, 2)
batches_in_epoch = len(batches)

for epoch_iter in range (epoch):
    for batch_iter in range (batches_in_epoch):
        batch_xy = batches[batch_iter]
        for g_iter in range (g_iters):
            sess.run(g_opt, feed_dict= {z: np.random.uniform(-1.0, 1.0, size= [batch_size, noise_num]), t: 1})
        for d_iter in range (d_iters):
            sess.run(d_opt, feed_dict= {x: batch_xy, z: np.random.uniform(-1.0, 1.0, size= [batch_size, noise_num]), t: 1})
            sess.run(d_clip)
        
        if batch_iter % 25 == 0:
            dl = sess.run(d_loss, feed_dict= {x: batch_xy, z: np.random.uniform(-1.0, 1.0, size= [batch_size, noise_num]), t: 0})
            gl = sess.run(g_loss, feed_dict= {z: np.random.uniform(-1.0, 1.0, size= [batch_size, noise_num]), t: 0})
            print('Epoch: %s, Batch %s / %s, D-loss: %s, G-loss: %s' %(epoch_iter, batch_iter, batches_in_epoch, dl, gl))
            
            g_img = sess.run(g, feed_dict= {z: np.random.uniform(-1.0, 1.0, size= [1, noise_num]), t: 0})
            g_img = image.array_to_img(np.reshape(g_img, [32, 32, 3]))
            images.append(g_img)
