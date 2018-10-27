# Chandler Supple, October 24, 2018

import tensorflow as tf
import numpy as np
from keras.preprocessing import image
sess = tf.InteractiveSession()
        
def normalize(li, r):
  l = np.array(li) 
  a = np.max(l)
  c = np.min(l)
  b = r[1]
  d = r[0]

  m = (b - d) / (a - c)
  pslope = (m * (l - c)) + d
  return pslope

def ret_data(data_len, batch_size):
    
    data = []
    for imag in range (1, data_len):
        image_inst = image.load_img(('cat.%s.jpg' %(imag)), target_size = [64, 64])
        inst_arr = image.img_to_array(image_inst)
        data.append(inst_arr)
        
    batches = []
    for batch_iter in range (len(data) // batch_size):
        b_iw = data[batch_size * batch_iter: (batch_size * batch_iter) + batch_size]
        resh_b_iw = np.reshape(b_iw, [batch_size, 64, 64, 3])
        batches.append(resh_b_iw)
    
    batches = normalize(batches, [-1, 1])
    
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
        
        l_one = tf.layers.dense(z, 16384)
        l_one = tf.layers.batch_normalization(l_one, training= t)
        
        resh_z = tf.reshape(l_one, [-1, 4, 4, 1024])
        
        conv_one = cnv_op_t(resh_z, 512, (5, 5), padding= 'same')
        conv_one = tf.layers.batch_normalization(conv_one, training= t)
        conv_one = tf.nn.leaky_relu(conv_one)
        
        conv_two = cnv_op_t(conv_one, 256, (5, 5), padding= 'same')
        conv_two = tf.layers.batch_normalization(conv_two, training= t)
        conv_two = tf.nn.leaky_relu(conv_two)
        
        conv_three = cnv_op_t(conv_two, 128, (5, 5), padding= 'same')
        conv_three = tf.layers.batch_normalization(conv_three, training= t)
        conv_three = tf.nn.leaky_relu(conv_three)
        
        conv_four = cnv_op_t(conv_three, 3, (5, 5), padding= 'same')
        act_conv_four = tf.nn.tanh(conv_four)
        
        resh_conv_four = tf.reshape(act_conv_four, [-1, 64, 64, 3])
        
    return resh_conv_four
    
def discriminator(x, tr):
    with tf.variable_scope('d', reuse= tf.AUTO_REUSE):
        
        if tr == 1:
            t = True
        else:
            t = False
        
        conv_one = cnv_op(x, 32, (4, 4), padding= 'same')
        conv_one = tf.layers.batch_normalization(conv_one, training= t)
        conv_one = tf.nn.leaky_relu(conv_one)
        
        conv_two = cnv_op(conv_one, 32, (4, 4), padding= 'same')
        conv_two = tf.layers.batch_normalization(conv_two, training= t)
        conv_two = tf.nn.leaky_relu(conv_two)
        
        conv_three = cnv_op(conv_two, 32, (4, 4), padding= 'same')
        conv_three = tf.nn.softmax(conv_three)
        
        l_one = tf.layers.dense(conv_three, 1)
        
    return l_one

noise_num = 100

x = tf.placeholder(tf.float32, shape= [None, 64, 64, 3])
z = tf.placeholder(tf.float32, shape= [None, noise_num])
t = tf.placeholder(tf.float32, shape= [])

g = generator(z, t)
dr = discriminator(x, t)
dg = discriminator(g, t)

dr_compar = tf.zeros_like(dr) * (0.9)

g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits= dg, labels= tf.zeros_like(dg)))
dr_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits= dr, labels= dr_compar))
dg_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits= dg, labels= tf.ones_like(dg)))
    
d_loss = dr_loss + dg_loss

t_vars = tf.trainable_variables()
g_vars = [variable for variable in t_vars if 'g' in variable.name]
d_vars = [variable for variable in t_vars if 'd' in variable.name]

g_opt = tf.train.AdamOptimizer(0.001).minimize(g_loss, var_list= g_vars)
d_opt = tf.train.AdamOptimizer(0.001).minimize(d_loss, var_list= d_vars)

sess.run(tf.global_variables_initializer())
    
images = []
epoch = 128
batch_size = 32
batches = ret_data(4000, batch_size)
batches_in_epoch = len(batches)

for epoch_iter in range (epoch):
    for batch_iter in range (batches_in_epoch):
        batch_xy = batches[batch_iter]
        sess.run(g_opt, feed_dict= {z: np.random.normal(size= [batch_size, noise_num]), t: 1})
        sess.run(d_opt, feed_dict= {x: batch_xy, z: np.random.normal(size= [batch_size, noise_num]), t: 1})
        
        if batch_iter % 25 == 0:
            dl = sess.run(d_loss, feed_dict= {x: batch_xy, z: np.random.normal(size= [batch_size, noise_num]), t: 0})
            gl = sess.run(g_loss, feed_dict= {z: np.random.normal(size= [batch_size, noise_num]), t: 0})
            print('Epoch: %s, Batch %s / %s, D-loss: %s, G-loss: %s' %(epoch_iter, batch_iter, batches_in_epoch, dl, gl))
            
            g_img = sess.run(g, feed_dict= {z: np.random.normal(size= [1, noise_num]), t: 0})
            g_img = image.array_to_img(255 * normalize(np.reshape(g_img, [64, 64, 3]), [0, 1]))
            images.append(g_img)
