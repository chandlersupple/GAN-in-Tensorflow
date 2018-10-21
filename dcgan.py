import tensorflow as tf
import numpy as np
from keras.preprocessing import image
sess = tf.InteractiveSession()
        
r_cond = tf.AUTO_REUSE

def batch(batch_size, batch_iter):    
    batch_iarr = []
    for inst in range (batch_iter * batch_size + 1, batch_iter * batch_size + batch_size + 1):
        image_inst = image.load_img(('dog.%s.jpg' %(inst)), target_size = [32, 32])
        inst_arr = image.img_to_array(image_inst)
        batch_iarr.append(inst_arr)
    
    return batch_iarr

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
        
        l_one = tf.layers.dense(z, 3072)
        l_one = tf.layers.batch_normalization(l_one, training= t)
        
        resh_z = tf.reshape(l_one, [-1, 32, 32, 3])
        
        conv_one = cnv_op_t(resh_z, 128, (5, 5), padding= 'same', stride= (1, 1))
        conv_one = tf.layers.batch_normalization(conv_one, training= t)
        conv_one = tf.nn.leaky_relu(conv_one)
        
        conv_two = cnv_op_t(conv_one, 128, (5, 5), padding= 'same', stride= (1, 1))
        conv_two = tf.layers.batch_normalization(conv_two, training= t)
        conv_two = tf.nn.leaky_relu(conv_two)
        
        conv_three = cnv_op_t(conv_one, 128, (5, 5), padding= 'same', stride= (1, 1))
        conv_three = tf.layers.batch_normalization(conv_three, training= t)
        conv_three = tf.nn.leaky_relu(conv_three)
        
        conv_four = cnv_op_t(conv_three, 3, (5, 5), padding= 'same', stride= (1, 1))
        
        resh_conv_four = tf.reshape(conv_four, [-1, 32, 32, 3])
        
    return resh_conv_four
    
def discriminator(x, tr):
    with tf.variable_scope('d', reuse= tf.AUTO_REUSE):
        
        if tr == 1:
            t = True
        else:
            t = False
        
        conv_one = cnv_op(x, 128, (3, 3), padding= 'same')
        conv_one = tf.layers.batch_normalization(conv_one, training= t)
        conv_one = tf.nn.leaky_relu(conv_one)
        
        conv_two = cnv_op(conv_one, 128, (3, 3), padding= 'same')
        conv_two = tf.layers.batch_normalization(conv_two, training= t)
        conv_two = tf.nn.leaky_relu(conv_two)
        
        conv_three = cnv_op(conv_two, 128, (4, 4), padding= 'valid')
        conv_three = tf.layers.batch_normalization(conv_three, training= t)
        conv_three = tf.nn.leaky_relu(conv_three)
        
        conv_four = cnv_op(conv_three, 128, (3, 3), padding= 'valid')
        conv_four = tf.layers.batch_normalization(conv_four, training= t)
        conv_four = tf.nn.leaky_relu(conv_four)
        
        l_one = tf.layers.dense(conv_four, 1)
        
    return l_one

noise_num = 100
epoch = 128
batch_size = 32
batches_in_epoch = 4000 // batch_size

x = tf.placeholder(tf.float32, shape= [None, 32, 32, 3])
z = tf.placeholder(tf.float32, shape= [None, noise_num])
t = tf.placeholder(tf.float32, shape= [])

g = generator(z, t)
dr = discriminator(x, t)
dg = discriminator(g, t)

g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits= dg, labels= tf.zeros_like(dg)))
dr_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits= dr, labels= tf.zeros_like(dr)))
dg_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits= dg, labels= tf.ones_like(dg)))
    
d_loss = dr_loss + dg_loss

t_vars = tf.trainable_variables()
g_vars = [variable for variable in t_vars if 'g' in variable.name]
d_vars = [variable for variable in t_vars if 'd' in variable.name]

g_opt = tf.train.AdamOptimizer(0.01).minimize(g_loss, var_list= g_vars)
d_opt = tf.train.AdamOptimizer(0.01).minimize(d_loss, var_list= d_vars)
    
sess.run(tf.global_variables_initializer())
    
images = []

for epoch_iter in range (epoch):
    for batch_iter in range (batches_in_epoch):
        batch_xy = batch(batch_size, batch_iter)
        sess.run(g_opt, feed_dict= {z: np.random.normal(size= [batch_size, noise_num]), t: 1})
        sess.run(d_opt, feed_dict= {x: batch_xy, z: np.random.normal(size= [batch_size, noise_num]), t: 1})
        
        if batch_iter % 1 == 0:
            dl = sess.run(d_loss, feed_dict= {x: batch_xy, z: np.random.normal(size= [batch_size, noise_num]), t: 0})
            gl = sess.run(g_loss, feed_dict= {z: np.random.normal(size= [batch_size, noise_num]), t: 0})
            print('Epoch: %s, Batch %s / %s, D-loss: %s, G-loss: %s' %(epoch_iter, batch_iter, batches_in_epoch, dl, gl))
            
            g_img = sess.run(g, feed_dict= {z: np.random.normal(size= [1, noise_num]), t: 0})
            g_img = image.array_to_img(np.reshape(g_img, [32, 32, 3]))
            images.append(g_img)
