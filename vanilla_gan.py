# Chandler Supple, 10-20-18

# Libraries
import tensorflow as tf
import numpy as np
from keras.preprocessing import image
sess = tf.InteractiveSession()
      
def batch(batch_size, batch_iter): # return image batches    
    batch_iarr = []
    for inst in range (batch_iter * batch_size + 1, batch_iter * batch_size + batch_size + 1):
        image_inst = image.load_img(('dog.%s.jpg' %(inst)), color_mode= 'grayscale', target_size = [64, 64])
        inst_arr = image.img_to_array(image_inst)
        batch_iarr.append(inst_arr)
    
    return batch_iarr

def generator(z, kp): # generates images from a latent variable
    with tf.variable_scope('g', reuse= tf.AUTO_REUSE):
        
        l_one = tf.layers.dense(z, 128)
        l_one = tf.nn.dropout(l_one, keep_prob= kp)
        l_one = tf.nn.leaky_relu(l_one)
        
        l_two = tf.layers.dense(l_one, 256)
        l_two = tf.nn.dropout(l_two, keep_prob= kp)
        l_two = tf.nn.leaky_relu(l_two)
        
        l_three = tf.layers.dense(l_two, 512)
        l_three = tf.nn.dropout(l_three, keep_prob= kp)
        l_three = tf.nn.leaky_relu(l_three)
        
        l_four = tf.layers.dense(l_three, 1024)
        l_four = tf.nn.dropout(l_four, keep_prob= kp)
        l_four = tf.nn.leaky_relu(l_four)
        
        l_five = tf.layers.dense(l_four, 2048)
        l_five = tf.nn.dropout(l_five, keep_prob= kp)
        l_five = tf.nn.leaky_relu(l_five)
        
        l_six = tf.layers.dense(l_five, 4096)
    
    return l_six
    
def discriminator(x, kp): # determines whether an image is real or fake
    with tf.variable_scope('d', reuse= tf.AUTO_REUSE):
        
        l_one = tf.layers.dense(x, 2048)
        l_one = tf.nn.dropout(l_one, keep_prob= kp)
        l_one = tf.nn.leaky_relu(l_one)
        
        l_two = tf.layers.dense(l_one, 512)
        l_two = tf.nn.dropout(l_two, keep_prob= kp)
        l_two = tf.nn.leaky_relu(l_two)
        
        l_three = tf.layers.dense(l_two, 128)
        l_three = tf.nn.dropout(l_three, keep_prob= kp)
        l_three = tf.nn.leaky_relu(l_three)
        
        l_four = tf.layers.dense(l_three, 32)
        l_four = tf.nn.dropout(l_four, keep_prob= kp)
        l_four = tf.nn.leaky_relu(l_four)
        
        l_five = tf.layers.dense(l_four, 2)
        l_five = tf.nn.dropout(l_five, keep_prob= kp)
        l_five = tf.nn.leaky_relu(l_five)
        
        l_six = tf.layers.dense(l_five, 1)
    
    return l_six

x = tf.placeholder(tf.float32, shape= [None, 4096])
z = tf.placeholder(tf.float32, shape= [None, 64])
kp = tf.placeholder_with_default(0.5, [])

g = generator(z, kp)
dr = discriminator(x, kp)
dg = discriminator(g, kp)

# Loss
g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits= dg, labels= tf.ones_like(dg)))
dr_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits= dr, labels= tf.ones_like(dr)))
dg_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits= dg, labels= tf.zeros_like(dg)))
    
d_loss = dr_loss + dg_loss

t_vars = tf.trainable_variables()
g_vars = [variable for variable in t_vars if 'g' in variable.name]
d_vars = [variable for variable in t_vars if 'd' in variable.name]

# Optimizers
g_opt = tf.train.AdamOptimizer(0.01).minimize(g_loss, var_list= g_vars)
d_opt = tf.train.AdamOptimizer(0.01).minimize(d_loss, var_list= d_vars)
    
sess.run(tf.global_variables_initializer())
    
epoch = 128
batch_size = 32
batches_in_epoch = 4000 // batch_size
    
images = []

# Training
for epoch_iter in range (epoch):
    for batch_iter in range (batches_in_epoch):
        batch_xy = np.reshape(batch(batch_size, batch_iter), [batch_size, 4096])
        sess.run(g_opt, feed_dict= {z: np.random.normal(size= [batch_size, 64]), kp: 0.5})
        sess.run(d_opt, feed_dict= {x: batch_xy, z: np.random.normal(size= [batch_size, 64]), kp: 0.5})
        
        if batch_iter % 15 == 0:
            dl = sess.run(d_loss, feed_dict= {x: batch_xy, z: np.random.normal(size= [batch_size, 64]), kp: 1.0})
            gl = sess.run(g_loss, feed_dict= {z: np.random.normal(size= [batch_size, 64]), kp: 1.0})
            print('Epoch: %s, Batch %s / %s, D-loss: %s, G-loss: %s' %(epoch_iter, batch_iter, batches_in_epoch, dl, gl))
        
        if batch_iter % 100 == 0:
            g_img = sess.run(g, feed_dict= {z: np.random.normal(size= [1, 64]), kp: 1.0})
            images.append(g_img)
