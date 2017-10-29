import skimage.io as io
import scipy.misc
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Load data
batch_size = 1
im_size = 320
out_size = 160
images = np.zeros([0,im_size,im_size,1])
onehot_labels = np.zeros([0,out_size,out_size,2])
for num in range(50):
    num_str = str(num).zfill(2)
    im_stack = io.imread('promise/TrainingData/Case'+num_str+'.mhd', plugin='simpleitk')
    gt_stack = io.imread('promise/TrainingData/Case'+num_str+'_segmentation.mhd', plugin='simpleitk')
    for b in range(im_stack.shape[0]):
        im = scipy.misc.imresize(im_stack[b,:,:],[im_size,im_size])
        gt = scipy.misc.imresize(gt_stack[b,:,:],[out_size,out_size],interp='nearest')
        gt = np.reshape(gt, [1, out_size, out_size,1])
        gt = np.float32(np.concatenate([gt == 0, gt > 0], 3))
        images = np.concatenate([images,np.reshape(im,[1,im_size,im_size,1])])
        onehot_labels = np.concatenate([onehot_labels, gt])
    # This is for saving the volumes as .png images
    #for i in range(im_stack.shape[0]):
    #    im = img[i,:,:]*255
    #    io.imsave('promise/single_seg/Case'+num_str+'_'+str(i).zfill(2)+'.png',im)

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

def batch_norm(x):
    batch_mean, batch_var = tf.nn.moments(x, [0,1,2])
    scale = tf.Variable(tf.ones(batch_mean.shape))
    beta = tf.Variable(tf.zeros(batch_mean.shape))
    return tf.nn.batch_normalization(x, batch_mean, batch_var, beta, scale, 1e-7)
