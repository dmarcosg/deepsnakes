import tensorflow as tf
import scipy.misc
import numpy as np
import csv
import os
import matplotlib.pyplot as plt
from active_contour_maps_GD_fast import draw_poly,derivatives_poly,draw_poly_fill
from snake_inference_fast_TF import active_contour_step
from snake_utils import imrotate, plot_snakes
from scipy import interpolate
from skimage.filters import gaussian
import scipy
import time
import skimage.morphology





def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def gaussian_filter(shape,sigma):
    x, y = [int(np.floor(edge / 2)) for edge in shape]
    grid = np.array([[((i ** 2 + j ** 2) / (2.0 * sigma ** 2)) for i in range(-x, x + 1)] for j in range(-y, y + 1)])
    filt = np.exp(-grid) / (2 * np.pi * sigma ** 2)
    filt /= np.sum(filt)
    var = np.zeros((shape[0],shape[1],1,1))
    var[:,:,0,0] = filt
    return tf.constant(np.float32(var))

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

#Load data
num_ims = 400
batch_size = 1
im_size = 64
out_size = 64
data_path = 'single_buildings/'
images = np.zeros([num_ims,im_size,im_size,3])
onehot_labels = np.zeros([num_ims,out_size,out_size,3])
for i in range(num_ims):
    this_im  = scipy.misc.imread(data_path+'building_'+str(i+1)+'.png')
    images[i,:,:,:] = np.float32(this_im)/255
    img_mask = scipy.misc.imread(data_path+'building_mask_' + str(i+1) + '.png')/255
    img_mask /= 257
    edge = skimage.morphology.binary_dilation(img_mask)-img_mask
    edge = np.float32(edge)
    onehot_labels[i,:,:,0] = scipy.misc.imresize(1-img_mask-edge,[out_size,out_size],interp='nearest')/255
    onehot_labels[i,:,:,1] = scipy.misc.imresize(img_mask,[out_size,out_size],interp='nearest')/255
    onehot_labels[i,:,:,2] = scipy.misc.imresize(edge,[out_size,out_size],interp='nearest')/255


with tf.device('/gpu:0'):

    #Input and output
    x_ = tf.placeholder(tf.float32, shape=[batch_size,im_size, im_size, 3])
    y_ = tf.placeholder(tf.float32, shape=[batch_size,im_size, im_size, 3])

    #First conv layer
    W_conv1 = weight_variable([3, 3, 3, 16])
    b_conv1 = bias_variable([16])
    h_conv1 = tf.nn.relu(conv2d(x_, W_conv1) + b_conv1)
    h_pool1 = batch_norm(max_pool_2x2(h_conv1))


    #Second conv layer
    W_conv2 = weight_variable([3, 3, 16, 32])
    b_conv2 = bias_variable([32])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = batch_norm(max_pool_2x2(h_conv2))

    #Third conv layer
    W_conv3 = weight_variable([3, 3, 32, 32])
    b_conv3 = bias_variable([32])
    h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
    h_pool3 = batch_norm(max_pool_2x2(h_conv3))


    #Resize and concat
    resized_out1 = tf.image.resize_images(h_pool1, [out_size, out_size])
    resized_out2 = tf.image.resize_images(h_pool2, [out_size, out_size])
    resized_out3 = tf.image.resize_images(h_pool3, [out_size, out_size])
    h_concat = tf.concat([resized_out1,resized_out2,resized_out3],3)

    #Final conv layer
    W_convf = weight_variable([1, 1, int(h_concat.shape[3]), 32])
    b_convf = bias_variable([32])
    h_convf = batch_norm(tf.nn.relu(conv2d(h_concat, W_convf) + b_convf))

    #Predict labels
    W_fc = weight_variable([1, 1, 32, 3])
    b_fc = bias_variable([3])
    pred = conv2d(h_convf, W_fc) + b_fc
    pred = tf.nn.softmax(pred)

    #Loss
    pixel_weights = y_ * [1, 1, 3]
    pixel_weights = tf.reduce_sum(pixel_weights, 3)
    cross_entropy = tf.reduce_mean(
        tf.losses.softmax_cross_entropy(y_, pred, pixel_weights))

#Prepare folder to save network
start_epoch = 0
model_path = 'models/base_bing1/'
if not os.path.isdir(model_path):
    os.makedirs(model_path)
else:
    modelnames = []
    modelnames += [each for each in os.listdir(model_path) if each.endswith('.net')]
    epoch = -1
    for s in modelnames:
        epoch = max(int(s.split('-')[-1].split('.')[0]),epoch)
    start_epoch = epoch + 1

# Add ops to save and restore all the variables.
saver = tf.train.Saver()

#Initialize CNN
optimizer = tf.train.AdamOptimizer(1e-4, epsilon=1e-7).minimize(cross_entropy)


def epoch(i,mode):
    # mode (str): train or test
    batch_ind = np.arange(i,i+batch_size)
    batch = images[batch_ind,:, :, :]
    batch_labels = onehot_labels[batch_ind,:, :, ]
    if mode is 'train':
        ang = np.random.rand() * 360
        for j in range(len(batch_ind)):
            for b in range(batch.shape[3]):
                batch[j,:, :, b] = imrotate(batch[j,:, :, b], ang)
                batch_labels[j,:, :, b] = imrotate(batch_labels[j,:, :, b], ang, resample='nearest')

    # prediction_np = sess.run(prediction,feed_dict={x:batch})
    tic = time.time()

    #print('%.2f' % (time.time() - tic) + ' s tf inference')
    if mode is 'train':
        _,loss,res = sess.run([optimizer,cross_entropy,pred],feed_dict={x_: batch, y_: batch_labels})
        prediction = np.int32(res[:,:,:,1] >= np.amax(res,axis=3))

    if mode is 'test':
        res = sess.run(pred, feed_dict={x_: batch})
        prediction = np.int32(res[:, :, :, 1] >= np.amax(res, axis=3))
        seed_im = np.zeros((out_size,out_size))
        g = np.abs(np.linspace(-1, 1, out_size))
        G0, G1 = np.meshgrid(g, g)
        d = (1-np.sqrt(G0*G0 + G1*G1))
        for j in range(len(batch_ind)):
            val = np.max(d*prediction[j,:,:])
            seed_im = np.int32(d*prediction[j,:,:] == val)
            prediction[j,:,:] = skimage.morphology.reconstruction(seed_im,prediction[j,:,:])
        #plt.imshow(res[0,:,:,:])
        #plt.show()

    intersection = (batch_labels[:,:,:,1]+prediction) == 2
    union = (batch_labels[:,:,:,1] + prediction) >= 1
    iou = np.sum(intersection) / np.sum(union)
    if mode is 'train':
        iou_train[len(iou_train)-1] += iou
    if mode is 'test':
        iou_test[len(iou_test)-1] += iou



with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,log_device_placement=True)) as sess:
    save_path = tf.train.latest_checkpoint(model_path)
    init = tf.global_variables_initializer()
    sess.run(init)
    if save_path is not None:
        saver.restore(sess,save_path)
        start_epoch = int(save_path.split('-')[-1].split('.')[0])+1
    iou_test = []
    iou_train = []
    for n in range(start_epoch,150):
        iou_test.append(0)
        iou_train.append(0)
        for i in range(0,300,batch_size):
            #print(i)
            #Do CNN inference
            epoch(i,'train')
        iou_train[len(iou_train)-1] /= 300
        print('Train. Epoch ' + str(n) + '. IoU = %.2f' % (iou_train[len(iou_train)-1]))
        saver.save(sess,model_path+'model', global_step=n)

        if (n >= 0):
            for i in range(300,400):
                epoch(i, 'test')
            iou_test[len(iou_test)-1] /= 100
            print('Test. Epoch ' + str(n) + '. IoU = %.2f' % (iou_test[len(iou_test)-1]))











