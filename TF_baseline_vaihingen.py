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



model_path = 'models/base_vai1/'
do_plot = False
do_save_results = True


def weight_variable(shape,wd=0.0):
    initial = tf.truncated_normal(shape, stddev=0.1)
    var = tf.Variable(initial)
    weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
    return var


def gaussian_filter(shape, sigma) :
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
num_ims = 168
batch_size = 1
im_size = 512
out_size = 256
data_path = '/mnt/bighd/Data/Vaihingen/buildings/'
images = np.zeros([num_ims,im_size,im_size,3])
onehot_labels = np.zeros([num_ims,out_size,out_size,3])
building_mask = np.zeros([num_ims,out_size,out_size,1])
for i in range(num_ims):
    this_im  = scipy.misc.imread(data_path+'building_'+str(i+1).zfill(3)+'.tif')
    images[i,:,:,:] = np.float32(this_im)/255
    img_mask = scipy.misc.imread(data_path+'all_buildings_mask_' + str(i+1).zfill(3) + '.tif')/255
    edge = skimage.morphology.binary_dilation(img_mask)-img_mask
    edge = skimage.morphology.binary_dilation(np.float32(edge))
    onehot_labels[i,:,:,0] = scipy.misc.imresize(1-img_mask-edge,[out_size,out_size],interp='nearest')/255
    onehot_labels[i,:,:,1] = scipy.misc.imresize(img_mask,[out_size,out_size],interp='nearest')/255
    onehot_labels[i,:,:,2] = scipy.misc.imresize(edge,[out_size,out_size],interp='nearest')/255

    building_mask[i,:,:,0] = scipy.misc.imresize(scipy.misc.imread(
        data_path + 'building_mask_' + str(i + 1).zfill(3) + '.tif'),[out_size,out_size],interp='nearest') / (255)


layers = 5
with tf.device('/gpu:0'):

    #Input and output
    x_ = tf.placeholder(tf.float32, shape=[batch_size,im_size, im_size, 3])
    y_ = tf.placeholder(tf.float32, shape=[batch_size,out_size, out_size, 3])

    W_conv = []
    b_conv = []
    h_conv = []
    h_pool = []
    resized_out = []
    W_conv.append(weight_variable([3, 3, 3, 32], wd=0.001))
    b_conv.append(bias_variable([32]))
    h_conv.append(tf.nn.relu(conv2d(x_, W_conv[-1]) + b_conv[-1]))
    h_pool.append(batch_norm(max_pool_2x2(h_conv[-1])))
    resized_out.append(tf.image.resize_images(h_pool[-1], [out_size, out_size]))
    for layer in range(1, layers):
        W_conv.append(weight_variable([3, 3, 32, 32], wd=0.001))
        b_conv.append(bias_variable([32]))
        h_conv.append(tf.nn.relu(conv2d(h_pool[-1], W_conv[-1]) + b_conv[-1]))
        h_pool.append(batch_norm(max_pool_2x2(h_conv[-1])))
        resized_out.append(tf.image.resize_images(h_conv[-1], [out_size, out_size]))

    h_concat = tf.concat(resized_out, 3)

    # Final conv layer
    W_convf = weight_variable([1, 1, int(h_concat.shape[3]), 32], wd=0.001)
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
    l2loss = tf.add_n(tf.get_collection('losses'), name='l2_loss')


#Prepare folder to save network
start_epoch = 0
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

#Prepare folder to save results
if do_save_results:
    if not os.path.isdir(model_path+'results/'):
        os.makedirs(model_path+'results/')

#Initialize CNN
optimizer = tf.train.AdamOptimizer(1e-4, epsilon=1e-7).minimize(cross_entropy)


def epoch(i,mode):
    # mode (str): train or test
    batch_ind = np.arange(i,i+batch_size)
    batch = images[batch_ind,:, :, :]
    batch_labels = onehot_labels[batch_ind,:, :, ]
    batch_mask = building_mask[batch_ind, :, :, ]
    if mode is 'train':
        ang = np.random.rand() * 360
        for j in range(len(batch_ind)):
            for b in range(batch.shape[3]):
                batch[j,:, :, b] = imrotate(batch[j,:, :, b], ang)
                batch_labels[j,:, :, b] = imrotate(batch_labels[j,:, :, b], ang, resample='nearest')
            batch_mask[j, :, :, 0] = imrotate(batch_mask[j, :, :, 0], ang, resample='nearest')

    # prediction_np = sess.run(prediction,feed_dict={x:batch})
    tic = time.time()

    #print('%.2f' % (time.time() - tic) + ' s tf inference')
    if mode is 'train':
        _,loss,loss_l2,res = sess.run([optimizer,cross_entropy,l2loss,pred],feed_dict={x_: batch, y_: batch_labels})
        prediction = np.int32(res[:,:,:,1] >= np.amax(res,axis=3))
        if do_plot:
            plt.imshow(res[0,:,:,:])
            plt.show()

    if mode is 'test':
        res = sess.run(pred, feed_dict={x_: batch})
        prediction = np.int32(res[:, :, :, 1] >= np.amax(res, axis=3))
        if do_plot:
            plt.imshow(res[0,:,:,:])
            plt.show()
    g = np.abs(np.linspace(-1, 1, out_size))
    G0, G1 = np.meshgrid(g, g)
    d = (1 - np.sqrt(G0 * G0 + G1 * G1))
    for j in range(len(batch_ind)):
        val = np.max(d * prediction[j, :, :])
        seed_im = np.int32(d * prediction[j, :, :] == val)
        if val > 0:
            prediction[j, :, :] = skimage.morphology.reconstruction(seed_im, prediction[j, :, :])

    intersection = (batch_mask[:,:,:,0]+prediction) == 2
    union = (batch_mask[:,:,:,0] + prediction) >= 1
    iou = np.sum(intersection) / np.sum(union)
    #prediction = scipy.misc.imresize(prediction[0,:,:],[im_size,im_size])
    #scipy.misc.imsave(results_path+'baseline_result_'+str(i).zfill(3)+'.tif',prediction)
    if do_save_results:
        scipy.misc.imsave(model_path + 'results/seg_' + str(i).zfill(3) + '.png', np.uint8(prediction[0, :, :] * 255))

    return iou



with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,log_device_placement=True)) as sess:
    save_path = tf.train.latest_checkpoint(model_path)
    init = tf.global_variables_initializer()
    sess.run(init)
    if save_path is not None:
        saver.restore(sess,save_path)
        start_epoch = int(save_path.split('-')[-1].split('.')[0])+1
    iou_test = []
    iou_train = []
    for n in range(start_epoch,50):
        iou_test = 0
        iou_train = 0
        for i in range(0,100,batch_size):
            #print(i)
            #Do CNN inference
            iou_train += epoch(i,'train')
        iou_train /= 100
        print('Train. Epoch ' + str(n) + '. IoU = %.2f' % (iou_train))
        saver.save(sess,model_path+'model', global_step=n)

        if (n >= 0):
            for i in range(100,168):
                iou_test += epoch(i, 'test')
            iou_test /= 68
            print('Test. Epoch ' + str(n) + '. IoU = %.2f' % (iou_test))











