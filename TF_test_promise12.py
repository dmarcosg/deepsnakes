import skimage.io as io
import scipy.misc
import tensorflow as tf
import numpy as np
import os
from snake_utils import imrotate
import matplotlib.pyplot as plt
import skimage.morphology

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

with tf.device('/gpu:0'):

    #Input and output
    x_ = tf.placeholder(tf.float32, shape=[batch_size,im_size, im_size, 1])
    y_ = tf.placeholder(tf.float32, shape=[batch_size,out_size, out_size, 2])

    #First conv layer
    W_conv1 = weight_variable([3, 3, 1, 16])
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

    # Forth conv layer
    W_conv4 = weight_variable([3, 3, 32, 32])
    b_conv4 = bias_variable([32])
    h_conv4 = tf.nn.relu(conv2d(h_pool3, W_conv4) + b_conv4)
    h_pool4 = batch_norm(max_pool_2x2(h_conv4))

    # Fifth conv layer
    W_conv5 = weight_variable([3, 3, 32, 32])
    b_conv5 = bias_variable([32])
    h_conv5 = tf.nn.relu(conv2d(h_pool4, W_conv5) + b_conv5)
    h_pool5 = batch_norm(max_pool_2x2(h_conv5))

    # Sixth conv layer
    W_conv6 = weight_variable([3, 3, 32, 32])
    b_conv6 = bias_variable([32])
    h_conv6 = tf.nn.relu(conv2d(h_pool5, W_conv6) + b_conv6)
    h_pool6 = batch_norm(max_pool_2x2(h_conv6))

    # Resize and concat
    # resized_out1 = tf.image.resize_images(h_pool1, [im_size, im_size])
    # resized_out2 = tf.image.resize_images(h_pool2, [im_size, im_size])
    resized_out3 = tf.image.resize_images(h_pool3, [out_size, out_size])
    resized_out4 = tf.image.resize_images(h_pool4, [out_size, out_size])
    resized_out5 = tf.image.resize_images(h_pool5, [out_size, out_size])
    resized_out6 = tf.image.resize_images(h_pool6, [out_size, out_size])
    h_concat = tf.concat([resized_out3, resized_out4, resized_out5, resized_out6], 3)

    #Final conv layer
    W_convf = weight_variable([1, 1, int(h_concat.shape[3]), 32])
    b_convf = bias_variable([32])
    h_convf = batch_norm(tf.nn.relu(conv2d(h_concat, W_convf) + b_convf))

    #Predict labels
    W_fc = weight_variable([1, 1, 32, 2])
    b_fc = bias_variable([2])
    pred = conv2d(h_convf, W_fc) + b_fc
    pred = tf.nn.softmax(pred)

    #Loss
    pixel_weights = y_ * [1, 1, 2]
    pixel_weights = tf.reduce_sum(pixel_weights, 3)
    cross_entropy = tf.reduce_mean(
        tf.losses.softmax_cross_entropy(y_, pred, pixel_weights))

#Prepare folder to save network
start_epoch = 0
model_path = 'models/base_promise/'
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

