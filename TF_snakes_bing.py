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

from tensorflow.python.client import timeline

def snake_process (mapE, mapA, mapB, mapK, init_snake):
    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()

    for i in range(mapE.shape[3]):
        Du = np.gradient(mapE[:,:,0,i], axis=0)
        Dv = np.gradient(mapE[:,:,0,i], axis=1)
        u = init_snake[:,0:1]
        v = init_snake[:,1:2]
        du = np.zeros(u.shape)
        dv = np.zeros(v.shape)
        snake_hist = []
        snake_hist.append(np.array([u[:, 0], v[:, 0]]).T)
        tic = time.time()
        for j in range(1):
            u, v, du, dv = sess2.run([tf_u, tf_v, tf_du, tf_dv], feed_dict={tf_Du: Du, tf_Dv: Dv,
                                                                               tf_u0: u, tf_v0: v, tf_du0: du, tf_dv0: dv,
                                                                               tf_alpha: mapA[:,:,0,i], tf_beta: mapB[:,:,0,i],
                                                                               tf_kappa: mapK[:,:,0,i]}) #,options=run_options, run_metadata=run_metadata
            snake_hist.append(np.array([u[:, 0], v[:, 0]]).T)
            # Create the Timeline object, and write it to a json
            tl = timeline.Timeline(run_metadata.step_stats)
            ctf = tl.generate_chrome_trace_format()
            with open('timeline.json', 'w') as f:
                f.write(ctf)
        #print('%.2f' % (time.time() - tic) + ' s snake')

    return np.array([u[:,0],v[:,0]]).T,snake_hist





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
L = 20
num_ims = 605
batch_size = 1
im_size = 64
out_size = 64
data_path = '/home/diego/PycharmProjects/snakes_prj/deepsnakes/single_buildings/'
csvfile=open(data_path+'building_coords.csv', newline='')
reader = csv.reader(csvfile)
images = np.zeros([im_size,im_size,3,num_ims])
dists = np.zeros([im_size,im_size,1,num_ims])
masks = np.zeros([im_size,im_size,1,num_ims])
GT = np.zeros([L,2,num_ims])
for i in range(num_ims):
    poly = np.zeros([5, 2])
    corners = reader.__next__()
    for c in range(4):
        poly[c, 1] = np.float(corners[1+2*c])
        poly[c, 0] = np.float(corners[2+2*c])
    poly[4,:] = poly[0,:]
    [tck, u] = interpolate.splprep([poly[:, 0], poly[:, 1]], s=2, k=1, per=1)
    [GT[:,0,i], GT[:,1,i]] = interpolate.splev(np.linspace(0, 1, L), tck)
    this_im  = scipy.misc.imread(data_path+'building_'+str(i)+'.png')
    images[:,:,:,i] = np.float32(this_im)/255
    img_mask = scipy.misc.imread(data_path+'building_mask_' + str(i) + '.png')/65535
    masks[:,:,0,i] = img_mask
    img_dist = scipy.ndimage.morphology.distance_transform_edt(img_mask) + \
               scipy.ndimage.morphology.distance_transform_edt(1 - img_mask)
    img_dist = gaussian(img_dist, 10)
    dists[:,:,0,i] =  img_dist
GT = np.minimum(GT,im_size-1)
GT = np.maximum(GT,0)


with tf.device('/gpu:0'):

    #Input and output
    x = tf.placeholder(tf.float32, shape=[im_size, im_size, 3, batch_size])
    x_image = tf.reshape(x, [-1, im_size, im_size, 3])
    y_ = tf.placeholder(tf.float32, shape=GT[:,:,0].shape)

    #First conv layer
    W_conv1 = weight_variable([3, 3, 3, 32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = batch_norm(max_pool_2x2(h_conv1))


    #Second conv layer
    W_conv2 = weight_variable([3, 3, 32, 32])
    b_conv2 = bias_variable([32])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = batch_norm(max_pool_2x2(h_conv2))

    #Third conv layer
    W_conv3 = weight_variable([3, 3, 32, 32])
    b_conv3 = bias_variable([32])
    h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
    h_pool3 = batch_norm(max_pool_2x2(h_conv3))

    #Resize and concat
    resized_out1 = tf.image.resize_images(h_pool1, [im_size, im_size])
    resized_out2 = tf.image.resize_images(h_pool2, [im_size, im_size])
    resized_out3 = tf.image.resize_images(h_pool3, [out_size, out_size])
    h_concat = tf.concat([resized_out1,resized_out2,resized_out3],3)

    #Final conv layer
    W_convf = weight_variable([1, 1, int(h_concat.shape[3]), 32])
    b_convf = bias_variable([32])
    h_convf = batch_norm(tf.nn.relu(conv2d(h_concat, W_convf) + b_convf))

    #Predict energy
    W_fcE = weight_variable([1, 1, 32, 1])
    b_fcE = bias_variable([1])
    h_fcE = conv2d(h_convf, W_fcE) + b_fcE
    G_filt = gaussian_filter((15,15), 2)
    predE = tf.reshape(conv2d(h_fcE,G_filt), [out_size, out_size, 1, -1])

    #Predict alpha
    W_fcA = weight_variable([1, 1, 32, 1])
    b_fcA = bias_variable([1])
    h_fcA = conv2d(h_convf, W_fcA) + b_fcA
    h_fcA = tf.reduce_mean(h_fcA) + h_fcA*0
    #predA = tf.nn.softplus(tf.reshape(h_fcA,[im_size,im_size,1,-1]))
    predA = tf.reshape(h_fcA,[out_size,out_size,1,-1])
    #Predict beta
    W_fcB = weight_variable([1, 1, 32, 1])
    b_fcB = bias_variable([1])
    h_fcB = conv2d(h_convf, W_fcB) + b_fcB
    predB = tf.reshape(h_fcB,[out_size,out_size,1,-1])
    #Predict kappa
    W_fcK = weight_variable([1, 1, 32, 1])
    b_fcK = bias_variable([1])
    h_fcK = conv2d(h_convf, W_fcK) + b_fcK
    predK = tf.reshape(h_fcK,[out_size,out_size,1,-1])

    #Inject the gradients
    grad_predE = tf.placeholder(tf.float32, shape=[out_size, out_size, 1, batch_size])
    grad_predA = tf.placeholder(tf.float32, shape=[out_size, out_size, 1, batch_size])
    grad_predB = tf.placeholder(tf.float32, shape=[out_size, out_size, 1, batch_size])
    grad_predK = tf.placeholder(tf.float32, shape=[out_size, out_size, 1, batch_size])
    tvars = tf.trainable_variables()
    grads = tf.gradients([predE,predA,predB,predK], tvars, grad_ys = [grad_predE,grad_predA,grad_predB,grad_predK])

#Prepare folder to save network
start_epoch = 0
model_path = 'models/bing1/'
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
optimizer = tf.train.AdamOptimizer(1e-4, epsilon=1e-7)
apply_gradients = optimizer.apply_gradients(zip(grads, tvars))

with tf.device('/cpu:0'):
    tf_alpha = tf.placeholder(tf.float32, shape=[out_size, out_size])
    tf_beta = tf.placeholder(tf.float32, shape=[out_size, out_size])
    tf_kappa = tf.placeholder(tf.float32, shape=[out_size, out_size])
    tf_Du = tf.placeholder(tf.float32, shape=[out_size, out_size])
    tf_Dv = tf.placeholder(tf.float32, shape=[out_size, out_size])
    tf_u0 = tf.placeholder(tf.float32, shape=[L, 1])
    tf_v0 = tf.placeholder(tf.float32, shape=[L, 1])
    tf_du0 = tf.placeholder(tf.float32, shape=[L, 1])
    tf_dv0 = tf.placeholder(tf.float32, shape=[L, 1])
    gamma = tf.constant(1, dtype=tf.float32)
    max_px_move = tf.constant(1, dtype=tf.float32)
    delta_s = tf.constant(1, dtype=tf.float32)

    tf_u = tf_u0
    tf_du = tf_du0
    tf_v = tf_v0
    tf_dv = tf_dv0

    for i in range(100):
        tf_u, tf_v, tf_du, tf_dv = active_contour_step(tf_Du, tf_Dv, tf_du, tf_dv, tf_u, tf_v,
                                                       tf_alpha, tf_beta, tf_kappa,
                                                       gamma, max_px_move, delta_s)


def epoch(i,mode):
    # mode (str): train or test
    batch_ind = [i]
    batch = images[:, :, :, batch_ind]
    batch_mask = masks[:, :, :, batch_ind]
    ang = np.random.rand() * 360
    for j in range(len(batch_ind)):
        for b in range(batch.shape[2]):
            batch[:, :, b, j] = imrotate(batch[:, :, b, j], ang)
        batch_mask[:, :, 0, j] = imrotate(batch_mask[:, :, 0, j], ang, resample='nearest')
    # prediction_np = sess.run(prediction,feed_dict={x:batch})
    tic = time.time()
    [mapE, mapA, mapB, mapK] = sess.run([predE, predA, predB, predK], feed_dict={x: batch})
    #print('%.2f' % (time.time() - tic) + ' s tf inference')
    if mode is 'train':
        for j in range(mapK.shape[3]):
            mapK[:, :, 0, j] -= batch_mask[:, :, 0, j] * 0.5 - 0.5 / 2
        # mapE_aug[:,:,0,j] = mapE[:,:,0,j]+np.maximum(0,20-batch_dists[:,:,0,j])*max_val/50
    # Do snake inference
    s = np.linspace(0, 2 * np.pi, L)
    init_u = out_size / 2 + 5 * np.cos(s)
    init_v = out_size / 2 + 5 * np.sin(s)
    init_u = init_u.reshape([L, 1])
    init_v = init_v.reshape([L, 1])
    init_snake = np.array([init_u[:, 0], init_v[:, 0]]).T
    snake, snake_hist = snake_process(mapE, 0.1+0*np.maximum(0, mapA), np.maximum(0, mapB), mapK, init_snake)
    # Get last layer gradients
    M = mapE.shape[0]
    N = mapE.shape[1]
    der1, der2 = derivatives_poly(snake)
    thisGT = GT[:, :, batch_ind[0]] - out_size / 2
    R = [[np.cos(ang * np.pi / 180), np.sin(ang * np.pi / 180)],
         [-np.sin(ang * np.pi / 180), np.cos(ang * np.pi / 180)]]
    thisGT = np.matmul(thisGT, R)
    thisGT += out_size / 2
    der1_GT, der2_GT = derivatives_poly(thisGT)

    grads_arrayE = mapE * 0.05
    grads_arrayA = mapA * 0.5
    grads_arrayB = mapB * 0.05
    grads_arrayK = mapK * 0.05
    grads_arrayE[:, :, 0, 0] -= draw_poly(snake, 1, [M, N],2) - draw_poly(thisGT, 1, [M, N],2)
    grads_arrayA[:, :, 0, 0] -= (np.mean(der1) - np.mean(der1_GT))
    grads_arrayB[:, :, 0, 0] -= (draw_poly(snake, der2, [M, N],2) - draw_poly(thisGT, der2_GT, [M, N],2))
    mask_gt = draw_poly_fill(thisGT, [M, N])
    mask_snake = draw_poly_fill(snake, [M, N])
    grads_arrayK[:, :, 0, 0] -= mask_gt - mask_snake

    intersection = (mask_gt+mask_snake) == 2
    union = (mask_gt + mask_snake) >= 1
    iou = np.sum(intersection) / np.sum(union)
    if mode is 'train':
        tic = time.time()
        apply_gradients.run(
            feed_dict={x: batch, grad_predE: grads_arrayE, grad_predA: grads_arrayA, grad_predB: grads_arrayB,
                       grad_predK: grads_arrayK})
        #print('%.2f' % (time.time() - tic) + ' s apply gradients')
        #print('IoU = %.2f' % (iou))
        iou_train[len(iou_train)-1] += iou
    if mode is 'test':
        #print('IoU = %.2f' % (iou))
        #plot_snakes(snake, snake_hist, thisGT, mapE, np.maximum(mapA, 0), np.maximum(mapB, 0), mapK, \
        #            grads_arrayE, grads_arrayA, grads_arrayB, grads_arrayK, batch, batch_mask)
        #plt.show()
        iou_test[len(iou_test)-1] += iou



with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,log_device_placement=True)) as sess:
    sess2 = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,log_device_placement=True))
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
        for i in range(400):
            #print(i)
            #Do CNN inference
            epoch(i,'train')
        iou_train[len(iou_train)-1] /= 400
        print('Train. Epoch ' + str(n) + '. IoU = %.2f' % (iou_train[len(iou_train)-1]))
        saver.save(sess,model_path+'model', global_step=n)

        if (n >= 0):
            for i in range(400,605):
                epoch(i, 'test')
            iou_test[len(iou_test)-1] /= 205
            print('Test. Epoch ' + str(n) + '. IoU = %.2f' % (iou_test[len(iou_test)-1]))






