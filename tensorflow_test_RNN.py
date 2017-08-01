import tensorflow as tf
import scipy.misc
import numpy as np
import csv
import matplotlib.pyplot as plt
from active_contour_maps_tensorflow import active_contour_step
from scipy import interpolate
from skimage.filters import gaussian
import scipy


def plot_snakes(u,v,GT,mapDu, mapDv, mapA, mapB, mapK, image):
    # Plot result
    fig0, (ax) = plt.subplots(ncols=1)
    im = ax.imshow(image[:,:,:])
    ax.plot(GT[:, 1], GT[:, 0], '--b', lw=3)
    ax.plot(u, v, '--r', lw=3)
    plt.colorbar(im, ax=ax)
    fig0.suptitle('Image, GT (blue) and converged snake (red)', fontsize=20)

    fig1, (ax0, ax1, ax2,ax3,ax4) = plt.subplots(ncols=5)
    im0 = ax0.imshow(mapDu)
    plt.colorbar(im0, ax=ax0)
    im1 = ax1.imshow(mapDv)
    plt.colorbar(im1, ax=ax1)
    im2 = ax2.imshow(mapA)
    plt.colorbar(im2, ax=ax2)
    im3 = ax3.imshow(mapB)
    plt.colorbar(im3, ax=ax3)
    im4 = ax4.imshow(mapK)
    plt.colorbar(im4, ax=ax4)
    fig1.suptitle('Output maps', fontsize=20)

    plt.show()

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

#Load data
L = 50
batch_size = 1
csvfile=open('random_rectangles/rects.csv', newline='')
reader = csv.reader(csvfile)
images = np.zeros([128,128,3,1000])
dists = np.zeros([128,128,1,1000])
masks = np.zeros([128,128,1,1000])
GT = np.zeros([L,2,1000])
for i in range(1000):
    poly = np.zeros([5, 2])
    corners = reader.__next__()
    for c in range(4):
        poly[c, 1] = np.float(corners[1+2*c])
        poly[c, 0] = np.float(corners[2+2*c])
    poly[4,:] = poly[0,:]
    [tck, u] = interpolate.splprep([poly[:, 0], poly[:, 1]], s=2, k=1, per=1)
    [GT[:,0,i], GT[:,1,i]] = interpolate.splev(np.linspace(0, 1, L), tck)
    this_im  = scipy.misc.imread('random_rectangles/rect'+str(i)+'.png')
    images[:,:,:,i] = np.float32(this_im)/255
    img_mask = scipy.misc.imread('random_rectangles/rectorig' + str(i) + '.png')/255
    masks[:,:,0,i] = img_mask
    img_dist = scipy.ndimage.morphology.distance_transform_edt(img_mask) + \
               scipy.ndimage.morphology.distance_transform_edt(1 - img_mask)
    img_dist = gaussian(img_dist, 10)
    dists[:,:,0,i] =  img_dist
GT = np.minimum(GT,127)
GT = np.maximum(GT,0)


sess = tf.InteractiveSession()

#Input and output
x = tf.placeholder(tf.float32, shape=[128, 128, 3])
x_image = tf.reshape(x, [-1, 128, 128, 3])
y_ = tf.placeholder(tf.float32, shape=GT[:,:,0].shape)

#First conv layer
W_conv1 = weight_variable([5, 5, 3, 16])
b_conv1 = bias_variable([16])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = batch_norm(max_pool_2x2(h_conv1))


#Second conv layer
W_conv2 = weight_variable([5, 5, 16, 32])
b_conv2 = bias_variable([32])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = batch_norm(max_pool_2x2(h_conv2))

#Third conv layer
W_conv3 = weight_variable([5, 5, 32, 32])
b_conv3 = bias_variable([32])
h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
h_pool3 = batch_norm(max_pool_2x2(h_conv3))

#Resize and concat
resized_out1 = tf.image.resize_images(h_pool1, [128, 128])
resized_out2 = tf.image.resize_images(h_pool2, [128, 128])
resized_out3 = tf.image.resize_images(h_pool3, [128, 128])
h_concat = tf.concat([resized_out1,resized_out2,resized_out3],3)

#Forth conv layer
W_conv4 = weight_variable([5, 5, int(h_concat.shape[3]), 32])
b_conv4 = bias_variable([32])
h_conv4 = batch_norm(tf.nn.relu(conv2d(h_concat, W_conv4) + b_conv4))

#Predict force field
W_fcDu = weight_variable([1, 1, 32, 1])
b_fcDu = bias_variable([1])
h_fcDu = conv2d(h_conv4, W_fcDu) + b_fcDu
predDu = tf.reshape(h_fcDu,[128,128])
W_fcDv = weight_variable([1, 1, 32, 1])
b_fcDv = bias_variable([1])
h_fcDv = conv2d(h_conv4, W_fcDv) + b_fcDv
predDv = tf.reshape(h_fcDv,[128,128])
#Predict alpha
W_fcA = weight_variable([1, 1, 32, 1])
b_fcA = bias_variable([1])
h_fcA = conv2d(h_conv4, W_fcA) + b_fcA
predA = tf.nn.softplus(tf.reshape(h_fcA,[128,128]))
#predA = tf.reshape(h_fcA,[128,128,1,-1])
#Predict beta
W_fcB = weight_variable([1, 1, 32, 1])
b_fcB = bias_variable([1])
h_fcB = conv2d(h_conv4, W_fcB) + b_fcB
predB = tf.nn.softplus(tf.reshape(h_fcB,[128,128]))
#Predict kappa
W_fcK = weight_variable([1, 1, 32, 1])
b_fcK = bias_variable([1])
h_fcK = conv2d(h_conv4, W_fcK) + b_fcK
predK = 0.1*tf.reshape(h_fcK,[128,128])


# Placeholders for the initial snakes
tf_snake_init_u = tf.placeholder(tf.float32, shape=[L, 1])
tf_snake_init_v = tf.placeholder(tf.float32, shape=[L, 1])
# Variables to store the step vectors
tf_du = tf.zeros([L, 1])
tf_dv = tf.zeros([L, 1])
# Constants
gamma = tf.constant(2,dtype=tf.float32)
max_px_move = tf.constant(4,dtype=tf.float32)
delta_s = tf.constant(1,dtype=tf.float32)

# Apply the snake evolution steps
tf_u = tf_snake_init_u
tf_v = tf_snake_init_v
for i in range(20):
    tf_u,tf_v,tf_du,tf_dv=active_contour_step(predDu, predDv, tf_du, tf_dv, tf_u, tf_v,
                    predA, predB, predK,
                    gamma,max_px_move, delta_s)
snake_u = tf_u
snake_v = tf_v
#Inject the gradients
optimizer = tf.train.AdamOptimizer(0.0001, epsilon=1e-7)
grad_u = tf.placeholder(tf.float32, shape=[L, 1])
grad_v = tf.placeholder(tf.float32, shape=[L, 1])
tvars = tf.trainable_variables()
grads = tf.gradients([snake_u,snake_u], tvars, grad_ys = [grad_u,grad_v])
apply_gradients = optimizer.apply_gradients(zip(grads, tvars))
#Initialize CNN
init = tf.global_variables_initializer()
sess.run(init)

s = np.linspace(0, 2*np.pi, L)
init_u = 64 + 40*np.cos(s)
init_v = 64 + 40*np.sin(s)
init_u = init_u.reshape([L,1])
init_v = init_v.reshape([L,1])

for epoch in range(1):
    for i in range(500):
        print(i)
        #Do CNN inference
        batch_ind = i
        batch = images[:,:,:,batch_ind]
        batch_dists = dists[:, :, :, batch_ind]
        [pred_u,pred_v,mapDu,mapDv,mapA,mapB,mapK] = \
            sess.run([tf_u,tf_v,predDu,predDv,predA, predB, predK],
                     feed_dict={x:batch,tf_snake_init_u:init_u,tf_snake_init_v:init_v})


        # get the inside-ouside values for the normal force
        k = []
        u = np.int32(np.round(pred_u))
        v = np.int32(np.round(pred_v))
        for j in range(L):
            k.append(masks[u[j, 0], v[j, 0],0,i])
        k = np.stack(k).reshape([L,1])*2-1
        # get normals
        n_u = np.concatenate([pred_v[1:L], pred_v[0:1]], axis=0) \
              - np.concatenate([pred_v[L - 1:L], pred_v[0:L - 1]], axis=0)
        n_v = np.concatenate([pred_u[L - 1:L], pred_u[0:L - 1]], axis=0) \
              - np.concatenate([pred_u[1:L], pred_u[0:1]], axis=0)
        norm = np.sqrt(np.power(n_u, 2) + np.power(n_v, 2))
        n_u = np.divide(n_u, norm)
        n_v = np.divide(n_v, norm)
        # get the gradients to apply from the normals
        grad_u_pred = -(n_u * k).reshape([L,1])
        grad_v_pred = -(n_v * k).reshape([L,1])

        thisGT = GT[:, :, batch_ind]
        if divmod(i,50)[1]==0:
            #plt.imshow(out[:,:,0,0])
            plot_snakes(pred_u,pred_v, thisGT, mapDu, mapDv,
                        mapA, mapB, mapK, batch)
        #Apply gradients
        apply_gradients.run(feed_dict={x:batch,grad_u:grad_u_pred,grad_v:grad_v_pred,
                            tf_snake_init_u: init_u, tf_snake_init_v: init_v})

plot_snakes(pred_u, pred_v, thisGT, mapDu, mapDv,
            mapA, mapB, mapK, batch)






