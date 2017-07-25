import tensorflow as tf
import scipy.misc
import numpy as np
import csv
import matplotlib.pyplot as plt
from active_contour_maps_GD2 import active_contour_step, draw_poly,derivatives_poly
from scipy import interpolate

def snake_process (maps, init_snake):
    gamma = 0.3
    max_px_move = 2
    delta_s = 1
    maxiter = 500
    for i in range(maps.shape[3]):
        Du = np.gradient(maps[:,:,0,i], axis=0)
        Dv = np.gradient(maps[:,:,0,i], axis=1)
        u = init_snake[:,0:1]
        v = init_snake[:,1:2]
        du = np.zeros(u.shape)
        dv = np.zeros(v.shape)
        for j in range(maxiter):
            u, v, du, dv = active_contour_step(Du, Dv, du, dv, u, v,
                                               maps[:, :, 1, i], maps[:,:,2,i],
                                               gamma, max_px_move, delta_s)



    return np.array([u[:,0],v[:,0]]).T


def plot_snakes(snake,GT,maps,grads_array,image):
    # Plot result
    fig0, (ax) = plt.subplots(ncols=1)
    im = ax.imshow(image[:,:,0,0])
    ax.plot(GT[:, 0, 0], GT[:, 1, 0], '--b', lw=3)
    ax.plot(snake[:, 0], snake[:, 1], '--r', lw=3)
    plt.colorbar(im, ax=ax)
    fig0.suptitle('Image, GT (blue) and converged snake (red)', fontsize=20)

    fig1, (ax0, ax1, ax2) = plt.subplots(ncols=3)
    im0 = ax0.imshow(maps[:, :, 0, 0])
    plt.colorbar(im0, ax=ax0)
    im1 = ax1.imshow(maps[:, :, 1, 0])
    plt.colorbar(im1, ax=ax1)
    im2 = ax2.imshow(maps[:, :, 2, 0])
    plt.colorbar(im2, ax=ax2)
    fig1.suptitle('Output maps', fontsize=20)

    fig1, (ax0, ax1, ax2) = plt.subplots(ncols=3)
    im0 = ax0.imshow(grads_array[:, :, 0, 0])
    plt.colorbar(im0, ax=ax0)
    im1 = ax1.imshow(grads_array[:, :, 1, 0])
    plt.colorbar(im1, ax=ax1)
    im2 = ax2.imshow(grads_array[:, :, 2, 0])
    plt.colorbar(im2, ax=ax2)
    fig1.suptitle('Gradient maps', fontsize=20)

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

#Load data
L = 70
batch_size = 1;
csvfile=open('random_rectangles/rects.csv', newline='')
reader = csv.reader(csvfile)
images = np.zeros([128,128,1,1000])
GT = np.zeros([L,2,1000])
for i in range(1000):
    poly = np.zeros([5, 2])
    corners = reader.__next__()
    for c in range(4):
        poly[c, 0] = np.float(corners[1+2*c])
        poly[c, 1] = np.float(corners[2+2*c])
    poly[4,:] = poly[0,:]
    [tck, u] = interpolate.splprep([poly[:, 0], poly[:, 1]], s=2, k=1, per=1)
    [GT[:,0,i], GT[:,1,i]] = interpolate.splev(np.linspace(0, 1, L), tck)
    images[:,:,0,i] = scipy.misc.imread('random_rectangles/rect'+str(i)+'.png')



sess = tf.InteractiveSession()

#Input and output
x = tf.placeholder(tf.float32, shape=[128, 128, 1, batch_size])
x_image = tf.reshape(x, [-1, 128, 128, 1])
y_ = tf.placeholder(tf.float32, shape=GT[:,:,0].shape)

#First conv layer
W_conv1 = weight_variable([5, 5, 1, 16])
b_conv1 = bias_variable([16])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

#Second conv layer
W_conv2 = weight_variable([5, 5, 16, 32])
b_conv2 = bias_variable([32])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

#Third conv layer
W_conv3 = weight_variable([5, 5, 32, 32])
b_conv3 = bias_variable([32])
h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
h_pool3 = max_pool_2x2(h_conv3)

#Resize and concat
resized_out1 = tf.image.resize_images(h_pool1, [128, 128])
resized_out2 = tf.image.resize_images(h_pool2, [128, 128])
resized_out3 = tf.image.resize_images(h_pool3, [128, 128])
h_concat = tf.concat([resized_out1,resized_out2,resized_out3],3)

#Forth conv layer
W_conv4 = weight_variable([1, 1, int(h_concat.shape[3]), 32])
b_conv4 = bias_variable([32])
h_conv4 = tf.nn.relu(conv2d(h_concat, W_conv4) + b_conv4)

#Fifth conv layer
W_conv5 = weight_variable([5, 5, 32, 3])
b_conv5 = bias_variable([3])
h_conv5 = conv2d(h_conv4, W_conv5) + b_conv5
prediction = tf.nn.softplus(tf.reshape(h_conv5,[128,128,3,-1]))


#Inject the gradients
optimizer = tf.train.AdamOptimizer(0.0001, epsilon=1e-7)


grad_pred = tf.placeholder(tf.float32, shape=[128, 128, 3, batch_size])
tvars = tf.trainable_variables()
grads = tf.gradients(prediction, tvars, grad_ys = grad_pred)

apply_gradients = optimizer.apply_gradients(zip(grads, tvars))

#Initialize CNN
init = tf.global_variables_initializer()
sess.run(init)

for i in range(100):
    print(i)
    #Do CNN inference
    batch = images[:,:,:,0:0+1]
    #prediction_np = sess.run(prediction,feed_dict={x:batch})
    out = sess.run(prediction,feed_dict={x:batch})
    #Do snake inference
    s = np.linspace(0, 2 * np.pi, L)
    init_u = 64 + 40 * np.cos(s)
    init_v = 64 + 40 * np.sin(s)
    init_u = init_u.reshape([L, 1])
    init_v = init_v.reshape([L, 1])
    init_snake = np.array([init_u[:,0],init_v[:,0]]).T
    snake = snake_process(out,  init_snake)
    # Get last layer gradients
    M = out.shape[0]
    N = out.shape[1]
    der1,der2 = derivatives_poly(snake)
    der1_GT, der2_GT = derivatives_poly(GT[:, :, 0])
    grads_array = np.zeros(out.shape)
    grads_array[:,:,0,0] -= draw_poly(snake,1,[M,N],200) - draw_poly(GT[:, :, 0],1,[M,N],200)
    grads_array[:, :, 1, 0] -= 0.01*(draw_poly(snake, der1 - np.mean(der1_GT), [M, N], 200))
    grads_array[:, :, 2, 0] -= 0.01*(draw_poly(snake, der2, [M, N], 200) - draw_poly(GT[:, :, 0], der2_GT, [M, N], 200))
    #plot_snakes(snake, GT, out, grads_array, batch)
    if divmod(i,10)[1]==0:
        plt.imshow(out[:,:,0,0])
        plt.show()
    #Apply gradients
    apply_gradients.run(feed_dict={x:batch,grad_pred:grads_array})

plot_snakes(snake, GT, out, grads_array,batch)







