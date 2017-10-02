import tensorflow as tf
import scipy.misc
import numpy as np
import csv
import matplotlib.pyplot as plt
from active_contour_maps_GD_fast import active_contour_step, draw_poly,derivatives_poly,draw_poly_fill
from scipy import interpolate
from skimage.filters import gaussian
import scipy

def snake_process (mapE, mapA, mapB, mapK, init_snake):
    gamma = 1
    max_px_move = 3
    delta_s = 1
    maxiter = 300

    for i in range(mapE.shape[3]):
        Du = np.gradient(mapE[:,:,0,i], axis=0)
        Dv = np.gradient(mapE[:,:,0,i], axis=1)
        u = init_snake[:,0:1]
        v = init_snake[:,1:2]
        du = np.zeros(u.shape)
        dv = np.zeros(v.shape)
        snake_hist = []
        for j in range(maxiter):
            u, v, du, dv = active_contour_step(Du, Dv, du, dv, u, v,
                                               mapA[:, :, 0, i], mapB[:,:,0,i],mapK[:,:,0,i],
                                               gamma, max_px_move, delta_s)
            snake_hist.append(np.array([u[:,0],v[:,0]]).T)



    return np.array([u[:,0],v[:,0]]).T,snake_hist


def plot_snakes(snake,snake_hist,GT,mapE, mapA, mapB, mapK, grads_arrayE, grads_arrayA, grads_arrayB, grads_arrayK, image, mask):
    # Plot result
    fig0, (ax) = plt.subplots(ncols=1)
    im = ax.imshow(image[:,:,:,0])
    for i in range(0,len(snake_hist),5):
        ax.plot(snake_hist[i][:, 1], snake_hist[i][:, 0], '-', color=[1-i / len(snake_hist), 1-i / len(snake_hist), i / len(snake_hist)], lw=1)
    ax.plot(snake[:, 1], snake[:, 0], '--k', lw=3)
    ax.plot(GT[:, 1], GT[:, 0], '--r', lw=3)
    plt.colorbar(im, ax=ax)
    fig0.suptitle('Image, GT (red) and converged snake (black)', fontsize=20)

    fig0, (ax) = plt.subplots(ncols=1)
    im = ax.imshow(mask[:, :, 0, 0])
    ax.plot(GT[:, 1], GT[:, 0], '--r', lw=3)
    plt.colorbar(im, ax=ax)

    fig1, ax = plt.subplots(ncols=2,nrows=2)
    im0 = ax[0,0].imshow(mapE[:, :, 0, 0])
    plt.colorbar(im0, ax=ax[0,0])
    ax[0, 0].set_title('D')
    im1 = ax[0,1].imshow(mapA[:, :, 0, 0])
    plt.colorbar(im1, ax=ax[0,1])
    ax[0, 1].set_title('alpha')
    im2 = ax[1,0].imshow(mapB[:, :, 0, 0])
    plt.colorbar(im2, ax=ax[1,0])
    ax[1, 0].set_title('beta')
    im3 = ax[1,1].imshow(mapK[:, :, 0, 0])
    plt.colorbar(im3, ax=ax[1,1])
    ax[1, 1].set_title('kappa')
    fig1.suptitle('Output maps', fontsize=20)

    fig2, ax = plt.subplots(ncols=2,nrows=2)
    im0 = ax[0,0].imshow(grads_arrayE[:, :, 0, 0])
    plt.colorbar(im0, ax=ax[0,0])
    ax[0, 0].set_title('D')
    im1 = ax[0,1].imshow(grads_arrayA[:, :, 0, 0])
    plt.colorbar(im1, ax=ax[0,1])
    ax[0, 1].set_title('alpha')
    im2 = ax[1,0].imshow(grads_arrayB[:, :, 0, 0])
    plt.colorbar(im2, ax=ax[1,0])
    ax[1, 0].set_title('beta')
    im3 = ax[1,1].imshow(grads_arrayK[:, :, 0, 0])
    plt.colorbar(im3, ax=ax[1,1])
    ax[1, 1].set_title('kappa')
    fig2.suptitle('Gradient maps', fontsize=20)

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
L = 60
batch_size = 1
im_size = 512
data_path = '/mnt/bighd/Data/Vaihingen/buildings/'
csvfile=open(data_path+'polygons.csv', newline='')
reader = csv.reader(csvfile)
images = np.zeros([im_size,im_size,3,168])
dists = np.zeros([im_size,im_size,1,168])
masks = np.zeros([im_size,im_size,1,168])
GT = np.zeros([L,2,168])
for i in range(168):
    corners = reader.__next__()
    num_points = np.int32(corners[0])
    poly = np.zeros([num_points, 2])
    for c in range(num_points):
        poly[c, 0] = np.float(corners[1+2*c])
        poly[c, 1] = np.float(corners[2+2*c])
    [tck, u] = interpolate.splprep([poly[:, 0], poly[:, 1]], s=2, k=1, per=1)
    [GT[:,0,i], GT[:,1,i]] = interpolate.splev(np.linspace(0, 1, L), tck)
    this_im  = scipy.misc.imread(data_path+'building_'+str(i+1).zfill(3)+'.tif')
    images[:,:,:,i] = np.float32(this_im)/255
    img_mask = scipy.misc.imread(data_path+'building_mask_' + str(i+1).zfill(3) + '.tif')/255
    masks[:,:,0,i] = img_mask
    img_dist = scipy.ndimage.morphology.distance_transform_edt(img_mask) + \
               scipy.ndimage.morphology.distance_transform_edt(1 - img_mask)
    img_dist = gaussian(img_dist, 10)
    dists[:,:,0,i] =  img_dist
GT = np.minimum(GT,im_size-1)
GT = np.maximum(GT,0)


sess = tf.InteractiveSession()

#Input and output
x = tf.placeholder(tf.float32, shape=[im_size, im_size, 3, batch_size])
x_image = tf.reshape(x, [-1, im_size, im_size, 3])
y_ = tf.placeholder(tf.float32, shape=GT[:,:,0].shape)

#First conv layer
W_conv1 = weight_variable([3, 3, 3, 16])
b_conv1 = bias_variable([16])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
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
resized_out1 = tf.image.resize_images(h_pool1, [im_size, im_size])
resized_out2 = tf.image.resize_images(h_pool2, [im_size, im_size])
resized_out3 = tf.image.resize_images(h_pool3, [im_size, im_size])
h_concat = tf.concat([resized_out1,resized_out2,resized_out3],3)

#Forth conv layer
W_conv4 = weight_variable([3, 3, int(h_concat.shape[3]), 32])
b_conv4 = bias_variable([32])
h_conv4 = batch_norm(tf.nn.relu(conv2d(h_concat, W_conv4) + b_conv4))

#Predict energy
W_fcE = weight_variable([1, 1, 32, 1])
b_fcE = bias_variable([1])
h_fcE = conv2d(h_conv4, W_fcE) + b_fcE
predE = tf.reshape(h_fcE,[im_size,im_size,1,-1])
#Predict alpha
W_fcA = weight_variable([1, 1, 32, 1])
b_fcA = bias_variable([1])
h_fcA = conv2d(h_conv4, W_fcA) + b_fcA
#predA = tf.nn.softplus(tf.reshape(h_fcA,[im_size,im_size,1,-1]))
predA = tf.reshape(h_fcA,[im_size,im_size,1,-1])
#Predict beta
W_fcB = weight_variable([1, 1, 32, 1])
b_fcB = bias_variable([1])
h_fcB = conv2d(h_conv4, W_fcB) + b_fcB
predB = tf.reshape(h_fcB,[im_size,im_size,1,-1])
#Predict kappa
W_fcK = weight_variable([1, 1, 32, 1])
b_fcK = bias_variable([1])
h_fcK = conv2d(h_conv4, W_fcK) + b_fcK
predK = 0.02*tf.reshape(h_fcK,[im_size,im_size,1,-1])



#Inject the gradients
optimizer = tf.train.AdamOptimizer(0.0003, epsilon=1e-7)
grad_predE = tf.placeholder(tf.float32, shape=[im_size, im_size, 1, batch_size])
grad_predA = tf.placeholder(tf.float32, shape=[im_size, im_size, 1, batch_size])
grad_predB = tf.placeholder(tf.float32, shape=[im_size, im_size, 1, batch_size])
grad_predK = tf.placeholder(tf.float32, shape=[im_size, im_size, 1, batch_size])
tvars = tf.trainable_variables()
grads = tf.gradients([predE,predA,predB,predK], tvars, grad_ys = [grad_predE,grad_predA,grad_predB,grad_predK])
apply_gradients = optimizer.apply_gradients(zip(grads, tvars))

#Initialize CNN
init = tf.global_variables_initializer()
sess.run(init)

for epoch in range(5):
    for i in range(100):
        print(i)
        #Do CNN inference
        batch_ind = [i]
        batch = images[:,:,:,batch_ind]
        batch_mask = masks[:, :, :, batch_ind]
        batch_dists = dists[:, :, :, batch_ind]
        #prediction_np = sess.run(prediction,feed_dict={x:batch})
        [mapE, mapA, mapB, mapK] = sess.run([predE,predA,predB,predK],feed_dict={x:batch})
        mapE_aug = np.zeros(mapE.shape)
        for j in range(mapE.shape[3]):
            max_val = np.amax(np.abs(mapE[:,:,0,j]))
            #mapE[:,:,0,j] = gaussian(mapE[:,:,0,j]/max_val,3)*max_val
            mapE_aug[:,:,0,j] = mapE[:,:,0,j]+np.maximum(0,20-batch_dists[:,:,0,j])*max_val/50
        #Do snake inference
        s = np.linspace(0, 2 * np.pi, L)
        init_u = 256 + 40 * np.cos(s)
        init_v = 256 + 40 * np.sin(s)
        init_u = init_u.reshape([L, 1])
        init_v = init_v.reshape([L, 1])
        init_snake = np.array([init_u[:,0],init_v[:,0]]).T
        snake,snake_hist = snake_process(mapE_aug, np.maximum(0,mapA), np.maximum(0,mapB), mapK,  init_snake)
        # Get last layer gradients
        M = mapE.shape[0]
        N = mapE.shape[1]
        der1,der2 = derivatives_poly(snake)
        thisGT = GT[:, :, batch_ind[0]]
        der1_GT, der2_GT = derivatives_poly(thisGT)

        # get the inside-ouside values for the normal force
        k = []
        u = np.int32(np.round(snake[:,0:1]))
        v = np.int32(np.round(snake[:,1:2]))
        for j in range(L):
            k.append(masks[u[j, 0], v[j, 0],0,i])
        k = np.stack(k).reshape([L, ])*2-1
        grads_arrayE = np.zeros(mapE.shape)
        grads_arrayA = np.zeros(mapA.shape)
        grads_arrayB = np.zeros(mapB.shape)
        grads_arrayK = np.zeros(mapK.shape)
        grads_arrayE[:,:,0,0] -= draw_poly(snake,1,[M,N],200) - draw_poly(thisGT,1,[M,N],200)
        grads_arrayA[:,:,0,0] -= (draw_poly(snake, der1 - np.mean(der1_GT), [M, N], 200))
        grads_arrayB[:,:,0,0] -= (draw_poly(snake, der2, [M, N], 200) - draw_poly(thisGT, der2_GT, [M, N], 200))
        grads_arrayK[:, :, 0, 0] -= draw_poly_fill(thisGT, [M, N]) - draw_poly_fill(snake, [M, N])

        if divmod(i,99)[1]==0:
            #plt.imshow(out[:,:,0,0])
            plot_snakes(snake, snake_hist, thisGT, mapE_aug, np.maximum(mapA, 0), np.maximum(mapB, 0), mapK, \
                        grads_arrayE, grads_arrayA, grads_arrayB, grads_arrayK, batch, batch_mask)
            plt.show()
        #Apply gradients
        apply_gradients.run(feed_dict={x:batch,grad_predE:grads_arrayE,grad_predA:grads_arrayA,grad_predB:grads_arrayB,grad_predK:grads_arrayK})

#plot_snakes(snake,snake_hist, thisGT, mapE, np.maximum(mapA,0), np.maximum(mapB,0), mapK,\
#                    grads_arrayE, grads_arrayA, grads_arrayB, grads_arrayK, batch)
#plt.show()






