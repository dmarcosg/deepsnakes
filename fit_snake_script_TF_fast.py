import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from skimage.filters import gaussian
from scipy import interpolate
import scipy
#from skimage.segmentation import active_contour
from snake_inference_fast_TF import active_contour_step
import time


filename = 'square_energy.png'
img = io.imread(filename)
img = img[:,:,0]
img = np.float32(img)/255
filename = 'square_corners.png'
img_beta = io.imread(filename)
img_beta = img_beta[:,:,0]
img_beta = np.float32(img_beta)*0.01
filename = 'square_alpha.png'
img_alpha = io.imread(filename)
img_kappa = (0.02 - np.float32(img_alpha[:,:,0])*0.0001)*1
img_alpha = img_alpha[:,:,0]
img_alpha = np.float32(img_alpha)*0.0001

Du = np.gradient(gaussian(img,4),axis=0)*500
Dv = np.gradient(gaussian(img,4),axis=1)*500


L = 60
s = np.linspace(0, 2*np.pi, L,endpoint=False)
init_u = 128 + 80*np.cos(s)
init_v = 128 + 80*np.sin(s)
init_u = init_u.reshape([L,1])
init_v = init_v.reshape([L,1])

sess = tf.InteractiveSession()

tf_alpha = tf.placeholder(tf.float32, shape=[256, 256])
tf_beta = tf.placeholder(tf.float32, shape=[256, 256])
tf_kappa = tf.placeholder(tf.float32, shape=[256, 256])
tf_Du = tf.placeholder(tf.float32, shape=[256, 256])
tf_Dv = tf.placeholder(tf.float32, shape=[256, 256])
tf_u = tf.placeholder(tf.float32, shape=[L, 1])
tf_v = tf.placeholder(tf.float32, shape=[L, 1])
tf_du = tf.placeholder(tf.float32, shape=[L, 1])
tf_dv = tf.placeholder(tf.float32, shape=[L, 1])
gamma = tf.constant(1,dtype=tf.float32)
max_px_move = tf.constant(1,dtype=tf.float32)
delta_s = tf.constant(1,dtype=tf.float32)


tf_u2,tf_v2,tf_du2,tf_dv2=active_contour_step(tf_Du, tf_Dv, tf_du, tf_dv, tf_u, tf_v,
                    tf_alpha, tf_beta, tf_kappa,
                    gamma,max_px_move, delta_s)


u = init_u
v = init_v
du = np.zeros([L, 1])
dv = np.zeros([L, 1])
snake_hist = []
snake_hist.append(np.array([u[:, 0], v[:, 0]]).T)
tic = time.time()
for i in range(200):
    u, v, du, dv = sess.run([tf_u2,tf_v2,tf_du2,tf_dv2],feed_dict={tf_Du:Du,tf_Dv:Dv,
                                tf_u:u,tf_v:v, tf_du:du, tf_dv:dv,
                                tf_alpha:img_alpha,tf_beta:img_beta,tf_kappa:img_kappa})
    snake_hist.append(np.array([u[:, 0], v[:, 0]]).T)
print('%.2f' % (time.time() - tic) + ' s apply gradients')

fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(111)
plt.gray()
ax.imshow(img)
for i in range(0,len(snake_hist)):
    ax.plot(snake_hist[i][:, 1], snake_hist[i][:, 0], '-', color=[1-i / len(snake_hist), 1-i / len(snake_hist), i / len(snake_hist)], lw=1)
ax.plot(init_v, init_u, '--b', lw=3)
ax.plot(v, u, '--r', lw=3)
ax.set_xticks([]), ax.set_yticks([])
ax.axis([0, img.shape[1], img.shape[0], 0])
plt.show()