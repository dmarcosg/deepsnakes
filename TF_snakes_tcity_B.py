print('Importing packages...')

import tensorflow as tf
import scipy.misc
import numpy as np
import csv
import os
from active_contour_maps_GD_fast import draw_poly,derivatives_poly,draw_poly_fill
from snake_utils import imrotate, plot_snakes, polygon_area, CNN_B, snake_graph
from scipy import interpolate
import scipy
import time

#print('Waiting...',flush=True)
#time.sleep(40*60)

print('Importing packages... done!',flush=True)

model_path = 'models/tcity_fullB1/'
do_plot = False
only_test = True
intoronto = True
epoch_batch_size = 1000
val_proportion = 0.2

def snake_process (mapE, mapA, mapB, mapK, init_snake):

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

        #print('%.2f' % (time.time() - tic) + ' s snake')

    return np.array([u[:,0],v[:,0]]).T,snake_hist







#Load data
L = 60
numfilt = [32,64,128,128,256,256]
batch_size = 1
im_size = 384
out_size = 192
if intoronto:
    images_path = '/ais/dgx1/marcosdi/TCityBuildings_improved/building_crops/'
    gt_path = '/ais/dgx1/marcosdi/TCityBuildings_improved/building_crops_gt/'
    dwt_path = '/ais/dgx1/marcosdi/TCityBuildings_improved/building_crops_dwt_dilate/'
else:
    images_path = '/mnt/bighd/Data/TorontoCityTile/building_crops/'
    gt_path = '/mnt/bighd/Data/TorontoCityTile/building_crops_gt/'
    dwt_path = '/mnt/bighd/Data/TorontoCityTile/building_crops_dwt_dilate/'

###########################################################################################
# LOAD POLYGON DATA
###########################################################################################
print('Preparing to read the polygons...',flush=True)
files = os.listdir(images_path)
csv_names = [f for f in files if f[-4:] == '.csv']
png_names = [f for f in files if f[-4:] == '.png']
total_num = len(png_names)
train_ims = np.int32(np.floor((1-val_proportion)*epoch_batch_size))
test_ims = np.int32(np.floor((val_proportion)*epoch_batch_size))
images = np.zeros([im_size,im_size,3,epoch_batch_size],dtype=np.uint8)
masks = np.zeros([out_size,out_size,1,epoch_batch_size],dtype=np.uint8)
GT = np.zeros([L,2,epoch_batch_size])
DWT = np.zeros([L,2,epoch_batch_size])

allGT = np.zeros([L,2,total_num])
allDWT = np.zeros([L,2,total_num])

all_building_names = []
all_building_sizes = []

# For each TCity tile, since there's one .csv per tile containing the bounding boxes
total_count = 0
for csv_name in csv_names:
    i = 0
    tile_name = csv_name[0:-7]
    print('Reading tile: '+ tile_name,flush=True)
    csvfile_gt = open(gt_path + tile_name + '_polygons.csv', newline='')
    reader_gt = csv.reader(csvfile_gt)
    csvfile_dwt = open(dwt_path + tile_name + '_polygons.csv', newline='')
    reader_dwt = csv.reader(csvfile_dwt)
    csvfile_bb = open(gt_path + tile_name + '_bb.csv', newline='')
    reader_bb = csv.reader(csvfile_bb)
    while True:
        try:
            corners_gt = reader_gt.__next__()
            corners_dwt = reader_dwt.__next__()
            bb = reader_bb.__next__()
        except:
            print('Buildings loaded: '+str(i)+', total: '+str(total_count),flush=True)
            break
        # Get GT polygons
        num_points = np.int32(corners_gt[0])
        poly = np.zeros([num_points, 2])
        for c in range(num_points):
            poly[c, 0] = np.float(corners_gt[1 + 2 * c]) * out_size / im_size
            poly[c, 1] = np.float(corners_gt[2 + 2 * c]) * out_size / im_size
        [tck, u] = interpolate.splprep([poly[:, 0], poly[:, 1]], s=2, k=1, per=1)
        [allGT[:, 0, total_count], allGT[:, 1, total_count]] = interpolate.splev(np.linspace(0, 1, L), tck)
        # Get DWT polygons
        num_points = np.int32(corners_dwt[0])
        poly = np.zeros([num_points, 2])
        for c in range(num_points):
            poly[c, 0] = np.float(corners_dwt[1 + 2 * c]) * out_size / im_size
            poly[c, 1] = np.float(corners_dwt[2 + 2 * c]) * out_size / im_size
        [tck, u] = interpolate.splprep([poly[:, 0], poly[:, 1]], s=2, k=1, per=1)
        [allDWT[:, 0, total_count], allDWT[:, 1, total_count]] = interpolate.splev(np.linspace(0, 1, L), tck)
        if polygon_area(allDWT[:, 0, total_count],allDWT[:, 1, total_count]) < 0:
            allDWT[:, :, total_count] = allDWT[::-1, :, total_count]
        # Get image and GT mask
        all_building_names.append(tile_name + '_building_' + str(i + 1).zfill(4) + '.png')
        all_building_sizes.append((np.float32(bb[2])+np.float32(bb[3]))/2)
        i += 1
        total_count += 1

allGT = np.minimum(allGT,out_size-1)
allGT = np.maximum(allGT,0)
allDWT = np.minimum(allDWT,out_size-1)
allDWT = np.maximum(allDWT,0)

all_relative_sizes = np.stack(all_building_sizes) / min(np.stack(all_building_sizes))
all_relative_sizes = np.int32(np.ceil(all_relative_sizes))
print('Max rel. size = '+str(max(all_relative_sizes)))
weighted_inds = []
for i in range(total_num):
    mult = np.maximum(1,all_relative_sizes[i]-5)
    mult = int(mult*np.log(np.e+mult))
    weighted_inds.append(np.int32(np.ones([mult])*i))
weighted_inds = np.concatenate(weighted_inds)


assert total_count == total_num, 'Different number of buildings found than expected!'


###########################################################################################
# DEFINE RESAMPLER TO ACTUALLY LOAD THE IMAGES
###########################################################################################
def resample_images():
    print('Resampling images...', flush=True)
    all_inds = np.random.permutation(weighted_inds)
    inds = all_inds[np.arange(epoch_batch_size)]
    for i in range(epoch_batch_size):
        this_im = scipy.misc.imread(images_path + all_building_names[inds[i]])
        images[:, :, :, i] = scipy.misc.imresize(this_im, [im_size, im_size])
        this_mask = scipy.misc.imread(gt_path + all_building_names[inds[i]])
        masks[:, :, 0, i] = scipy.misc.imresize(this_mask, [out_size, out_size], interp='nearest') > 0
        GT[:,:,i] = allGT[:,:,inds[i]]
        DWT[:, :, i] = allDWT[:, :, inds[i]]


###########################################################################################
# DEFINE CNN ARCHITECTURE
###########################################################################################
print('Creating CNN...',flush=True)
with tf.device('/gpu:0'):
    tvars, grads, predE, predA, predB, predK, l2loss, grad_predE, \
    grad_predA, grad_predB, grad_predK, grad_l2loss, x, y_ = CNN_B(im_size, out_size, L,
                                                                 batch_size=1,wd=0.01,layers=len(numfilt),numfilt=numfilt)

#Initialize CNN
optimizer = tf.train.AdamOptimizer(1e-6, epsilon=1e-7)
apply_gradients = optimizer.apply_gradients(zip(grads, tvars))

###########################################################################################
# DEFINE SNAKE INFERENCE
###########################################################################################
niter = 50
print('Creating snake inference graph...',flush=True)
with tf.device('/cpu:0'):
    tf_u, tf_v, tf_du, tf_dv, tf_Du, tf_Dv, tf_u0, tf_v0, tf_du0, tf_dv0, \
    tf_alpha, tf_beta, tf_kappa = snake_graph(out_size, L,niter=niter)

###########################################################################################
#Prepare folder to save network
###########################################################################################
print('Preparing model folder...',flush=True)
start_epoch = 0
if not os.path.isdir(model_path):
    os.makedirs(model_path)

# Add ops to save and restore all the variables.
saver = tf.train.Saver()


###########################################################################################
# DEFINE EPOCH
###########################################################################################
def epoch(n,i,mode):
    # mode (str): train or test
    batch_ind = np.arange(i,i+batch_size)
    batch = np.float32(images[:, :, :, batch_ind])/255
    batch_mask = np.copy(masks[:, :, :, batch_ind])
    thisGT = np.copy(GT[:, :, batch_ind[0]])
    thisDWT = np.copy(DWT[:, :, batch_ind[0]])
    if mode is 'train':
        ang = np.random.rand() * 360
        for j in range(len(batch_ind)):
            for b in range(batch.shape[2]):
                batch[:, :, b, j] = imrotate(batch[:, :, b, j], ang)
            batch_mask[:, :, 0, j] = imrotate(batch_mask[:, :, 0, j], ang, resample='nearest')
        R = [[np.cos(ang * np.pi / 180), np.sin(ang * np.pi / 180)],
             [-np.sin(ang * np.pi / 180), np.cos(ang * np.pi / 180)]]
        thisGT -= out_size / 2
        thisGT = np.matmul(thisGT, R)
        thisGT += out_size / 2
        thisDWT -= out_size / 2
        thisDWT = np.matmul(thisDWT, R)
        thisDWT += out_size / 2
        thisGT = np.minimum(thisGT, out_size - 1)
        thisGT = np.maximum(thisGT, 0)
        thisDWT = np.minimum(thisDWT, out_size - 1)
        thisDWT = np.maximum(thisDWT, 0)
    # prediction_np = sess.run(prediction,feed_dict={x:batch})
    [mapE, mapA, mapB, mapK, l2] = sess.run([predE, predA, predB, predK, l2loss], feed_dict={x: batch})

    if mode is 'train':
        for j in range(mapK.shape[3]):
            mapK[:, :, 0, j] -= batch_mask[:, :, 0, j] * 0.5 - 0.5 / 2
    mapA = np.maximum(mapA, 0)
    mapB = np.maximum(mapB, 0)
    mapK = np.maximum(mapK, 0)
    # Do snake inference
    for j in range(batch_size):
        init_snake = thisDWT
        snake, snake_hist = snake_process(mapE, mapA, mapB, mapK, init_snake)
        # Get last layer gradients
        M = mapE.shape[0]
        N = mapE.shape[1]
        der1, der2 = derivatives_poly(snake)


        der1_GT, der2_GT = derivatives_poly(thisGT)

        grads_arrayE = mapE * 0.01
        grads_arrayA = mapA * 0.01
        grads_arrayB = mapB * 0.01
        grads_arrayK = mapK * 0.01
        grads_arrayE[:, :, 0, 0] -= draw_poly(snake, 1, [M, N],12) - draw_poly(thisGT, 1, [M, N],12)
        grads_arrayA[:, :, 0, 0] -= (np.mean(der1) - np.mean(der1_GT))
        grads_arrayB[:, :, 0, 0] -= (draw_poly(snake, der2, [M, N],12) - draw_poly(thisGT, der2_GT, [M, N],12))
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
                       grad_predK: grads_arrayK, grad_l2loss: 1})
        #print('%.2f' % (time.time() - tic) + ' s apply gradients')
        #print('IoU = %.2f' % (iou))
    #if mode is 'test':
        #print('IoU = %.2f' % (iou))
    if do_plot and n >=0:
        plot_snakes(snake, snake_hist, thisGT, mapE, np.maximum(mapA, 0), np.maximum(mapB, 0), mapK, \
                grads_arrayE, grads_arrayA, grads_arrayB, grads_arrayK, batch, batch_mask)
        #plt.show()
    return iou


###########################################################################################
# RUN THE TRAINING
###########################################################################################
with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,log_device_placement=True)) as sess:
    sess2 = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,log_device_placement=True))
    save_path = tf.train.latest_checkpoint(model_path)
    init = tf.global_variables_initializer()
    sess.run(init)
    if save_path is not None:
        saver.restore(sess,save_path)
        start_epoch = int(save_path.split('-')[-1].split('.')[0])+1

    for n in range(start_epoch,1000):
        resample_images()
        iou_test = 0
        iou_train = 0
        iter_count = 0
        if not only_test:
            for i in range(0,train_ims,batch_size):
                #print(i)
                #Do CNN inference
                iou_train += epoch(n,i,'train')
                iter_count += 1
                print('Train. Epoch ' + str(n) + '. Iter ' + str(iter_count) + '/' + str(train_ims) + ', IoU = %.4f' % (
                iou_train / iter_count),flush=True)
            iou_train /= train_ims
            saver.save(sess,model_path+'model', global_step=n)

        iter_count = 0
        for i in range(train_ims,train_ims+test_ims):
            iou_test += epoch(n,i, 'test')
            iter_count += 1
            print('Test. Epoch ' + str(n) + '. Iter ' + str(iter_count) + '/' + str(test_ims) + ', IoU = %.4f' % (
            iou_test / iter_count),flush=True)
        if not only_test:
            iou_test /= test_ims
            iou_csvfile = open(model_path + 'iuo_train_test.csv', 'a', newline='')
            iou_writer = csv.writer(iou_csvfile)
            iou_writer.writerow([n,iou_train,iou_test])
            iou_csvfile.close()











