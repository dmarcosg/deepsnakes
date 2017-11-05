print('Importing packages...')

import tensorflow as tf
import scipy.misc
import numpy as np
import csv
import os
from active_contour_maps_GD_fast import draw_poly,derivatives_poly,draw_poly_fill
from snake_utils import imrotate, plot_snakes, polygon_area, CNN, snake_graph
from scipy import interpolate
import scipy
import time

#print('Waiting...',flush=True)
#time.sleep(40*60)

print('Importing packages... done!',flush=True)

model_path = 'models/tcity1/'
do_plot = True
only_test = True
intoronto = False
epoch_batch_size = 1000

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
L = 80
batch_size = 1
im_size = 384
out_size = 192
if intoronto:
    images_path = '/ais/dgx1/marcosdi/TCityBuildings/building_crops/'
    gt_path = '/ais/dgx1/marcosdi/TCityBuildings/building_crops_gt/'
    dwt_path = '/ais/dgx1/marcosdi/TCityBuildings/building_crops_dwt/'
else:
    images_path = '/mnt/bighd/Data/TorontoCityTile/building_crops/'
    gt_path = '/mnt/bighd/Data/TorontoCityTile/building_crops_gt/'
    dwt_path = '/mnt/bighd/Data/TorontoCityTile/building_crops_dwt/'

###########################################################################################
# LOAD POLYGON DATA
###########################################################################################
print('Preparing to read the polygons...',flush=True)
files = os.listdir(images_path)
csv_names = [f for f in files if f[-4:] == '.csv']
png_names = [f for f in files if f[-4:] == '.png']
total_num = len(png_names)
images = np.zeros([im_size,im_size,3,epoch_batch_size],dtype=np.uint8)
masks = np.zeros([out_size,out_size,1,epoch_batch_size],dtype=np.uint8)
GT = np.zeros([L,2,epoch_batch_size])
DWT = np.zeros([L,2,epoch_batch_size])

allGT = np.zeros([L,2,total_num])
allDWT = np.zeros([L,2,total_num])

all_building_names = []

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
    while True:
        try:
            corners_gt = reader_gt.__next__()
            corners_dwt = reader_dwt.__next__()
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
        i += 1
        total_count += 1

allGT = np.minimum(allGT,out_size-1)
allGT = np.maximum(allGT,0)
allDWT = np.minimum(allDWT,out_size-1)
allDWT = np.maximum(allDWT,0)

assert total_count == total_num, 'Different number of buildings found than expected!'


###########################################################################################
# DEFINE RESAMPLER TO ACTUALLY LOAD THE IMAGES
###########################################################################################
def resample_images(inds):
    print('Resampling images...', flush=True)
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
    grad_predA, grad_predB, grad_predK, grad_l2loss, x, y_ = CNN(im_size, out_size, L, batch_size=1)

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
print('Loading model...',flush=True)
start_epoch = 0
assert os.path.isdir(model_path), 'No model found in the specified directory'
modelnames = []
modelnames += [each for each in os.listdir(model_path) if each.endswith('.net')]
epoch = -1
for s in modelnames:
    epoch = max(int(s.split('-')[-1].split('.')[0]),epoch)
start_epoch = epoch + 1


###########################################################################################
# DEFINE EPOCH
###########################################################################################
def epoch(n,i,mode):
    # mode (str): train or test
    batch_ind = np.arange(i,i+batch_size)
    batch = np.float32(images[:, :, :, batch_ind])/255
    batch_mask = np.copy(masks[:, :, :, batch_ind])
    thisNames = all_building_names[batch_ind]
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
    mapB = np.maximum(mapB, 0)
    mapK = np.maximum(mapK, 0)
    #print('%.2f' % (time.time() - tic) + ' s tf inference')
    if mode is 'train':
        for j in range(mapK.shape[3]):
            mapK[:, :, 0, j] -= batch_mask[:, :, 0, j] * 0.5 - 0.5 / 2
    # Do snake inference
    for j in range(batch_size):
        init_snake = thisDWT
        snake, snake_hist = snake_process(mapE, mapA, mapB, mapK, init_snake)
        # Get last layer gradients
        M = mapE.shape[0]
        N = mapE.shape[1]
        der1, der2 = derivatives_poly(snake)


        der1_GT, der2_GT = derivatives_poly(thisGT)

        grads_arrayE = mapE * 0.001
        grads_arrayA = mapA * 0.001
        grads_arrayB = mapB * 0.001
        grads_arrayK = mapK * 0.001
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
    if do_plot and n >=3:
        plot_snakes(snake, snake_hist, thisGT, mapE, np.maximum(mapA, 0), np.maximum(mapB, 0), mapK, \
                grads_arrayE, grads_arrayA, grads_arrayB, grads_arrayK, batch, batch_mask)
        #plt.show()
    return iou


###########################################################################################
# RUN THE TESTING
###########################################################################################
with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,log_device_placement=True)) as sess:
    sess2 = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,log_device_placement=True))
    save_path = tf.train.latest_checkpoint(model_path)
    init = tf.global_variables_initializer()
    sess.run(init)
    start_ind = 0
    total_batches = np.ceil(total_num / epoch_batch_size)
    for n in range(total_batches):
        inds = np.arange(start_ind,np.maximum(total_num,start_ind+epoch_batch_size))
        resample_images(inds)
        iou_test = 0
        iter_count = 0

        for i in range(len(inds)):
            iou_test += epoch(n,i, 'test')
            iter_count += 1
            print('Test. Batch ' + str(n) + '. Iter ' + str(iter_count) + '/' + str(total_num) + ', IoU = %.4f' % (
            iou_test / iter_count),flush=True)













