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
from shutil import copyfile,rmtree
from shapely.geometry import Polygon, MultiPolygon, mapping
import json
from shapely.affinity import affine_transform

#print('Waiting...',flush=True)
#time.sleep(40*60)

print('Importing packages... done!',flush=True)


do_plot = False
do_write_results = True
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
L = 60
numfilt = [32,64,128,128,256,256]
batch_size = 1
im_size = 384
out_size = 192
if intoronto:
    images_path = '/ais/dgx1/marcosdi/TCityBuildings/val_building_crops/'
    #gt_path = '/ais/dgx1/marcosdi/TCityBuildings/building_crops_gt/'
    dwt_path = '/ais/dgx1/marcosdi/TCityBuildings/val_building_crops_dwt/'
    model_path = 'models/tcity_fullB1_dilated/'
    results_path = '/ais/dgx1/marcosdi/resultsB1/crops/'
    results_path_geojson = '/ais/dgx1/marcosdi/resultsB1/geojson/'
else:
    images_path = '/mnt/bighd/Data/TorontoCityTile/building_crops/'
    gt_path = '/mnt/bighd/Data/TorontoCityTile/building_crops_gt/'
    dwt_path = '/mnt/bighd/Data/TorontoCityTile/building_crops_dwt_dilated/'
    model_path = 'models/tcity_B1_dilated/'
    results_path = '/mnt/bighd/Data/TorontoCityTile/resultsB1/crops/'
    results_path_geojson = '/mnt/bighd/Data/TorontoCityTile/resultsB1/geojson/'




###########################################################################################
# LOAD POLYGON DATA
###########################################################################################
print('Preparing to read the polygons...',flush=True)
files = os.listdir(images_path)
csv_names = [f for f in files if f[-4:] == '.csv']
png_names = [f for f in files if f[-4:] == '.png']
total_num = len(png_names)
images = np.zeros([im_size,im_size,3,epoch_batch_size],dtype=np.uint8)
#masks = np.zeros([out_size,out_size,1,epoch_batch_size],dtype=np.uint8)
#GT = np.zeros([L,2,epoch_batch_size])
DWT = np.zeros([L,2,epoch_batch_size])

#allGT = np.zeros([L,2,total_num])
allDWT = np.zeros([L,2,total_num])

all_building_names = []
all_bounding_boxes = []
building_names = []
all_snakes = []

# For each TCity tile, since there's one .csv per tile containing the bounding boxes
total_count = 0
for csv_name in csv_names:
    i = 0
    tile_name = csv_name[0:-7]
    print('Reading tile: '+ tile_name,flush=True)
    #copyfile(dwt_path + tile_name + '_bb.csv',results_path + tile_name + '_bb.csv')
    csvfile_bb = open(dwt_path + tile_name + '_bb.csv', newline='')
    reader_bb = csv.reader(csvfile_bb)
    csvfile_dwt = open(dwt_path + tile_name + '_polygons.csv', newline='')
    reader_dwt = csv.reader(csvfile_dwt)
    while True:
        try:
            bb = reader_bb.__next__()
            corners_dwt = reader_dwt.__next__()
        except:
            print('Buildings loaded: '+str(i)+', total: '+str(total_count),flush=True)
            break
        # Get bounding boxes
        all_bounding_boxes.append(np.int32(bb))
        
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
        # Get image
        all_building_names.append(tile_name + '_building_' + str(i + 1).zfill(4) + '.png')
        i += 1
        total_count += 1

#allGT = np.minimum(allGT,out_size-1)
#allGT = np.maximum(allGT,0)
allDWT = np.minimum(allDWT,out_size-1)
allDWT = np.maximum(allDWT,0)

assert total_count == total_num, 'Different number of buildings found than expected!'


###########################################################################################
# DEFINE RESAMPLER TO ACTUALLY LOAD THE IMAGES
###########################################################################################
def resample_images(inds):
    print('Resampling images...', flush=True)
    for i in range(len(inds)):
        this_im = scipy.misc.imread(images_path + all_building_names[inds[i]])
        images[:, :, :, i] = scipy.misc.imresize(this_im, [im_size, im_size])
        #this_mask = scipy.misc.imread(gt_path + all_building_names[inds[i]])
        #masks[:, :, 0, i] = scipy.misc.imresize(this_mask, [out_size, out_size], interp='nearest') > 0
        #GT[:,:,i] = allGT[:,:,inds[i]]
        DWT[:, :, i] = allDWT[:, :, inds[i]]
        building_names.append(all_building_names[inds[i]])
        bounding_boxes.append(all_bounding_boxes[inds[i]])


###########################################################################################
# DEFINE CNN ARCHITECTURE
###########################################################################################
print('Creating CNN...',flush=True)
with tf.device('/gpu:0'):
    tvars, grads, predE, predA, predB, predK, l2loss, grad_predE, \
    grad_predA, grad_predB, grad_predK, grad_l2loss, x, y_ = CNN_B(im_size, out_size, L, batch_size=1,wd=0.01,layers=len(numfilt),numfilt=numfilt)

#Initialize CNN
optimizer = tf.train.AdamOptimizer(1e-5, epsilon=1e-7)
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
#Prepare folder to save network and results
###########################################################################################
print('Loading model...',flush=True)
if not os.path.isdir(results_path):
    os.makedirs(results_path)
else:
    rmtree(results_path)
    os.makedirs(results_path)

if not os.path.isdir(results_path_geojson):
    os.makedirs(results_path_geojson)
else:
    rmtree(results_path_geojson)
    os.makedirs(results_path_geojson)

if not os.path.isdir(results_path_geojson+'init/'):
    os.makedirs(results_path_geojson+'init/')
else:
    rmtree(results_path_geojson+'init/')
    os.makedirs(results_path_geojson+'init/')

saver = tf.train.Saver()



###########################################################################################
# DEFINE EPOCH
###########################################################################################
def epoch(n,i,mode):
    # mode (str): train or test
    batch_ind = np.arange(i,i+batch_size)
    batch = np.float32(images[:, :, :, batch_ind])/255
    #batch_mask = np.copy(masks[:, :, :, batch_ind])
    thisNames = building_names[batch_ind[0]]
    this_bb = np.copy(bounding_boxes[batch_ind[0]])
    base_name = thisNames.split('_')[0]+'_'+thisNames.split('_')[1]
    #thisGT = np.copy(GT[:, :, batch_ind[0]])
    thisDWT = np.copy(DWT[:, :, batch_ind[0]])
    # prediction_np = sess.run(prediction,feed_dict={x:batch})
    [mapE, mapA, mapB, mapK, l2] = sess.run([predE, predA, predB, predK, l2loss], feed_dict={x: batch})
    mapA = np.maximum(mapA, 0)
    mapB = np.maximum(mapB, 0)
    mapK = np.maximum(mapK, 0)

    for j in range(batch_size):
        init_snake = thisDWT
        snake, snake_hist = snake_process(mapE, mapA, mapB, mapK, init_snake)
        # Get last layer gradients
        M = mapE.shape[0]
        N = mapE.shape[1]
        der1, der2 = derivatives_poly(snake)


        #der1_GT, der2_GT = derivatives_poly(thisGT)

        #grads_arrayE = mapE * 0.001
        #grads_arrayA = mapA * 0.001
        #grads_arrayB = mapB * 0.001
        #grads_arrayK = mapK * 0.001
        #grads_arrayE[:, :, 0, 0] -= draw_poly(snake, 1, [M, N],12) - draw_poly(thisGT, 1, [M, N],12)
        #grads_arrayA[:, :, 0, 0] -= (np.mean(der1) - np.mean(der1_GT))
        #grads_arrayB[:, :, 0, 0] -= (draw_poly(snake, der2, [M, N],12) - draw_poly(thisGT, der2_GT, [M, N],12))
        #mask_gt = draw_poly_fill(thisGT, [M, N])
        #mask_snake = draw_poly_fill(snake, [M, N])
        #grads_arrayK[:, :, 0, 0] -= mask_gt - mask_snake

        #intersection = (mask_gt+mask_snake) == 2
        #union = (mask_gt + mask_snake) >= 1
        #iou = np.sum(intersection) / np.sum(union)

    if do_plot:
        plot_snakes(snake, snake_hist, None, mapE, mapA, mapB, mapK, \
                        None, None, None, None, batch, None)
    if do_write_results:
        if all_snakes.__contains__(base_name):
            all_snakes[base_name]['snakes'].append(np.copy(snake)*im_size/out_size)
            all_snakes[base_name]['init'].append(thisDWT * im_size / out_size)
            all_snakes[base_name]['bb'].append(this_bb)
        else:
            all_snakes[base_name] = {}
            all_snakes[base_name]['snakes'] = [np.copy(snake)*im_size/out_size]
            all_snakes[base_name]['init'] = [thisDWT * im_size / out_size]
            all_snakes[base_name]['bb'] = [this_bb]

    return


###########################################################################################
# RUN THE TESTING
###########################################################################################
all_polygons = {}
all_snakes = {}
with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,log_device_placement=False)) as sess:
    sess2 = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,log_device_placement=False))
    save_path = tf.train.latest_checkpoint(model_path)
    init = tf.global_variables_initializer()
    sess.run(init)
    saver.restore(sess, save_path)
    start_epoch = int(save_path.split('-')[-1].split('.')[0]) + 1
    print('Loading epoch '+str(start_epoch),flush=True)
    start_ind = 0
    total_iter_count = 0
    total_batches = np.int32(np.ceil(total_num / epoch_batch_size))
    iou_test = 0
    for n in range(total_batches):
        inds = np.arange(start_ind,np.minimum(total_num,start_ind+epoch_batch_size))
        start_ind += epoch_batch_size
        building_names = []
        bounding_boxes = []
        resample_images(inds)

        iter_count = 0

        for i in range(len(inds)):
            epoch(n,i, 'test')
            iter_count += 1
            total_iter_count += 1
            print('Test. Batch ' + str(n) + '. Iter ' + str(total_iter_count) + '/' + str(total_num),flush=True)


for name in all_snakes.keys():
    all_polygons = []
    all_inits = []
    for count in range(len(all_snakes[name]['snakes'])):
        snake = np.copy(all_snakes[name]['snakes'][count]) / im_size
        this_bb = np.copy(all_snakes[name]['bb'][count])
        snake_as_tuples = []
        for i in range(snake.shape[0]):
            snake_as_tuples.append((snake[i, 1]*this_bb[2] , snake[i, 0]*this_bb[3]))
        all_polygons.append(Polygon(snake_as_tuples).buffer(0).simplify(0.5, preserve_topology = True))
        xcoord = np.float32(name.split('_')[0])
        ycoord = np.float32(name.split('_')[1])
        all_polygons[count] = affine_transform(all_polygons[count],
                                       [0.10, 0, 0, -0.10, xcoord + this_bb[0] / 10, ycoord - this_bb[1] / 10 + 500.0])

        init = np.copy(all_snakes[name]['init'][count]) / im_size
        init_as_tuples = []
        for i in range(snake.shape[0]):
            init_as_tuples.append((init[i, 1] * this_bb[2], init[i, 0] * this_bb[3]))
        all_inits.append(Polygon(init_as_tuples).buffer(0).simplify(0.5, preserve_topology = True))
        all_inits[count] = affine_transform(all_inits[count],
                                               [0.10, 0, 0, -0.10, xcoord + this_bb[0] / 10,
                                                ycoord - this_bb[1] / 10 + 500.0])

    polygons = MultiPolygon(all_polygons)
    inits = MultiPolygon(all_inits)
    with open(results_path_geojson+name+'_snakes.geojson', 'w') as gj:
        json.dump(mapping(polygons), gj)
    with open(results_path_geojson+'init/'+name+'_init.geojson', 'w') as gj:
        json.dump(mapping(inits), gj)
    print(name)




# v, u = polygon.exterior.xy
# new_snake = np.stack([np.array(u), np.array(v)]).T
# mask_snake = draw_poly_fill(new_snake, [im_size, im_size])
# scipy.misc.imsave(results_path+thisNames,scipy.misc.imresize(mask_snake,[im_size,im_size],interp='nearest'))
# f = open(results_path + name + '_polygons.csv', 'a', newline='')
# writer = csv.writer(f)
# writer.writerow([len(new_snake), new_snake.reshape(2 * len(new_snake))])
# f.close()









