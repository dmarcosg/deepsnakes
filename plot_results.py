import matplotlib.pyplot as plt
import scipy.misc
import numpy as np
import csv
import os

dataset = 'vaihingen'


if dataset is 'vaihingen':
    imnums = [1, 10, 40, 67, 3, 5,31]
    im_size = 512
    out_size = 256
    im_path = '/mnt/bighd/Data/Vaihingen/buildings/val_set/'
    poly_paths = ['/mnt/bighd/Data/Vaihingen/buildings/val_set/polygons.csv',
                  '/home/diego/PycharmProjects/snakes_prj/deepsnakes/models/base_vaiB1/results/polygons_val.csv',
                  '/home/diego/PycharmProjects/snakes_prj/deepsnakes/models/vaiB1/results/polygons.csv']
elif dataset is 'bing':
    imnums = [1,30,50,100,150,200,220]
    im_size = 80
    out_size = 80
    im_path = '/mnt/bighd/Data/BingJohn/buildings_osm/single_buildings/test/'
    poly_paths = ['/mnt/bighd/Data/BingJohn/buildings_osm/single_buildings/test/building_coords.csv',
                  '/home/diego/PycharmProjects/snakes_prj/deepsnakes/models/base_bingB1/results/polygons.csv',
                  '/home/diego/PycharmProjects/snakes_prj/deepsnakes/models/bingB1/results/polygons.csv']

files = os.listdir(im_path)
im_names = [f for f in files if f[-4:] == '.png' or f[-4:] == '.tif']
im_names = sorted(im_names)
images = np.zeros([im_size,im_size,3,len(im_names)],dtype=np.uint8)
for i in range(len(im_names)):
    this_im = scipy.misc.imread(im_path + im_names[i])
    if len(this_im.shape) == 3:
        images[:, :, :, i] = scipy.misc.imresize(this_im, [im_size, im_size])
polygons = []
for i in range(len(poly_paths)):
    print(i)
    polygons.append([])
    csvfile = open(poly_paths[i], newline='')
    reader = csv.reader(csvfile)
    while True:
        try:
            line = reader.__next__()
        except:
            break
        num_points = np.int32(line[0])+1
        if dataset is 'bing' and i == 0:
            num_points = 5
        poly = np.zeros([num_points, 2])
        for c in range(num_points-1):
            poly[c, 0] = np.float(line[1 + 2 * c])
            poly[c, 1] = np.float(line[2 + 2 * c])
        poly[-1,:] = poly[0,:]
        if i > 0:
            poly = poly*im_size/out_size
        polygons[i].append(poly)

colors = [[0,0.7,0],[0,0,1],[1,1,0]]
styles = ['-','-.','--']
fig, (ax) = plt.subplots(ncols=len(imnums),nrows=1)
for i in range(len(imnums)):
    im = ax[i].imshow(np.float32(np.abs(images[:, :, :, imnums[i]]))/255)
    for j in range(len(poly_paths)):
        ax[i].plot(polygons[j][imnums[i]][:, 1], polygons[j][imnums[i]][:, 0], styles[j], lw=3,color=colors[j])
    ax[i].axis('off')
    if dataset is 'vaihingen':
        ax[i].set_xlim([100,400])
        ax[i].set_ylim([100,400])
    if dataset is 'bing':
        ax[i].set_xlim([15,65])
        ax[i].set_ylim([15,65])

plt.show()