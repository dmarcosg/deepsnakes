import matplotlib.pyplot as plt
import scipy.misc
import numpy as np
import csv

im_size = 512
out_size = 256

im_path = '/mnt/bighd/Data/Vaihingen/buildings'

poly_paths = ['/mnt/bighd/Data/Vaihingen/buildings/polygons.csv',
              '/home/diego/PycharmProjects/snakes_prj/deepsnakes/models/base_vaiB1/results',
              '/home/diego/PycharmProjects/snakes_prj/deepsnakes/models/vaiB1/results']


polygons = []
for i in range(len(poly_paths)):
    polygons[i] = []
    csvfile = open(poly_paths[i] + 'polygons.csv', newline='')
    reader = csv.reader(csvfile)
    while True:
        try:
            line = reader.__next__()
        except:
            break
        num_points = np.int32(line[0])
        poly = np.zeros([num_points, 2])
        for c in range(num_points):
            poly[c, 0] = np.float(line[1 + 2 * c]) * out_size / im_size
            poly[c, 1] = np.float(line[2 + 2 * c]) * out_size / im_size
        polygons[i].append(poly)