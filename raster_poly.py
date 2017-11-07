import numpy as np
import scipy.misc
import os
import glob
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import time
import json
import argparse
import rasterio.features
from shapely.geometry import mapping, shape, MultiPolygon, Polygon
from shapely.affinity import affine_transform
from skimage.morphology import remove_small_objects, erosion, dilation, opening, closing, disk


def do_generate(input_folder, output_folder, postfix = ''):
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    im_list = glob.glob(os.path.join(input_folder, '*'+postfix+'.png'))
    print(os.path.join(input_folder, '*'+postfix+'.png'))
    for im_file in im_list:
        im_id = os.path.basename(im_file).split('.')[0].split('_')[0] + '_' + os.path.basename(im_file).split('.')[0].split('_')[1]
        # print(im_id)
        seg_file = os.path.join(input_folder, im_id + postfix+'.png') 
        pred_mask = scipy.misc.imread(seg_file)
        poly = raster_poly(im_id + '_pred', output_folder, pred_mask, verbose = True)
        poly.polygonize()
        poly.simplify()
        poly.to_geojson()
        print('%s polygonization is finished.' % im_id)

def get_poly_no(massing_data, h=500, w=500):
    poly_list = []
    for feature in massing_data['features']:
        object_id = feature['properties']['OBJECTID']
        pts_list = [] 
        poly = feature['geometry']['coordinates'][0]
        for pts in poly:
            pts_list.append((pts[0], pts[1]))
        if len(pts_list) > 4:
            poly = Polygon(pts_list)
            poly_list.append(poly)
    return MultiPolygon(poly_list)

def generate_gt(exp):
    ortho_folder = '/ais/gobi4/TorontoCity/data/Generated/CVPR_DemoArea/' + exp + '/Buildings/'
    output_folder = '/ais/gobi4/TorontoCity/test/shenlong/struct_polygon/' + exp
    if (not os.path.exists(output_folder)):
        os.mkdir(output_folder)
    print('Loadinng massing data...')
    start = time.time()
    print(time.time()-start)
    
    im_list = glob.glob(os.path.join(ortho_folder, "*.geojson"))
    # for name in ['10311048330.sdw']:
    print(im_list) 
    k = 0
    for name in im_list:
        start = time.time()
        im_id = os.path.basename(name).split('.')[0].split('_b')[0]
        out_file = os.path.join(output_folder, im_id+'_buildings.geojson')
        with open('/ais/gobi4/TorontoCity/data/Generated/CVPR_DemoArea/'+exp+'/Buildings/'+im_id + '_buildings.geojson') as data_file:    
            data = json.load(data_file)
        polys = get_poly_no(data)
        with open (out_file, 'w') as f:
            json.dump(mapping(polys), f)

class raster_poly(object):

    def __init__(self, name, output_folder, mask, verbose = True):
        self.name = name
        self.x = float(name.split('_')[0])
        self.y = float(name.split('_')[1])
        print(self.x, self.y)
        self.geojson = os.path.join(output_folder, name + '.geojson')
        self.mask = remove_small_objects(mask, 1000)
        self.verbose = verbose

    def polygonize(self):
        poly_list = [] 
        for vec in rasterio.features.shapes(self.mask): 
            poly = affine_transform(shape(vec[0]), [0.10, 0, 0, -0.10, self.x, self.y + 500.0]) 
            if self.verbose:
                print(poly, type(poly))
            if (poly.length < 1000):
                poly_list.append(poly)
        self.poly = MultiPolygon(poly_list)
   
    def simplify(self):
        simple_poly = []
        for poly in self.poly:
            simple_poly.append(poly.simplify(0.5, preserve_topology = True))
        self.poly = MultiPolygon(simple_poly)

    def to_geojson(self):
        print(self.poly)
        with open (self.geojson, 'w') as f:
            json.dump(mapping(self.poly), f)

    def update_outfolder(self, output_folder):
        self.geojson = os.path.join(output_folder, self.name + '.geojson')

    def to_turning(self):
        self.north_pole = self.find_north()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Instance segmentation evaluation")
    parser.add_argument("-i", "--pred_folder", default = '/ais/gobi4/TorontoCity/test/shenlong/gellert', help="pred folder")
    parser.add_argument("-p", "--postfix", default = '', help="postfix")
    parser.add_argument("-o", "--out_folder", default = '/ais/gobi4/TorontoCity/test/shenlong/gellert', help="output folder")
    args=parser.parse_args()
    do_generate(args.pred_folder, args.out_folder, args.postfix)
    # generate_gt('train')
