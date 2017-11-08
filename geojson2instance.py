import geojson
from rasterio import features
import rasterio
from shapely.geometry import shape
import png
import os

intoronto = True

if intoronto:
    geojson_path = "/ais/gobi4/TorontoCity/test/shenlong/improved_gt_train/"
    output_path = "/ais/dgx1/marcosdi/improved_gt_train/"
else:
    geojson_path = "/mnt/bighd/Data/TorontoCityTile/"
    output_path = "/mnt/bighd/Data/TorontoCityTile/"



input_postfix = "_buildings"
ouput_postfix = "_instances"

files = os.listdir(geojson_path)
geojson_names = [f for f in files if len(f.split(input_postfix+'.')) > 1 and f.split(input_postfix+'.')[1] == 'geojson']

if not os.path.isdir(output_path):
    os.makedirs(output_path)

for i in range(len(geojson_names)):
    name = geojson_names[i].split(input_postfix)[0]
    x = float(name.split('_')[0])
    y = float(name.split('_')[1])
    #t = rasterio.transform.from_bounds(0.10, 0, 0, -0.10, x, y + 500.0)
    t = rasterio.transform.from_bounds(x, y, x+500, y+500, 5000, 5000)
    with open(geojson_path+geojson_names[i]) as f:
        gj = geojson.load(f)
        s = shape(gj)
        #b = features.rasterize(((g, 255) for g in s), out_shape=[5000, 5000], transform=t)
        val = 0
        pairs = []
        for g in s:
            val += 1
            pairs.append((g,val))
        b = features.rasterize(pairs, out_shape=[5000, 5000], transform=t)
        imf = open(output_path+name+ouput_postfix+'.png', 'wb')
        w = png.Writer(b.shape[0], b.shape[1], greyscale=True, bitdepth=16)
        w.write(imf, b)
        imf.close()




