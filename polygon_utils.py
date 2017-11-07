import numpy as np
import scipy.misc
import os
import sys 
import glob
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches
from matplotlib.collections import PatchCollection
import json
import argparse
import rasterio.features
from shapely.geometry import mapping, shape, MultiPolygon
from shapely.geometry import Polygon, Point, LineString
from shapely.affinity import affine_transform
from skimage.morphology import remove_small_objects, erosion, dilation, opening, closing, disk

import math, random
import pdb

def read_poly(poly_file):
    with open (poly_file, 'r') as f:
        json_dict = json.load(f)
        return shape(json_dict)

def clip(x, min, max):
    if( min > max ) :  return x    
    elif( x < min ) :  return min
    elif( x > max ) :  return max
    else :             return x

def poly_sim(gt_polys, pred_polys):
    sum_sim = 0.0
    sum_iou = 0.0
    sum_area = 0.0
    new_pred = [] 
    for pred_poly in pred_polys:
        if not pred_poly.is_valid:
            pred_poly = pred_poly.buffer(0)
        new_pred.append(pred_poly)
    pred_polys = MultiPolygon(new_pred)
 
    for gt_poly in gt_polys:
        max_iou = 0.0
        max_sim = 0.0
        max_shape = 0.0
        area = gt_poly.area
        for pred_poly in pred_polys:
            temp_poly = pred_poly
            temp_iou = iou(gt_poly, temp_poly)
            if (temp_iou > max_iou):
                max_iou = temp_iou
                shape_sim = distance_turning(turning(gt_poly), turning(temp_poly))
                max_sim = max_iou * shape_sim 
                max_shape = shape_sim 
        sum_iou += max_iou * area
        sum_sim += max_sim * area
        sum_area += area
        print('Area: %2.4f, IOU: %2.4f, SHAPE: %2.4f, SIM: %2.4f' % (area, max_iou, max_shape, max_sim))
        sys.stdout.flush()
    return sum_sim, sum_iou, sum_area

def is_intersect(poly_exterior, line):
    x, y = poly_exterior.coords.xy
    for i in range(len(x)-1):
        line_temp = LineString([Point(x[i], y[i]), Point(x[i+1], y[i+1])])
        # print(i, line_temp.intersects(line))
        if line_temp.intersects(line):
            return i, line_temp.intersection(line)
    return -1, 0

def insert_point(poly):
    center = poly.centroid
    poly_exterior = poly.exterior
    coords = poly_exterior.coords[:]
    north = Point(center.x, center.y+5000.0)
    north_line = LineString([center, north])
    idx_north, intersect_pts_north = is_intersect(poly_exterior, north_line)
    
    south = Point(center.x, center.y-5000.0)
    south_line = LineString([center, south])
    idx_south, intersect_pts_south = is_intersect(poly_exterior, south_line)
   
    if ((idx_north == -1) and (idx_south == -1)):
        return poly
    elif (idx_north != -1):
        idx = idx_north
        intersect_pts = intersect_pts_north
    else:
        idx = idx_south
        intersect_pts = intersect_pts_south
    coords1 = coords[0:idx+1]
    coords2 = coords[idx+1:-1]
    coords2.insert(0, (intersect_pts.x, intersect_pts.y))
    return Polygon(coords2+coords1)

def line_angle(line):
    coords = line.coords[:]
    x = coords[1][0] - coords[0][0]
    y = coords[1][1] - coords[0][1]
    return math.atan2(x, y)
    
def turning(poly):
    poly_new = insert_point(poly)
    length = poly_new.length
    coords = poly_new.exterior.coords[:]
    turn_fun = []
    current = 0
    for i in range(len(coords)-1):
        line = LineString([Point(poly_new.exterior.coords[:][i]), Point(poly_new.exterior.coords[:][i+1])])
        turn_fun.append((current, line.length / length, line_angle(line)))
        current += line.length / length
    return turn_fun

def iou(poly1, poly2):
    # print(poly1.exterior.coords, poly2.exterior.coords)
    # pdb.set_trace()
    # i = Polygon(poly1.exterior).union(Polygon(poly2.exterior)).area
    # sys.stdout.flush()
    # if not poly2.is_valid:
    #     print(poly2.is_valid)
    # else:
    i = poly1.intersection(poly2).area
    if i == 0:
        return 0.0
    # print(i, list(poly1.exterior.coords), list(poly2.exterior.coords)) 
    # u = Polygon(poly1.exterior).union(Polygon(poly2.exterior)).area
    u = poly1.union(poly2).area
    return i/u

def plot_turning(turn_fun, mycolor):
    for i in range(len(turn_fun)):
        plt.hold(True)
        plt.plot(turn_fun[i][0], turn_fun[i][2], mycolor+'o', markersize = 10.0)
        plt.plot([turn_fun[i][0], turn_fun[i][0]+turn_fun[i][1]], [turn_fun[i][2] , turn_fun[i][2]], mycolor, linewidth = 5)
        plt.plot(turn_fun[i][0]+turn_fun[i][1], turn_fun[i][2], 'wo', markeredgecolor = mycolor, markersize = 10.0)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-math.pi-0.15, math.pi+0.15])
    plt.grid(True)
    plt.xlabel('Perimeter Distance')
    plt.ylabel('Turning Angle')

def angle_dist(ang1, ang2):
    dist = min(abs(ang1 - ang2), abs(ang1+math.pi*2 - ang2))
    dist = min(dist, abs(ang1-math.pi*2 - ang2))
    return dist
    
def distance_turning(turn_fun1, turn_fun2):
    curr_pts = 0.0
    stat1 = 0
    stat2 = 0
    dist = 0.0
    while(stat1 < len(turn_fun1)-1 or stat2 < len(turn_fun2)-1):
        if stat1 < len(turn_fun1)-1:
            next1 = turn_fun1[stat1+1][0]
        else:
            next1 = 1.0
            
        if stat2 < len(turn_fun2)-1:
            next2 = turn_fun2[stat2+1][0]
        else:
            next2 = 1.0
        
        dist_temp = angle_dist(turn_fun1[stat1][2], turn_fun2[stat2][2])
        
        if next1 < next2:
            next_pts = next1
            stat1 = stat1 + 1
        else:
            next_pts = next2
            stat2 = stat2 + 1
        # print(curr_pts, next_pts - curr_pts, dist_temp)    
        dist += dist_temp / math.pi * (next_pts - curr_pts)
        curr_pts = next_pts
        
    dist_temp = angle_dist(turn_fun1[stat1][2], turn_fun2[stat2][2])
    dist += dist_temp / math.pi * (1 - curr_pts)
    # print(curr_pts, 1 - curr_pts, dist_temp)   
    return (1.0 - dist)


def generatePolygon( ctrX, ctrY, aveRadius, irregularity, spikeyness, numVerts ):
    '''Start with the centre of the polygon at ctrX, ctrY, 
    then creates the polygon by sampling points on a circle around the centre. 
    Randon noise is added by varying the angular spacing between sequential points,
    and by varying the radial distance of each point from the centre.

    Params:
    ctrX, ctrY - coordinates of the "centre" of the polygon
    aveRadius - in px, the average radius of this polygon, this roughly controls how large the polygon is, really only useful for order of magnitude.
    irregularity - [0,1] indicating how much variance there is in the angular spacing of vertices. [0,1] will map to [0, 2pi/numberOfVerts]
    spikeyness - [0,1] indicating how much variance there is in each vertex from the circle of radius aveRadius. [0,1] will map to [0, aveRadius]
    numVerts - self-explanatory

    Returns a list of vertices, in CCW order.
    '''

    irregularity = clip( irregularity, 0,1 ) * 2*math.pi / numVerts
    spikeyness = clip( spikeyness, 0,1 ) * aveRadius

    # generate n angle steps
    angleSteps = []
    lower = (2*math.pi / numVerts) - irregularity
    upper = (2*math.pi / numVerts) + irregularity
    sum = 0
    for i in range(numVerts) :
        tmp = random.uniform(lower, upper)
        angleSteps.append( tmp )
        sum = sum + tmp
    # normalize the steps so that point 0 and point n+1 are the same
    k = sum / (2*math.pi)
    for i in range(numVerts) :
        angleSteps[i] = angleSteps[i] / k

    # now generate the points
    points = []
    angle = random.uniform(0, 2*math.pi)
    for i in range(numVerts) :
        r_i = clip( random.gauss(aveRadius, spikeyness), 0, 2*aveRadius )
        x = ctrX + r_i*math.cos(angle)
        y = ctrY + r_i*math.sin(angle)
        points.append( (int(x),int(y)) )

        angle = angle + angleSteps[i]

    return points

# for i in range(100):
#     pred = Polygon(generatePolygon( 0, 0, 50, 0.5, 0.5, 5 ))
#     gt = Polygon(generatePolygon( 0, 0, 50, 0.5, 0.5, 4 ))
#     polygroup = MultiPolygon([pred, gt])
#     shape_sim = distance_turning(turning(pred), turning(gt))
# 
#     plt.clf()
#     plot_turning(turning(pred), 'g')
#     plt.hold(True)
#     plot_turning(turning(gt), 'r')
#     plt.title('Shape Similarity: %.2f' % shape_sim)
#     plt.savefig('shape_sim%d.png' % i)
# 
#     plt.figure()
#     ax = plt.gca()
#     ax.add_patch(matplotlib.patches.Polygon(np.array(pred.exterior.coords[:]), True, facecolor = 'r', alpha = 0.6, linewidth = 1))
#     ax.add_patch(matplotlib.patches.Polygon(np.array(gt.exterior.coords[:]), True, facecolor = 'g', alpha = 0.6, linewidth = 2))
#     ax.set_xlim(-100, 100)
#     ax.set_ylim(-100, 100)
#     ax.set_aspect(1.0)
#     plt.grid(True)
#     plt.xlabel('x')
#     plt.ylabel('y')
#     plt.title('IOU: %.2f' % iou(pred, gt))
#     plt.savefig('iou_%d.png' % i)
