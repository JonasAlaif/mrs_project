import xml.etree.ElementTree as ET
import numpy as np
import itertools

import struct
import os
from argparse import ArgumentParser

X = 0
Y = 1

def boxOccupied(box, positions):
    boxDims = box.find('size').text.split()
    return np.array([abs(pos[X]) <= float(boxDims[0]) / 2 and abs(pos[Y]) <= float(boxDims[1]) / 2 for pos in positions])

def cylinderOccupied(cylinder, positions):
    return False

def parse_point(point):
    pointPos = point.text.split()
    return np.array([float(pointPos[0]), float(pointPos[1])])

def compose_lr_of_line(p1, p2, my_pos, curr_lr):
    if (curr_lr > 2):
        return curr_lr
    line = p2 - p1
    new_lr = np.sign(np.dot(my_pos - p1, np.array([line[Y], -line[X]])))
    if (curr_lr == 0):
        return new_lr
    if (new_lr == curr_lr):
        return curr_lr
    else:
        return 3

def polylineOccupied(polyline, positions):
    all_points = polyline.findall('point')
    curr_point = parse_point(all_points[-1])
    pos_lrs = np.zeros(positions.shape[0])
    for point in all_points:
        new_point = parse_point(point)
        pos_lrs = np.array([compose_lr_of_line(curr_point, new_point, positions[i], pos_lrs[i]) for i in range(0, positions.shape[0])])
        curr_point = new_point
    return np.array([lrs_pos < 2 for lrs_pos in pos_lrs])

# map the inputs to the function blocks
handleShape = {
    'box' : boxOccupied,
    'cylinder': cylinderOccupied,
    'polyline': polylineOccupied
}

def getFrame(elem):
    refFrame = elem.find('pose')
    if(refFrame == None):
        return np.zeros(2, np.float32)
    else:
        refFrameValues = refFrame.text.split()
        return np.array((float(refFrameValues[0]), float(refFrameValues[1])))

def areOccupied(models, coordsToCheck):
    coordsChecked = np.zeros(coordsToCheck.shape[0], dtype=bool)
    for model in models:
        if(model.tag != 'model'):
            continue
        coordsChecked = np.logical_or(coordsChecked, isOccupied(model, coordsToCheck))
    return coordsChecked

def isOccupied(model, coordsToCheck):
    coordsChecked = np.zeros(coordsToCheck.shape[0], dtype=bool)
    refFrame = getFrame(model)

    for link in model:
        if(link.tag != 'link'):
            continue
        linkFrame = refFrame + getFrame(link)
        collision = link.find('collision')
        if(collision == None):
            continue
        poseFrame = linkFrame + getFrame(collision)
        geom = collision.find('geometry')
        for shape in geom:
            coordsInRelFrame = np.subtract(coordsToCheck, poseFrame)
            coordsCurrOccupied = handleShape[shape.tag](shape, coordsInRelFrame)
            coordsChecked = np.logical_or(coordsChecked, coordsCurrOccupied)
    return coordsChecked





def main():
    parser = ArgumentParser()
    parser.add_argument("-f", "--file", dest="filename",
                        help="XML map file (.sdf)", default="model.sdf")
    parser.add_argument("-o", "--out", dest="output",
                        help="File name to output (defualt map)", default="map_city_3")
    parser.add_argument("-d", "--dims", dest="dimensions",
                        help="Width and Height of map to display", default=9.5)
    parser.add_argument("-r", "--res", dest="resolution",
                        help="Resolution in pixels to display", default=300)
    args = parser.parse_args()
    if args.filename == None:
        print "No input file provided, use -f to choose the .sdf map file!"
        return
    root = ET.parse(args.filename).getroot()

    # define the width  (columns) and height (rows) of your image
    dims = float(args.dimensions)
    res = int(args.resolution)
    width = np.linspace(-dims, dims, num=res)
    height = np.linspace(-dims, dims, num=res)

    coords = np.array(list(itertools.product(width, height)))
    occupancy = areOccupied(root, coords)

    # open file for writing 
    fout=open(args.output + '.pgm', 'wb')

    # define PGM Header
    pgmHeader = 'P5' + '\n' + str(width.size) + ' ' + str(height.size) + ' ' + str(255) +  '\n'

    pgmHeader_byte = bytearray(pgmHeader,'utf-8')

    # write the header to the file
    fout.write(pgmHeader_byte)

    for occupied in np.nditer(occupancy):
        fout.write(struct.pack(">B", (not occupied)*255))

    fout.close()

    # open file for writing 
    fout=open(args.output + '.yaml', 'wb')

    # write the header to the file
    image = 'image: ./' + args.output + '.pgm\n'
    resolution = 'resolution: ' + str(2.0 * dims / res) + '\n'
    origin = 'origin: [' + str(-dims) + ', ' + str(-dims) + ', 0.0]\n'
    negate = 'negate: 0\n'
    occ = 'occupied_thresh: 0.65\n'
    free = 'free_thresh: 0.196\n'
    config = image + resolution + origin + negate + occ + free
    fout.write(config)

    fout.close()


if __name__ == '__main__':
    main()
