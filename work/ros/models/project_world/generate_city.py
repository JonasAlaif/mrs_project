from xml.etree import ElementTree
from xml.etree.ElementTree import Element, SubElement, Comment
from xml.dom import minidom

import os

def prettify(elem):
    """Return a pretty-printed XML string for the Element.
    """
    rough_string = ElementTree.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")


def make_link(parent, name, pose):
    link = SubElement(parent, 'link')
    link.set('name', name)
    col = SubElement(link, 'collision')
    col.set('name', name + "_Collision")
    vis = SubElement(link, 'visual')
    vis.set('name', name + "_Visual")
    pose_total = SubElement(link, 'pose')
    pose_total.set('frame', '')
    pose_total.text = pose

    geom1 = SubElement(col, 'geometry')
    pose_col = SubElement(col, 'pose')
    pose_col.set('frame', '')
    pose_col.text = "0 0 0 0 -0 0"

    pose_vis = SubElement(vis, 'pose')
    pose_vis.set('frame', '')
    pose_vis.text = "0 0 0 0 -0 0"
    geom2 = SubElement(vis, 'geometry')
    mat = SubElement(vis, 'material')
    amb = SubElement(mat, 'ambient')
    amb.text = "1 1 1 1"

    return geom1, geom2

def set_box_geom(geom1, geom2, size):
    box1 = SubElement(geom1, 'box')
    box2 = SubElement(geom2, 'box')
    size1 = SubElement(box1, 'size')
    size2 = SubElement(box2, 'size')
    size1.text = size
    size2.text = size

def set_triangle_geom(geom1, geom2, c1, c2, c3, height):
    pl1 = SubElement(geom1, 'polyline')
    pl2 = SubElement(geom2, 'polyline')
    p1_1 = SubElement(pl1, 'point')
    p1_2 = SubElement(pl2, 'point')
    p1_1.text = c1
    p1_2.text = c1
    p2_1 = SubElement(pl1, 'point')
    p2_2 = SubElement(pl2, 'point')
    p2_1.text = c2
    p2_2.text = c2
    p3_1 = SubElement(pl1, 'point')
    p3_2 = SubElement(pl2, 'point')
    p3_1.text = c3
    p3_2.text = c3
    h_1 = SubElement(pl1, 'height')
    h_2 = SubElement(pl2, 'height')
    h_1.text = height
    h_2.text = height

CITY_SIZE = 8
STREET_WIDTH = 1
BLOCK_WIDTH = 2
WHOLE_BLOCK_WIDTH = BLOCK_WIDTH + STREET_WIDTH

WALL_WIDTH = 1
WALL_HEIGHT = 0.5
X = 0
Y = 1

def make_walls(model):
    for x in range (0, 2):
        for y in range (0, 2):
            x_pos = (CITY_SIZE + WALL_WIDTH * 0.5) * ((x + y) - 1)
            y_pos = (CITY_SIZE + WALL_WIDTH * 0.5) * (x - y)
            if x == 1 and y == 1:
                geom_col, geom_vis = make_link(model, "Wall_" + str(x) + "_" + str(y) + "_1", str(x_pos) + " " + str(- (CITY_SIZE + 2 * WALL_WIDTH) / 2.0) + " " + str(WALL_HEIGHT/2) + " 0 -0 0")
                set_box_geom(geom_col, geom_vis, str(WALL_WIDTH) + " " + str(abs(x_pos) - WALL_WIDTH) + " " + str(WALL_HEIGHT))
                geom_col, geom_vis = make_link(model, "Wall_" + str(x) + "_" + str(y) + "_2", str(x_pos) + " " + str((CITY_SIZE + 2 * WALL_WIDTH) / 2.0) + " " + str(WALL_HEIGHT/2) + " 0 -0 0")
                set_box_geom(geom_col, geom_vis, str(WALL_WIDTH) + " " + str(abs(x_pos) - WALL_WIDTH) + " " + str(WALL_HEIGHT))
            else:
                geom_col, geom_vis = make_link(model, "Wall_" + str(x) + "_" + str(y), str(x_pos) + " " + str(y_pos) + " " + str(WALL_HEIGHT/2) + " 0 -0 0")
                set_box_geom(geom_col, geom_vis, str(abs(2*y_pos) + WALL_WIDTH) + " " + str(abs(2*x_pos) + WALL_WIDTH) + " " + str(WALL_HEIGHT))

            x_pos = CITY_SIZE * (2 * x - 1)
            y_pos = CITY_SIZE * (2 * y - 1)
            geom_col, geom_vis = make_link(model, "Corner_" + str(x) + "_" + str(y), str(x_pos) + " " + str(y_pos) + " 0 0 -0 0")
            set_triangle_geom(geom_col, geom_vis, "0 0", str((1 - 2 * x) * 0.5) + " 0", "0 " + str((1 - 2 * y) * 0.5), str(WALL_HEIGHT))


def make_box(model, center, size):
    geom_col, geom_vis = make_link(model, "Box_" + str(center[X]) + "_" + str(center[Y]), str(center[X]) + " " + str(center[Y]) + " " + str(WALL_HEIGHT/2) + " 0 -0 0")
    set_box_geom(geom_col, geom_vis, str(size[X]) + " " + str(size[Y]) + " " + str(WALL_HEIGHT))

def make_city(model):
    number_of_blocks = int((2 * CITY_SIZE - STREET_WIDTH) / WHOLE_BLOCK_WIDTH)
    for x in range (0, number_of_blocks):
        for y in range (0, number_of_blocks):
            x_pos = (x * WHOLE_BLOCK_WIDTH + BLOCK_WIDTH / 2) - CITY_SIZE + STREET_WIDTH
            y_pos = (y * WHOLE_BLOCK_WIDTH + BLOCK_WIDTH / 2) - CITY_SIZE + STREET_WIDTH
            make_box(model, [x_pos, y_pos], [BLOCK_WIDTH, BLOCK_WIDTH])

root = Element('sdf')
root.set('version', '1.6')

comment = Comment('Generated City')
root.append(comment)

model = SubElement(root, 'model')
model.set('name', 'city')

pose1 = SubElement(model, 'pose')
pose1.set('frame', '')
pose1.text = '0 0 0 0 -0 0'

make_walls(model)
make_city(model)

static = SubElement(model, 'static')
static.text = '1'

final_xml = prettify(root)

model_file = open(os.path.dirname(os.path.abspath(__file__)) + "/model.sdf","w") 
model_file.write(final_xml) 
model_file.close() 

print final_xml
