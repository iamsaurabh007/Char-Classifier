import cv2
import numpy as np


def get_energy_density(image_path):
    
    image   = cv2.imread(image_path,0)
    #height, width = image.shape[:2]
    #image = cv2.resize(image, (2*width, 1*height))
    binary  = cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
    #laplacian = cv2.Laplacian(binary.copy(),cv2.CV_32F)
    #laplacian = np.uint8(laplacian)
    distance_transform =cv2.distanceTransform(binary.copy(), distanceType=cv2.DIST_L2, maskSize=5)
    
    #energy_density = 1 /  np.log(distance_transform + np.exp(1))
    energy_density = 1 /  (distance_transform + 1)**2
    
    return binary

def get_equilibrium_delta(boundry,energy_density ,axis,flag):

    # y is axis 1, x is axis 0    
    boundry_energy  = energy_density[int(boundry[0]) : int(boundry[3]) , int(boundry[1]) : int(boundry[2])]
    inital_boundry = boundry_energy.shape[1 - axis]  * 0.5
    p=boundry_energy.sum(axis=axis)[::-1]
    if flag=='l':
        delta=inital_boundry-(p.shape[0]-1-np.argmax(p))
    else:
        delta = inital_boundry - np.argmax(boundry_energy.sum(axis=axis))
    #print(inital_boundry)
    #print(delta)
    return delta
def correct_region(region,energy_density,rat):
    image_height  = energy_density.shape[0]
    image_widht=energy_density.shape[1]
    text=region['text']
    iou=region['iou']
    box = region['input']['boundingBox']["vertices"]

    box_height =  box[3]['y'] - box[0]['y']
    box_widht  =  box[1]['x'] - box[0]['x']
    box_left   =  box[0]['x']
    box_right  =  box[1]['x']
    box_top    =  box[0]['y']
    box_bottom =  box[3]['y']


    #order : top, left, right, bottom
    
    #boundry_top    = [ max(box_top - box_height * 0.5 ,0), box_left ,box_right ,box_top + box_height * 0.5]
    #boundry_bottom = [ box_bottom - box_height * 0.5 , box_left ,box_right ,min(box_bottom + box_height * 0.5,image_height)]
    
    boundry_left=[box_top, max(box_left - box_widht * rat ,0),box_left+box_widht*rat, box_bottom]
    boundry_right = [ box_top , box_right-box_widht*rat,min(box_right + box_widht*rat,image_widht),box_bottom]
    
    #top_delta    = get_equilibrium_delta(boundry_top , energy_density,axis=1)
    #bottom_delta = get_equilibrium_delta(boundry_bottom, energy_density,axis=1)
    
    left_delta    = get_equilibrium_delta(boundry_left , energy_density,axis=0,flag='l')
    right_delta = get_equilibrium_delta(boundry_right, energy_density,axis=0,flag='r')
    #print(left_delta, right_delta)

    return {'text':text,'iou':iou,'input': {'boundingBox':{'vertices'  : [{'x':int(box_left-left_delta),'y':box_top},\
                                                        {'x':int(box_right-right_delta),'y':box_top},\
                                                        {'x':int(box_right-right_delta),'y':box_bottom},\
                                                        {'x':int(box_left-left_delta),'y':box_bottom}]}},
            'ground':{"text":text,'boundingBox':{'vertices'  : [{'x':int(box_left-left_delta),'y':box_top},\
                                                        {'x':int(box_right-right_delta),'y':box_top},\
                                                        {'x':int(box_right-right_delta),'y':box_bottom},\
                                                        {'x':int(box_left-left_delta),'y':box_bottom}]}}}

def get_corrected_regions(regions,energy_density,rat):
    corrected_regions = {}
    corrected_regions['iou']=[]
    for region in regions['iou'] :
            #print(region)
            if not (region['ground'] and region['input']):
                continue
            corrected_regions['iou'].append(correct_region(region,energy_density,rat))
    return corrected_regions

def bounds_refine(bounds,imgpath,rat):
    energy_density = get_energy_density(imgpath)
    corrected_regions = get_corrected_regions(bounds,energy_density,rat)
    return corrected_regions
