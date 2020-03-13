#! /usr/bin/env python

import cv2
import numpy as np
from PIL import Image, ImageOps

noink_detection_threshold = 0.95
line_remove_margin = 4 # pixles

def convert_im(img, conversion=None):
    """Convert img to numpy array, and optionally to grey color pallette or perform image cleanup using dilation and erosion."""
    im = img.copy()
    if conversion == 'grey':
        im = np.array(im)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)  # convert to grey
    if conversion == 'de': # Apply dilation and erosion to remove some noise
        box = (7,7)
        kernel = np.ones((2,2), np.uint8)
        im = cv2.erode(im, kernel, iterations=2)
        im = cv2.dilate(im, kernel, iterations=2)
        im = cv2.GaussianBlur(im, box, 1)  # Apply blur to smooth out the edges
        im = cv2.threshold(im, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]  # Apply threshold to get image with only b&w (binarization)
        im = cv2.GaussianBlur(im, box, 1)  # Apply blur to smooth out the edges
    return im
    
def find_box(im):
    """Find ID box with the longest horizontal lines"""
    h, w = im.shape
    X = [0,0]
    Y = [0,0]
    s = np.sum(im,1)
    Y[0] = np.argmin(s)
    s[Y[0]-line_remove_margin:Y[0]+line_remove_margin] = 255*w
    Y[1] = np.argmin(s)
    Y.sort()
    line = np.min(im[Y[0]-line_remove_margin:Y[0]+line_remove_margin,:],axis=0) # addresses slightly tilted boxes by looking within the 'line_remove_margin' neibourhood
    non_empty_px = np.argwhere(line<127).T[0]
    diff = np.diff(non_empty_px)
    gaps2 = np.argwhere(diff>line_remove_margin).T[0]
    gaps1 = gaps2 + 1
    starts = np.append(non_empty_px[gaps1],np.min(non_empty_px))
    ends = np.append(non_empty_px[gaps2],np.max(non_empty_px))
    starts.sort()
    ends.sort()
    lines = [(s,e,e-s) for (s,e) in list(zip(starts,ends))]
    lines.sort(key=lambda x:x[2])
    X[:] = lines[-1][:2]
    # assert X[1]>X[0] and Y[1]>Y[0],f'Couldn\'t find digits box, perhaps the lines are too faint (find_box returned X[1]={X[1]}, X[0]={X[0]}, Y[1]={Y[1]}, Y[0]={Y[0]})'
    if X[1]<=X[0] or Y[1]<=Y[0]: raise ValueError
    return (X[0], Y[0]), X[1]-X[0]-1, Y[1]-Y[0]-1

def find_digit_boxes(xy, w, h, nDigits = 9):
    """Find digit boxes in the ID box"""
    step = w/nDigits
    W = int(step) - 2*line_remove_margin
    H = h - 2*line_remove_margin
    # assert W>0 and H>0 and step>0,f'Error in finding digit boxes, find_digit_boxes W={W}, H={H}, step={step}'
    if W<=0 or H<=0 or step<=0: raise ValueError
    return [ ( (xy[0]+int(round(i*step))+line_remove_margin,xy[1]+line_remove_margin),W,H) for i in range(nDigits)]

def resize_digit(im):
    """Resize and center digit image to fit in 20x20 pixles with a margin of 4 pixles. final 24x24"""
    s = np.sum(im,0)
    non_empty_cols = np.argwhere(s<s.max()*noink_detection_threshold).T[0]
    x = im[:,non_empty_cols]
    s = np.sum(x,1)
    non_empty_rows = np.argwhere(s<s.max()*noink_detection_threshold).T[0]
    x = x[non_empty_rows,:]
    h, w = x.shape
    d = max(h,w)
    tmp = np.ones((d,d)) * 255
    if h<d:
        diff = int((d-h)/2)
        tmp[diff:h+diff,:] = x[:,:] 
        x = tmp
    elif w<d:
        diff = int((d-w)/2)
        tmp[:,diff:w+diff] = x[:,:]
        x = tmp
    else:  # already a square, just make it of type float
        x = np.array(x,dtype='float')
    x = Image.fromarray(x)
    x = x.resize((20,20))
    x = ImageOps.expand(x,border=4,fill='white')
    return x

def get_im_segments(img, boxes):
    """Get a list of final processed digit images"""
    return [resize_digit(convert_im( img[ b[0][1]:b[0][1]+b[2], b[0][0]:b[0][0]+b[1] ] ,conversion='de')) for b in boxes]

def GetDigitImages(img):
    """Get a list of digit images; main function to pull all previous function together"""
    im = convert_im(img, conversion='grey')
    b_xy,b_w,b_h = find_box(im)
    boxes = find_digit_boxes(b_xy,b_w,b_h,9)    
    return get_im_segments(im, boxes)
    
if __name__ == '__main__': # TEST
    img = Image.open('./_Jupyter_Exp/test.jpg')
    im_segments = GetDigitImages(img)

    import matplotlib.pyplot as plt

    plt.figure()
    for i in range(9):
        ax = plt.subplot(1,9,i+1)
        ax.axis('off')
        plt.imshow(im_segments[i])
    plt.show()