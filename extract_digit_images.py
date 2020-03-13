#! /usr/bin/env python

import cv2
import numpy as np
from PIL import Image, ImageOps

nDigits = 9
line_detection_threshold = 0.99  # % of maximum horizontal/vertical sum value
line_remove_margin = 3 # pixles


def convert_im(img, conversion=None):
    im = img.copy()
    if conversion == 'grey':
        im = np.array(im)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)  # convert to grey
    if conversion == 'bw': # convert to B&W
        im = cv2.threshold(im, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]  # Apply threshold to get image with only b&w (binarization)
    if conversion == 'de': # Apply dilation and erosion to remove some noise
        kernel = np.ones((2, 2), np.uint8)
        im = cv2.erode(im, kernel, iterations=2)
        im = cv2.dilate(im, kernel, iterations=2)
        im = cv2.GaussianBlur(im, (3, 3), 1)  # Apply blur to smooth out the edges
        im = cv2.threshold(im, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]  # Apply threshold to get image with only b&w (binarization)
        im = cv2.GaussianBlur(im, (3, 3), 1)  # Apply blur to smooth out the edges
    return im

def remove_box(im):
    h, w = im.shape
    # remove 2 horizontal lines
    for i in range(2):
        s_h = np.sum(im,1)
        argmin = np.argmin(s_h)
        mask = range(argmin-line_remove_margin,argmin+line_remove_margin+1)
        im[mask,:] = 255

    # remove vertical lines
    s_v = np.sum(im,0)
    s_v_start=np.argmax(s_v<np.max(s_v)*line_detection_threshold)
    s_v_end=w-np.argmax(s_v[::-1]<np.max(s_v)*line_detection_threshold)
    for x in np.arange(s_v_start,s_v_end+1,(s_v_end-s_v_start)/nDigits):
        i = int(x)
        im[:,max(0,i-line_remove_margin):min(i+line_remove_margin+1,w+1)] = 255

# def find_digit_boxes(xy, w, h, nDigits = 9):
def find_digit_boxes(im):
    s_v = np.sum(im,0)
    non_empty_cols = np.argwhere(s_v<s_v.max()*line_detection_threshold).T[0]
    start = non_empty_cols[0]
    im_ranges = []
    for i in range(len(non_empty_cols)-1):
        if non_empty_cols[i+1] - non_empty_cols[i] > line_remove_margin:
            im_ranges.append(range(start,non_empty_cols[i]+1))
            start = non_empty_cols[i+1]
    im_ranges.append(range(start,non_empty_cols[-1]+1))
    return im_ranges

def get_im_segments(im, im_ranges):
    im_segments = []
    for r in im_ranges:
        s_h = np.sum(im[:,r],1)
        non_empty_rows = np.argwhere(s_h<s_h.max()*line_detection_threshold).T[0]
        x = im[non_empty_rows.min():non_empty_rows.max(),r]
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
        im_segments.append(x)
    return im_segments

def GetDigitImages(img):
    im = convert_im(img, conversion='grey')
    remove_box(im)
    im = convert_im(im, conversion='de')
    boxes = find_digit_boxes(im)
    return get_im_segments(im, boxes)
    
if __name__ == '__main__': # TEST
    img = Image.open('test.jpg')
    im_segments = GetDigitImages(img)

    import matplotlib.pyplot as plt

    plt.figure()
    for i in range(9):
        ax = plt.subplot(1,9,i+1)
        ax.axis('off')
        plt.imshow(im_segments[i])
    plt.show()