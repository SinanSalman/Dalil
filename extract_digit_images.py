#! /usr/bin/env python

import cv2
import numpy as np
from PIL import Image, ImageOps

nDigits = 9
line_detection_threshold = 0.99  # % of maximum horizontal/vertical sum value
line_remove_margin = 3 # pixles
np.set_printoptions(linewidth=200)

def GetDigitImages(im, ShowSteps=False):
    im = np.array(im)
    if ShowSteps: clean_progress = [('orginal',Image.fromarray(im).copy())]
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)  # convert to grey
    if ShowSteps: clean_progress.append(('gray',Image.fromarray(im).copy()))

    h, w = im.shape
    # remove 2 horizontal lines
    for _ in range(2):
        s_h = np.sum(im,1)
        argmin = np.argmin(s_h)
        mask = range(argmin-line_remove_margin,argmin+line_remove_margin+1)
        im[mask,:] = 255

    # ** does not work for gray box lines **
    # # remove nDigits+1 vertical lines
    # for _ in range(nDigits+1):
    #     s_v = np.sum(im,0)
    #     argmin = np.argmin(s_v)
    #     mask = range(argmin-line_remove_margin,argmin+line_remove_margin+1)
    #     print(mask)
    #     im[:,mask] = 255

    # remove vertical lines
    s_v = np.sum(im,0)
    s_v_start=np.argmax(s_v<np.max(s_v)*line_detection_threshold)
    s_v_end=w-np.argmax(s_v[::-1]<np.max(s_v)*line_detection_threshold)
    for x in np.arange(s_v_start,s_v_end+1,(s_v_end-s_v_start)/nDigits):
        i = int(x)
        im[:,max(0,i-line_remove_margin):min(i+line_remove_margin+1,w+1)] = 255
    if ShowSteps: clean_progress.append(('no border',Image.fromarray(im).copy()))

    # Apply dilation and erosion to remove some noise
    kernel = np.ones((2, 2), np.uint8)
    im = cv2.erode(im, kernel, iterations=2)
    if ShowSteps: clean_progress.append(('erode',Image.fromarray(im).copy()))
    im = cv2.dilate(im, kernel, iterations=2)
    if ShowSteps: clean_progress.append(('dilate',Image.fromarray(im).copy()))
    im = cv2.GaussianBlur(im, (3, 3), 1)  # Apply blur to smooth out the edges
    if ShowSteps: clean_progress.append(('blur',Image.fromarray(im).copy()))
    im = cv2.threshold(im, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]  # Apply threshold to get image with only b&w (binarization)
    if ShowSteps: clean_progress.append(('threshold',Image.fromarray(im).copy()))
    im = cv2.GaussianBlur(im, (3, 3), 1)  # Apply blur to smooth out the edges
    if ShowSteps: clean_progress.append(('blur',Image.fromarray(im).copy()))

    s_v = np.sum(im,0)
    non_empty_cols = np.argwhere(s_v<s_v.max()*line_detection_threshold).T[0]
    start = non_empty_cols[0]
    im_ranges = []
    for i in range(len(non_empty_cols)-1):
        if non_empty_cols[i+1] - non_empty_cols[i] > line_remove_margin:
            im_ranges.append(range(start,non_empty_cols[i]+1))
            start = non_empty_cols[i+1]
    im_ranges.append(range(start,non_empty_cols[-1]+1))

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
        if ShowSteps: print(x.shape,'\n',str(x/255).replace('.',''),'\n'*5)
        x = Image.fromarray(x)
        x = x.resize((20,20))
        x = ImageOps.expand(x,border=4,fill='white')
        im_segments.append(x)

    if ShowSteps: 
        import matplotlib.pyplot as plt

        s_v = np.sum(im,0)
        plt.plot(s_v)
        plt.show()

        n = len(clean_progress)
        plt.figure(figsize=(6,12))
        for i in range(n):
            plt.subplot(n, 1, i+1)
            plt.imshow(clean_progress[i][1], cmap=plt.cm.gray)
            plt.xlabel(clean_progress[i][0])
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(12,12))
        for i in range(9):
            plt.subplot(3, 3, i+1)
            plt.xticks(range(0,50,2))
            plt.yticks(range(0,50,2))
            plt.imshow(im_segments[i], cmap=plt.cm.binary)
            plt.title(f'Digit #{i}')
            plt.grid()
        plt.show()

    return im_segments
    

if __name__ == '__main__': # TEST
    img = Image.open('test.jpg')
    im_segments = GetDigitImages(img ,ShowSteps=False)

    import matplotlib.pyplot as plt

    plt.figure()
    for i in range(9):
        ax = plt.subplot(1,9,i+1)
        ax.axis('off')
        plt.imshow(im_segments[i])
    plt.show()

