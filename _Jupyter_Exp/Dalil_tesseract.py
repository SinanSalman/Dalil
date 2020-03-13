#! /usr/bin/env python

import PyPDF4 as pdf
import glob
import io
import os
import cv2
import numpy as np
import PIL
import pytesseract
import re
import readline

tesseract_path = r'/usr/local/Cellar/tesseract/4.0.0_1/bin/tesseract'
pytesseract.pytesseract.tesseract_cmd = tesseract_path
img_modes = {'/DeviceRGB': 'RGB', '/DefaultRGB': 'RGB', '/DeviceCMYK': 'CMYK', '/DefaultCMYK': 'CMYK',
             '/DeviceGray': 'L', '/DefaultGray': 'L', '/Indexed': 'P'}

def cleanID(im,line_detection_threshold):
#     display(PIL.Image.fromarray(im))
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)  # convert to grey
#     display(PIL.Image.fromarray(im))  

    # remove horizontal lines
    h,w = im.shape
    s = np.sum(im,1)
    s_start=np.argmax(s<np.max(s)*line_detection_threshold)
    s_end=h-np.argmax(s[::-1]<np.max(s)*line_detection_threshold)
#     print(f'h={h},w={w},s_start={s_start},s_end={s_end},s={s},np.max(s)={np.max(s)}')
    mask = int(h*0.04)
    im[:s_start+mask,:] = 255
    im[s_end-mask:,:] = 255
    # remove vertical lines
    s = np.sum(im,0)
    s_start=np.argmax(s<np.max(s)*line_detection_threshold)
    s_end=w-np.argmax(s[::-1]<np.max(s)*line_detection_threshold)
#     print(f'h={h},w={w},s_start={s_start},s_end={s_end},s={s},np.max(s)={np.max(s)}')
    mask = int(w*0.005)
    for x in np.arange(s_start,s_end+1,(s_end-s_start)/9):
        i = int(x)
        im[:,i-mask:i+mask] = 255
#     display(PIL.Image.fromarray(im))  

    # Apply dilation and erosion to remove some noise
    kernel = np.ones((2, 2), np.uint8)
    im = cv2.erode(im, kernel, iterations=2)
#     display(PIL.Image.fromarray(im))  
    im = cv2.dilate(im, kernel, iterations=2)
#     display(PIL.Image.fromarray(im))  
    im = cv2.GaussianBlur(im, (3, 3), 0)  # Apply blur to smooth out the edges
#     display(PIL.Image.fromarray(im))  
    im = cv2.threshold(im, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]  # Apply threshold to get image with only b&w (binarization)
    PIL.Image.fromarray(im).show() 
    return im

def Tesseract_ocrID(img, box,line_detection_threshold):
    w, h = img.size
    im = np.array(img.crop((box[0]*w,box[1]*h,box[2]*w,box[3]*h)))
    im = cleanID(im,line_detection_threshold)
    im = PIL.Image.fromarray(im)
    id = pytesseract.image_to_string(im,config="-l script/Arabic") # ,config="--psm 7 --oem 1"
    return re.sub("[^0-9]", "",id)

def get_img_from_page(page, pdfreader):
    """pull image from pdf page; 
    adapted from https://github.com/claird/PyPDF4/blob/master/scripts/pdf-image-extractor.py
    Other sources of info include: https://gist.github.com/gstorer/f6a9f1dfe41e8e64dcf58d07afa9ab2a and
    https://stackoverflow.com/questions/32192671/pil-image-mode-i-is-grayscale
    """
    if '/XObject' in page['/Resources']:
        xObject = page['/Resources']['/XObject'].getObject()
        for obj in xObject:
            if xObject[obj]['/Subtype'] == '/Image':
                size = (xObject[obj]['/Width'], xObject[obj]['/Height'])
                data = xObject[obj].getData()
                color = xObject[obj]['/ColorSpace']
                if type(color) != str:
                    color = pdfreader.getObject(xObject[obj]['/ColorSpace'][1])['/Alternate']
                mode = img_modes[color]
                if '/Filter' in xObject[obj]:
                    if xObject[obj]['/Filter'] == '/FlateDecode':
                        try:
                            img = PIL.Image.frombytes(mode, size, data)
                        except:
                            mode = '1'  # maybe its B&W?
                            img = PIL.Image.frombytes(mode, size, data)
                    elif xObject[obj]['/Filter'] == '/DCTDecode':
                        img = PIL.Image.open(io.BytesIO(data))
                    elif xObject[obj]['/Filter'] == '/JPXDecode':
                        img = PIL.Image.open(io.BytesIO(data))
                    elif xObject[obj]['/Filter'] == '/CCITTFaxDecode':
                        img = PIL.Image.open(io.BytesIO(data))
                else:
                    img = PIL.Image.frombytes(mode, size, data)
    else:
        print("No image found in page.")
    return img

def has_header(im):
    if len(im.shape) == 3:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)  # convert to grey
        m = 255
    else:
        m = 1
    h,w = im.shape
    s = np.sum(im)
#     print(im[0,0],s,(h*w*m),s/(h*w*m))
    if s/(h*w*m)<0.99:
        return True
    else:
        return False

def rlinput(prompt, prefill=''):
   readline.set_startup_hook(lambda: readline.insert_text(prefill))
   try:
      return input(prompt)  # or raw_input in Python 2
   finally:
      readline.set_startup_hook()

def merge(paths, output):
    pdf_writer = pdf.PdfFileWriter()
    for file in paths:
        pdf_reader = pdf.PdfFileReader(file)
        print(f'found {pdf_reader.getNumPages()} pages in {file}')
        for page in range(pdf_reader.getNumPages()):
            pdf_writer.addPage(pdf_reader.getPage(page))
    with open(output, 'wb') as out:
        pdf_writer.write(out)
    print(f'merged into {output}')

def split_by_count(file, base_name, pagecount):
    pdf_reader = pdf.PdfFileReader(file)
    pages = pdf_reader.getNumPages()
    n=0
    for first_page in range(0,pages,pagecount):
        pdf_reader = pdf.PdfFileReader(file)
        pdf_writer = pdf.PdfFileWriter()  # needed to avoid a bug in pyPDF4
        for i in range(pagecount):
            pdf_writer.addPage(pdf_reader.getPage(first_page+i))
        n += 1
        output = f'output/{base_name}{n:03d}.pdf'
        with open(output, 'wb') as output_pdf:
            pdf_writer.write(output_pdf)
        print(f'split {pagecount} pages into {output}')

def show_header(file, box, pagescount):
    pdf_reader = pdf.PdfFileReader(file)
    for i in range(pagescount):
        img = get_img_from_page(pdf_reader.getPage(i),pdf_reader)
        w, h = img.size
        img.crop((box[0]*w,box[1]*h,box[2]*w,box[3]*h)).show()

def split_by_header(file, base_name, box):
    pdf_reader = pdf.PdfFileReader(file)
    n = 0
    pageNo = 0
    doc = []
    while (pageNo < pdf_reader.numPages):
        page = pdf_reader.getPage(pageNo)
        img = get_img_from_page(page,pdf_reader)
        w, h = img.size
        im = np.array(img.crop((box[0]*w,box[1]*h,box[2]*w,box[3]*h)))
#         display(PIL.Image.fromarray(im))
        pagecount = len(doc)
        if has_header(im) and pagecount > 0:
            n += 1
            pdf_writer = pdf.PdfFileWriter()
            for p in doc:
                pdf_writer.addPage(p)
            output = f'output/{base_name}{n:03d}.pdf'
            with open(output, 'wb') as output_pdf:
                pdf_writer.write(output_pdf)
            print(f'split {pagecount} pages into {output}')
            doc = [page]
        else:
            doc.append(page)    
        pageNo += 1
    #  save last document
    n += 1
    pdf_writer = pdf.PdfFileWriter()
    for p in doc:
        pdf_writer.addPage(p)
    output = f'output/{base_name}{n:03d}.pdf'
    with open(output, 'wb') as output_pdf:
        pdf_writer.write(output_pdf)
    print(f'split {pagecount} pages into {output}')

def rename2id(paths,box,line_detection_threshold):
    """attempt to get student ID and rename pdf file using ID"""
    for file in paths:
        pdf_reader = pdf.PdfFileReader(file)
        img = get_img_from_page(pdf_reader.getPage(0),pdf_reader)
        id = Tesseract_ocrID(img,box,line_detection_threshold)
        id = rlinput('correct id? ',prefill=id)
        newfilename = os.path.join(os.path.split(file)[0],f'{id}.pdf')
        print(f'renaming {file} to {newfilename}')
        os.rename(file,newfilename)    

if __name__ == '__main__':
    inputfolder = '/Users/sinan/OneDrive/ZU/BlackBoard/examagic/input/*.pdf'
    outputfolder = '/Users/sinan/OneDrive/ZU/BlackBoard/examagic/output/*.pdf'

    show_header(file='merged-1.pdf', box=(0.33,0.1,0.66,0.17), pagescount = 18)  # for Susu's exam

    # merge(paths=glob.glob(inputfolder), output='merged.pdf')
    # split_by_count(file='merged.pdf', base_name='exam', pagecount=8)
    # split_by_header(file='merged.pdf', base_name='exam', box=(0.08,0.03,0.3,0.06))  # for Sinan's exams
    # rename2id(paths=glob.glob(outputfolder), box=(0.685,0.077,0.93,0.13), line_detection_threshold=0.99)   # for Sinan's exams
    print('Done')
    
