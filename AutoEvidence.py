#! /usr/bin/env python

# ToDo
# GUI for checking work
# ID box identification when it moves or written over

import os
import io
import cv2
import PIL
import click
import json
import numpy as np
import PyPDF4 as pdf
from extract_digit_images import GetDigitImages

print(f'Loading config file...')
config = json.load(open('AutoEvidence.cfg','r'))
ID_box = config['ID_box']
Header_box = config['Header_box']
header_detection_threshold = config['header_detection_threshold']

header_detection_threshold = 0.99
img_modes = {'/DeviceRGB': 'RGB', '/DefaultRGB': 'RGB', '/DeviceCMYK': 'CMYK', '/DefaultCMYK': 'CMYK',
             '/DeviceGray': 'L', '/DefaultGray': 'L', '/Indexed': 'P'}


def ExtractID(img, model):
    w, h = img.size
    img_segs = GetDigitImages(img.crop((ID_box[0]*w,ID_box[1]*h,ID_box[2]*w,ID_box[3]*h)))
    img_segs = np.array([np.array(x) for x in img_segs])
    n = int(img_segs.size/(28*28))
    img_segs = 1 - img_segs.reshape(n, 28, 28, 1).astype('float32') / 255.0  # invert and convert to max = 1.0
    prediction = model.predict(img_segs)
    if n>9:
        p = np.amax(prediction,axis=1)
        # print(f'found more digits than 9, dropping the digits with the least confidence from:\n{p}')
    while prediction.shape[0] > 9:
        p = np.amax(prediction,axis=1)
        i = p.argmin()
        prediction = np.delete(prediction,i,axis=0)
        # print(np.amax(prediction,axis=1))
    id = [str(x) if x<10 else 'M' for x in prediction.argmax(axis=1)]
    return ''.join(id)


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
                if type(color) != pdf.generic.NameObject:
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


def has_header(im, chksum = 0):
    if len(im.shape) == 3:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)  # convert to grey
        m = 255
    else:
        m = 1
    h,w = im.shape
    s = np.sum(im) / (h*w*m)
    if chksum == 0:
        return s
    else:
        if s/chksum > header_detection_threshold and s/chksum < (2-header_detection_threshold):
            return True
        else:
            return False


@click.group()
def main():
    """AutoEvidence is a automation tool for blackboard exam evidence"""


@main.command('show_h')
@click.argument('file', type=click.Path(exists=True))
@click.option('-c', '--count', type=click.INT, default=1)
def show_header(file, count):
    """show header in pdf file by page"""
    pdf_reader = pdf.PdfFileReader(file)
    for i in range(count):
        img = get_img_from_page(pdf_reader.getPage(i), pdf_reader)
        w, h = img.size
        img.crop((Header_box[0]*w, Header_box[1]*h,Header_box[2]*w, Header_box[3]*h)).show()


@main.command('show_id')
@click.argument('file', type=click.Path(exists=True))
@click.option('-c', '--count', type=click.INT, default=1)
def show_header(file, count):
    """show ID in pdf file by page"""
    pdf_reader = pdf.PdfFileReader(file)
    for i in range(count):
        img = get_img_from_page(pdf_reader.getPage(i), pdf_reader)
        w, h = img.size
        img.crop((ID_box[0]*w, ID_box[1]*h,ID_box[2]*w, ID_box[3]*h)).show()


@main.command('merge')
@click.argument('files', nargs=-1, type=click.Path())
@click.option('-o', '--out', default='merged.pdf')
def merge(files,out):
    """merge pdf files"""
    pdf_writer = pdf.PdfFileWriter()
    n = 0
    for file in files:
        pdf_reader = pdf.PdfFileReader(file)
        print(f'found {pdf_reader.getNumPages()} pages in {file}')
        for page in range(pdf_reader.getNumPages()):
            pdf_writer.addPage(pdf_reader.getPage(page))
        n += 1
    if n>0:
        with open(out, 'wb') as outfile:
            pdf_writer.write(outfile)
        print(f'merged {n} files into {out}')


@main.command('split_c')
@click.argument('file', type=click.Path(exists=True))
@click.option('-c', '--count', type=click.INT, required=True)
def split_by_count(file, count):
    """split pdf file by page count"""
    base = click.format_filename(file)[:-4]  # lose the ext
    pdf_reader = pdf.PdfFileReader(file)
    pages = pdf_reader.getNumPages()
    n=0
    for first_page in range(0, pages, count):
        pdf_reader = pdf.PdfFileReader(file)
        pdf_writer = pdf.PdfFileWriter()  # needed to avoid a bug in pyPDF4
        for i in range(count):
            pdf_writer.addPage(pdf_reader.getPage(first_page+i))
        n += 1
        output = f'output/{base}{n:03d}.pdf'
        with open(output, 'wb') as output_pdf:
            pdf_writer.write(output_pdf)
        print(f'created {output}')


@main.command('split_h')
@click.argument('file', type=click.Path(exists=True))
def split_by_header(file):
    """split pdf file by header detection"""
    base = os.path.basename(click.format_filename(file))[:-4]
    pdf_reader = pdf.PdfFileReader(file)
    n = 0
    pageNo = 0
    doc = []
    chksum = 0
    while (pageNo < pdf_reader.numPages):
        page = pdf_reader.getPage(pageNo)
        img = get_img_from_page(page,pdf_reader)
        w, h = img.size
        im = np.array(img.crop((Header_box[0]*w,Header_box[1]*h,Header_box[2]*w,Header_box[3]*h)))
        pagecount = len(doc)
        if chksum == 0:
            chksum = has_header(im)
        if has_header(im,chksum) and pagecount > 0:
            n += 1
            pdf_writer = pdf.PdfFileWriter()
            for p in doc:
                pdf_writer.addPage(p)
            output = f'output/{base}{n:03d}.pdf'
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
    output = f'output/{base}{n:03d}.pdf'
    with open(output, 'wb') as output_pdf:
        pdf_writer.write(output_pdf)
    print(f'split {pagecount+1} pages into {output}')


@main.command('id')
@click.argument('files', nargs=-1, type=click.Path(exists=True))
def rename2id(files):
    """rename pdf file using read student ID"""
    print(f'Loading tensorflow model...')
    from tensorflow.keras.models import load_model
    model = load_model('model.h5')
    for file in files:
        pdf_reader = pdf.PdfFileReader(file)
        img = get_img_from_page(pdf_reader.getPage(0),pdf_reader)
        id = ExtractID(img, model)
        newfilename = os.path.join(os.path.split(file)[0],f'{id}.pdf')
        if not os.path.exists(newfilename):
            print(f'renaming {file} to {newfilename}')
            os.rename(file,newfilename)
        else:
            print(f'** Warning: skiped {file} as it already exists')


@main.command('test')
@click.argument('files', nargs=-1, type=click.Path(exists=True))
def test(files):
    """test pdf file IDs"""
    print(f'Loading tensorflow model...')
    from tensorflow.keras.models import load_model
    model = load_model('model.h5')
    p = 0
    f = 0
    for file in files:
        pdf_reader = pdf.PdfFileReader(file)
        img = get_img_from_page(pdf_reader.getPage(0),pdf_reader)
        id = ExtractID(img, model)
        newfilename = os.path.join(os.path.split(file)[0],f'{id}.pdf')
        if os.path.exists(newfilename):
            # print(f'Matched: {file}')
            p += 1
        else:
            print(f'DID NOT Matched: {file}, OCR:{id}')
            f += 1
    print(f'\nTest success rate:{p/(p+f):5.1%}')


if __name__ == '__main__':
    main()
