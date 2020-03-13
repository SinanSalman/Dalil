#! /usr/bin/env python

# ToDo
#   GUI for checking work

import os
import io
import cv2
import PIL
import click
import json
import numpy as np
import PyPDF4 as pdf
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import glob
import pandas as pd
import difflib
import extract_digit_images as edi

print(f'Loading config file...')
config = json.load(open('Dalil.cfg','r'))
ID_box = config['ID_box']
Header_box = config['Header_box']
header_detection_threshold = config['header_detection_threshold']
ratio_threshold = config['ratio_threshold']
img_modes = {'/DeviceRGB': 'RGB', '/DefaultRGB': 'RGB', '/DeviceCMYK': 'CMYK', '/DefaultCMYK': 'CMYK',
             '/DeviceGray': 'L', '/DefaultGray': 'L', '/Indexed': 'P'}


def ocr_files(files):
    """run a list of PDF files in OCR using the first page"""
    print(f'Loading tensorflow model...')
    from tensorflow.keras.models import load_model
    model = load_model('model.h5')
    print(f'OCR in progress...\n')
    ID_list = []
    for filename in files:
        pdf_reader = pdf.PdfFileReader(filename)
        img = get_img_from_page(pdf_reader.getPage(0),pdf_reader)
        ID_list.append(ExtractID(img, model))
    return pd.DataFrame({'ID':ID_list, 'File':files})


def ExtractID(img, model):
    """ Extract student ID from ID box in image"""
    w, h = img.size
    img_segs = edi.GetDigitImages(img.crop((ID_box[0]*w,ID_box[1]*h,ID_box[2]*w,ID_box[3]*h)))
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
    s_id = [str(x) if x<10 else 'M' for x in prediction.argmax(axis=1)]
    return ''.join(s_id)


def get_img_from_page(page, pdfreader):
    """
    pull image from pdf page; 
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
    """confirm that a page had a header that matches a provided checksum"""
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


def load_crossref():
    """Load cross reference files; downloaded from Blackboard grade center"""
    df = []
    for f in glob.glob(config['CrossRef']):
        df.append(pd.read_csv(f,encoding='utf_16',sep='\t',usecols=[0,1,3]))
        print(f'using crossref: {f}')
    if df == []:
        return None
    else:
        df = pd.concat(df)
        df['ID'] = df['Student ID'].astype(str).map(str.upper)
        df['Name'] = df['First Name'] + ' ' + df['Last Name'].map(lambda x: x.split()[-1])
        df = df.drop(['First Name','Last Name','Student ID'], axis=1)
        df = df.drop_duplicates(subset='ID')
        return df.reset_index(drop=True)


def crossref(results):
    """cross reference OCR results with Blackboard grade center data to fix errors and add student name (optionally)"""
    xr = load_crossref()
    results['MatchRatio'] = 0
    results['OldID'] = ''
    results['Name'] = ''
    if xr is not None:
        results.loc[results.ID.isin(xr.ID),'MatchRatio'] = 1
        results = results.merge(xr,how='left',on='ID',suffixes=('_x','')).drop(['Name_x'],axis=1)
        xr = xr.drop(xr[xr.ID.isin(results.ID)].index)
        xr_df = pd.DataFrame(columns=results[results.MatchRatio < 1].ID, index=xr.ID)
        for index, row in results[results.MatchRatio < 1].iterrows():
            xr_df[row.ID] = [difflib.SequenceMatcher(None, row.ID, x).ratio() for x in xr.ID]
        for s_id, maxratio in xr_df.max().sort_values(ascending=False).items():
            newID = xr_df[s_id].idxmax()
            ratio = xr_df[s_id].max()
            if ratio >= ratio_threshold:
                results.loc[results.ID==s_id,'MatchRatio'] = ratio
                results.loc[results.ID==s_id,'Name'] = xr[xr.ID==newID].Name.iloc[0]
                results.loc[results.ID==s_id,'OldID'] = s_id
                results.loc[results.ID==s_id,'ID'] = newID
                xr_df = xr_df.drop(newID)
    return results[['ID','OldID','MatchRatio','Name','File']]


@click.group()
def main():
    """
    Dalil is a tool for managing student paper exam electronic evidences; it merges and splits PDF 
    scans of student exams, perform OCR of student IDs, cross-reference it with Blackboard grade center 
    downloads to rename files with student IDs and names. The word Dalil is Arabic for Evidence.
    """


@main.command('show_h')
@click.argument('filename', type=click.Path(exists=True))
@click.option('-c', '--count', type=click.INT, default=1)
def show_header(filename, count):
    """show header in pdf file by page"""
    pdf_reader = pdf.PdfFileReader(filename)
    for i in range(count):
        img = get_img_from_page(pdf_reader.getPage(i), pdf_reader)
        w, h = img.size
        img.crop((Header_box[0]*w, Header_box[1]*h,Header_box[2]*w, Header_box[3]*h)).show()


@main.command('show_id')
@click.argument('filename', type=click.Path(exists=True))
@click.option('-c', '--count', type=click.INT, default=1)
def show_header(filename, count):
    """show ID in pdf file by page"""
    pdf_reader = pdf.PdfFileReader(filename)
    for i in range(count):
        try:
            img = get_img_from_page(pdf_reader.getPage(i), pdf_reader)
            w, h = img.size
            img = img.crop((ID_box[0]*w, ID_box[1]*h,ID_box[2]*w, ID_box[3]*h))
            im = edi.convert_im(img, conversion='grey')
            b_xy,b_w,b_h = edi.find_box(im)
            boxes = edi.find_digit_boxes(b_xy,b_w,b_h,9)
            # fg = plt.figure()
            ax = plt.subplot(1, 1, 1)
            # ax.axis('off')
            ax.imshow(img)
            rect = patches.Rectangle(b_xy,b_w,b_h,linewidth=1,edgecolor='lime',facecolor='none')
            ax.add_patch(rect)
            for b in boxes:
                rect = patches.Rectangle(b[0],b[1],b[2],linewidth=1,edgecolor='red',facecolor='none')
                ax.add_patch(rect)
            plt.show()
        except:
            print(f'Couldn\'t find ID area in page {i+1}')


@main.command('merge')
@click.argument('filenames', nargs=-1, type=click.Path())
@click.option('-o', '--out', default='merged.pdf')
def merge(filenames,out):
    """merge pdf files"""
    pdf_writer = pdf.PdfFileWriter()
    n = 0
    for filename in filenames:
        pdf_reader = pdf.PdfFileReader(filename)
        print(f'found {pdf_reader.getNumPages()} pages in {filename}')
        for page in range(pdf_reader.getNumPages()):
            pdf_writer.addPage(pdf_reader.getPage(page))
        n += 1
    if n>0:
        with open(out, 'wb') as outfilename:
            pdf_writer.write(outfilename)
        print(f'merged {n} files into {out}')


@main.command('split_c')
@click.argument('filename', type=click.Path(exists=True))
@click.option('-c', '--count', type=click.INT, required=True)
def split_by_count(filename, count):
    """split pdf file by page count"""
    base = click.format_filename(filename)[:-4]  # lose the ext
    pdf_reader = pdf.PdfFileReader(filename)
    pages = pdf_reader.getNumPages()
    n=0
    for first_page in range(0, pages, count):
        pdf_reader = pdf.PdfFileReader(filename)
        pdf_writer = pdf.PdfFileWriter()  # needed to avoid a bug in pyPDF4
        for i in range(count):
            pdf_writer.addPage(pdf_reader.getPage(first_page+i))
        n += 1
        output = f'output/{base}{n:03d}.pdf'
        with open(output, 'wb') as output_pdf:
            pdf_writer.write(output_pdf)
        print(f'created {output}')


@main.command('split_h')
@click.argument('filename', type=click.Path(exists=True))
def split_by_header(filename):
    """split pdf file by header detection"""
    base = os.path.basename(click.format_filename(filename))[:-4]
    pdf_reader = pdf.PdfFileReader(filename)
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
@click.argument('filenames', nargs=-1, type=click.Path(exists=True))
@click.option('-n', '--name', is_flag=True)
def rename2id(filenames, name):
    """rename pdf file using read student ID"""
    results = ocr_files(filenames)
    results = crossref(results)
    for index, row in results.iterrows():
        if name:
            newfilename = os.path.join(os.path.split(row['File'])[0],f'{row.ID}_{row.Name}.pdf')
        else:
            newfilename = os.path.join(os.path.split(row['File'])[0],f'{row.ID}.pdf')
        if not os.path.exists(newfilename):
            print(f'renaming {row["File"]} to {newfilename}')
            os.rename(row['File'],newfilename)
        else:
            print(f'** Warning: skiped {row["File"]} as it already exists')


@main.command('test')
@click.argument('filenames', nargs=-1, type=click.Path(exists=True))
@click.option('-n', '--name', is_flag=True)
def test(filenames, name):
    """test pdf file IDs against confirmed file names and report results"""
    p = 0
    f = 0
    results = ocr_files(filenames)
    print(f'Checking results...')
    for index, row in results.iterrows():
        newfilename = os.path.join(os.path.split(row['File'])[0],f'{row.ID}.pdf')
        if os.path.exists(newfilename):
            p += 1
        else:
            print(f'DID NOT Matched: {row["File"]}, OCR:{row["ID"]}')
            f += 1
    print(f'Test success rate:{p/(p+f):5.1%}\n')
    p = 0
    f = 0
    print(f'Trying to crossreference IDs...')
    results = crossref(results)
    for index, row in results.iterrows():
        if name:
            newfilename = os.path.join(os.path.split(row['File'])[0],f'{row.ID}_{row.Name}.pdf')
        else:
            newfilename = os.path.join(os.path.split(row['File'])[0],f'{row.ID}.pdf')
        if os.path.exists(newfilename):
            p += 1
        else:
            print(f'DID NOT Matched: {row["File"]}, xRef-OCR:{row["ID"]}')
            f += 1
    print(f'Test success rate:{p/(p+f):5.1%}\n')
    print(f'Check the following results:\n{results[results.MatchRatio < 1].sort_values(by="MatchRatio",ascending=False)}')


if __name__ == '__main__':
    main()
