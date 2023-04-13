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
import fitz
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import glob
import pandas as pd
import difflib
import extract_digit_images as edi

print(f'Loading config file...')
config = json.load(open('dalil.cfg','r'))
ID_box = config['ID_box']
Header_box = config['Header_box']
header_detection_threshold = config['header_detection_threshold']
ratio_threshold = config['ratio_threshold']

def ocr_files(files):
    """run a list of PDF files in OCR using the first page"""
    print(f'Loading tensorflow model...')
    from tensorflow.keras.models import load_model
    model = load_model('model.h5')
    print(f'OCR in progress...\n')
    ID_list = []
    for filename in files:
        doc = fitz.open(filename)
        img = get_img_from_page(doc[0])
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


def get_img_from_page(page):
    """
    pull image from pdf page
    """
    images = page.get_images()
    if len(images) != 1:
        print(f'warning: found {len(images)} images in pdf page')
    img_obj = page.parent.extract_image(images[0][0])
    img_bytes = img_obj["image"]
    img = PIL.Image.open(io.BytesIO(img_bytes))
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
            if type(ratio) == pd.Series:
                ratio = ratio[0]
                newID = newID[0]
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
    doc = fitz.open(filename)
    for i in range(count):
        img = get_img_from_page(doc[i])
        w, h = img.size
        img.crop((Header_box[0]*w, Header_box[1]*h,Header_box[2]*w, Header_box[3]*h)).show()


@main.command('show_id')
@click.argument('filename', type=click.Path(exists=True))
@click.option('-c', '--count', type=click.INT, default=1)
def show_header(filename, count):
    """show ID in pdf file by page"""
    doc = fitz.open(filename)
    for i in range(count):
        try:
            img = get_img_from_page(doc[i])
            w, h = img.size
            img = img.crop((ID_box[0]*w, ID_box[1]*h,ID_box[2]*w, ID_box[3]*h))
            im = edi.convert_im(img, conversion='grey')
            b_xy,b_w,b_h = edi.find_box(im)
            boxes = edi.find_digit_boxes(b_xy,b_w,b_h,9)
            plt.imshow(img)
            ax = plt.gca()
            rect = patches.Rectangle(b_xy,b_w,b_h,linewidth=1,edgecolor='lime',facecolor='none')
            ax.add_patch(rect)
            for b in boxes:
                rect = patches.Rectangle(b[0],b[1],b[2],linewidth=1,edgecolor='red',facecolor='none')
                ax.add_patch(rect)
            plt.show()
        except:
            print(f'Couldn\'t find an ID in area selected in page {i+1}')
            if img:
                plt.imshow(img)
                plt.show()


@main.command('merge')
@click.argument('filenames', nargs=-1, type=click.Path())
@click.option('-o', '--out', default='merged.pdf')
def merge(filenames,out):
    """merge pdf files"""
    newdoc = fitz.open()
    n = 0
    for filename in filenames:
        doc = fitz.open(filename)
        print(f'found {doc.page_count} pages in {filename}')
        newdoc.insert_pdf(doc)
        n += 1
    if n>0:
        newdoc.ez_save(out)
    print(f'merged {n} files into {out}')


@main.command('split_c')
@click.argument('filename', type=click.Path(exists=True))
@click.option('-c', '--count', type=click.INT, required=True)
def split_by_count(filename, count):
    """split pdf file by page count"""
    base = os.path.basename(click.format_filename(filename))[:-4]  # lose the ext and path
    doc = fitz.open(filename)
    n = 0
    for first_page in range(0, doc.page_count, count):
        n += 1
        outfile = f'output/{base}-{n:03d}.pdf'
        newdoc = fitz.open()
        newdoc.insert_pdf(doc, from_page=first_page, to_page=first_page+count-1)
        newdoc.ez_save(outfile)
        print(f'created {outfile}')


@main.command('split_h')
@click.argument('filename', type=click.Path(exists=True))
def split_by_header(filename):
    """split pdf file by header detection"""
    base = os.path.basename(click.format_filename(filename)).split('.')[0]
    doc = fitz.open(filename)
    n = 0
    pageNo = 0
    pagecount = 0
    chksum = 0
    while (pageNo < doc.page_count):
        img = get_img_from_page(doc[pageNo])
        w, h = img.size
        im = np.array(img.crop((Header_box[0]*w,Header_box[1]*h,Header_box[2]*w,Header_box[3]*h)))
        if chksum == 0:
            chksum = has_header(im)
        if has_header(im,chksum) and pagecount > 0:
            n += 1
            outfile = f'output/{base}-{n:03d}.pdf'
            newdoc = fitz.open()
            newdoc.insert_pdf(doc, from_page=pageNo-pagecount, to_page=pageNo-1)
            newdoc.ez_save(outfile)
            print(f'split {pagecount} pages into {outfile}')
            pagecount = 1
        else:
            pagecount += 1
        pageNo += 1
    #  save last document
    n += 1
    outfile = f'output/{base}-{n:03d}.pdf'
    newdoc = fitz.open()
    newdoc.insert_pdf(doc, from_page=pageNo-pagecount, to_page=pageNo-1)
    newdoc.ez_save(outfile)
    print(f'split {pagecount} pages into {outfile}')


@main.command('id')
@click.argument('filenames', nargs=-1, type=click.Path(exists=True))
@click.option('-n', '--name', is_flag=True)
def rename2id(filenames, name):
    """rename pdf file using read student ID"""
    results = ocr_files(filenames)
    results = crossref(results)
    for index, row in results.iterrows():
        basename = os.path.basename(row.File).split('.')[0]
        if name:
            newfilename = os.path.join(os.path.split(row['File'])[0],f'{row.ID}_{row.Name}_{basename}.pdf')
        else:
            newfilename = os.path.join(os.path.split(row['File'])[0],f'{row.ID}_{basename}.pdf')
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
        basename = os.path.basename(row.File).split('.')[0]
        newfilename = os.path.join(os.path.split(row['File'])[0],f'{row.ID}_{basename}.pdf')
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
        basename = os.path.basename(row.File).split('.')[0]
        if name:
            newfilename = os.path.join(os.path.split(row['File'])[0],f'{row.ID}_{row.Name}_{basename}.pdf')
        else:
            newfilename = os.path.join(os.path.split(row['File'])[0],f'{row.ID}_{basename}.pdf')
        if os.path.exists(newfilename):
            p += 1
        else:
            print(f'DID NOT Matched: {row["File"]}, xRef-OCR:{row["ID"]}')
            f += 1
    print(f'Test success rate:{p/(p+f):5.1%}\n')
    print(f'Check the following results:\n{results[results.MatchRatio < 1].sort_values(by="MatchRatio",ascending=False)}')


if __name__ == '__main__':
    main()