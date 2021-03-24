import os,sys
import pandas as pd
import pydicom as dicom
import numpy as np
import cv2
from struct import pack, unpack

def _read_image_dcm(dcm_file):
    ## Get new image file name
    filename = os.path.basename(dcm_file)
    basename, ext = os.path.splitext(filename)

    ds = dicom.read_file(dcm_file, force=True)
    ds.decompress()

    ## Convert by manual
    if ds.get('WindowCenter'):
        x_arr = load_att_img_with_Window(ds)
    elif ds.get('VOILUTSequence'):
        x_arr = load_att_img_with_LUT(ds)
    else:
        x_arr = load_raw_pixel_val(ds)

    x_arr = load_att_with_clahe_then_1to95(x_arr)
    x_arr = np.expand_dims(x_arr, axis=-1)
    x_arr = x_arr.astype('uint8')

    return x_arr


def load_att_with_clahe_then_1to95(img):
    img = img.astype('uint16')
    clahe = cv2.createCLAHE(clipLimit=3, tileGridSize=(1, 1))
    try:
        new_img = clahe.apply(img)
        new_img = load_att_with_1to95_norm(new_img)
    except cv2.error as e:
        new_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        new_img = clahe.apply(new_img)
        new_img = load_att_with_1to95_norm(new_img)
    return new_img


def load_att_with_1to95_norm(img):
    max_val = np.percentile(img, 95)
    min_val = np.percentile(img, 1)
    #print("Norm with 95 percent to 1 percent:", max_val, min_val )

    new_img = np.clip(img, a_min=min_val, a_max=max_val)
    new_img = (new_img - min_val) / (max_val - min_val) * 255
    return new_img


def load_att_img_with_Window(ds):
    img = ds.pixel_array

    mean_val = np.mean(img)
    if is_inverse(ds):
        img = np.vectorize(lambda x: (x - mean_val) * -1 + mean_val)(img)

    wc, ww = get_window_info_from_ds(ds)
    img = convert_to_8bit_with_window(img, wc, ww)
    return img


def get_window_info_from_ds(ds):
    ## Check window_level
    if type(ds.WindowCenter) == dicom.multival.MultiValue:
        win_center = int(ds.WindowCenter[0])
    else:
        win_center = int(ds.WindowCenter)
    if type(ds.WindowWidth) == dicom.multival.MultiValue:
        wind_width = int(ds.WindowWidth[0])
    else:
        wind_width = int(ds.WindowWidth)
    return win_center, wind_width


def convert_to_8bit_with_window(img, center, width):

    min_val = 0.0
    max_val = 255.0
    center = float(center)
    width = float(width)

    img = img.astype(np.float64)
    img_result = np.zeros(img.shape, dtype=np.float64)

    th_min = center - 0.5 - (width - 1.0) / 2.0
    th_max = center - 0.5 + (width - 1.0) / 2.0

    idx_left = np.where(img <= th_min)
    idx_right = np.where(img > th_max)
    idx_mid = np.where((img > th_min) & (img <= th_max))

    img_result[idx_left] = min_val
    img_result[idx_right] = max_val
    img_result[idx_mid] = ((img[idx_mid] - (center - 0.5)) / (width - 1.0) + 0.5) * (max_val - min_val) + min_val

    img_result = img_result.astype(np.uint8)
    return img_result


def load_att_img_with_LUT(ds):
    def chunks(l, n):
        """Yield successive n-sized chunks from l."""
        for i in range(0, len(l), n):
            yield l[i:i + n]

    def byte_to_numlist(data):
        num_list = []
        for byte_chunk in chunks(data, 2):
            # print("Byte chunk:", byte_chunk)
            num = unpack("<h", byte_chunk)
            num_list.append(num[0])
        return num_list
    try:
        img = ds.pixel_array
        b_data = ds.VOILUTSequence[0].LUTData
        descriptor = ds.VOILUTSequence[0].LUTDescriptor
    except:
        img = ds.pixel_array
        b_data = ds.VOILUTSequence[0].LUTData
        descriptor = ds.VOILUTSequence[0].LUTDescriptor

    ## Make lookuptalbe descriptor
    if type(descriptor._list) == list:
        lut_desc = descriptor
    else:
        lut_desc = byte_to_numlist(descriptor)

    ## Make lookup table
    if type(b_data) == list:
        lut_table = b_data
    else:
        lut_table = byte_to_numlist(b_data)

    assert(len(lut_table) == lut_desc[0])

    def mapping(value):
        '''
        Plase refer to DICOM standard link below :
        http://dicom.nema.org/dicom/2013/output/chtml/part03/sect_C.11.html
        '''
        if value < lut_desc[1]:  ## Min value == lut_table[0]
            result = lut_table[0]
        elif value >= lut_desc[1] + lut_desc[0]:
            result = lut_table[-1]  ## Max value == lut_table[-1]
        else:
            result = lut_table[value - lut_desc[1]]
        return result

    ## conver image pixel values
    new_img = np.vectorize(mapping)(img)

    if is_inverse(ds):
        new_img = -new_img

    return new_img


def load_raw_pixel_val(ds):
    img = ds.pixel_array

    if is_inverse(ds):
        mean_val = np.mean(img)
        img = np.vectorize(lambda x: (x - mean_val) * -1 + mean_val)(img)

    return img


def is_inverse(ds):
    return ds.get('PresentationLUTShape', False) and ds.PresentationLUTShape == 'INVERSE'\
            or ds.PhotometricInterpretation == 'MONOCHROME1'

