import os
from pathlib import Path
import threading
from tqdm import tqdm
import pydicom as dicom
import cv2
import pandas as pd
from PIL import Image
import numpy as np

from preprocess.image_modules import _read_image_dcm

def removing_file(dcm_path):
    for f in dcm_path.glob('*.dcm'):
        f.unlink()

    for f in dcm_path.glob('*.csv'):
        f.unlink()

class DcmToPng():
    def __init__(self, name_labels, dcm_path=Path("./dicoms").resolve(), png_path=Path("./pngs").resolve(), ds=None, localOp=False, input_type='dicom'):
        #At first step: previous file removing
        if  localOp == False:
            removing_file(dcm_path)


        self.dcm_path = dcm_path
        self.name_labels = name_labels
        self.list_file = dcm_path.joinpath('dcms.csv').resolve()
        self.png_path = png_path
        self.png_path.mkdir(mode=0o755, parents=True, exist_ok=True)

        self.list = []
        if ds != None:
            try:
                accnum = ds.AccessionNumber
                modality = ds.Modality
                sr_num = int(ds.SeriesNumber)
                inst_num = int(ds.InstanceNumber)
                file_name = f'{accnum}_{modality}_{sr_num:06d}_{inst_num:06d}.dcm'
            except:
                file_name = f'E0000000_000000_000000_000000.dcm'

            ds.save_as(str(self.dcm_path.joinpath(file_name).resolve()))  # save dicom to dicom folder

        self.setup_demo(input_type)


    def setup_demo(self, input_type='dicom'):
        print("\n---------------------")
        print("Image preprocessing")
        print("---------------------")

        if input_type=='dicom':
            self.dcm_files = [dcm_file.name for dcm_file in self.dcm_path.iterdir() if dcm_file.suffix == '.dcm']
            df = pd.DataFrame(0, index=np.arange(0, len(self.dcm_files)), columns=self.name_labels)
            df['PATH'] = self.dcm_files
            df.to_csv(self.dcm_path.joinpath('dcms.csv').resolve(), index=False)
        else:
            self.img_files = [png_file.name for png_file in self.png_path.iterdir() if (png_file.suffix in ['.png', '.jpg', '.jpeg'])]
            df = pd.DataFrame(0, index=np.arange(0, len(self.img_files)), columns=self.name_labels)
            df['PATH'] = self.img_files
            df.to_csv(self.png_path.joinpath('images.csv').resolve(), index=False)


    def get_list(self):
        if not self.list_file.exists():
            raise RuntimeError
        else:
            print(f'{self.list_file} is loaded to convert pngs.')

        df = pd.read_csv(self.list_file.resolve())
        self.list = df['PATH'].tolist()

    def check_file(self, ds):
        try:
            accnum = ds.AccessionNumber
            sr_num = int(ds.SeriesNumber)
            modality = ds.Modality
            inst_num = int(ds.InstanceNumber)
            view = ds.ViewPosition
            originality = ds.ImageType[0]
            #Do add a view condition as below if needed
            if originality == 'ORIGINAL':
                return True
            return False
        except Exception as e:
            return False


    def get_png(self, file_name):
        # get dcm file
        out_file = self.dcm_path.joinpath(file_name).resolve()

        # pydicom read
        ds = dicom.read_file(str(out_file))

        ## dicom file meta information checking
        #if self.check_file(ds):
        #    # make a file name of png
        #    png_name = f"{out_file.stem}.png"
        #    png_name = self.png_path.joinpath(png_name).resolve()
        #else:
        #    png_name = None

        png_name = f"{out_file.stem}.png"
        png_name = self.png_path.joinpath(png_name).resolve()
        print(f'{png_name}')

        return out_file, png_name

    def resize_image(self, f_name):
        fp = f_name.resolve()
        img = Image.open(fp)
        w, h = img.size
        rs = (512, int(h/w*512)) if w < h else (int(w/h*512), 512)
        resized = img.resize(rs, Image.LANCZOS)

        return resized


    def preprocess(self, dcm_name, png_name, count=0, overwrite=True):

        # check whether the file already exists
        if (overwrite) or (png_name.exists()==False):
            ## Save results
            try:
                #breakpoint()
                img = _read_image_dcm(str(dcm_name))
                cv2.imwrite(str(png_name), img)
                resized = self.resize_image(png_name)
                resized.save(png_name, 'PNG')
                print(f"success index: {count}")
            except Exception as e:
                img = _read_image_dcm(str(dcm_name))
                cv2.imwrite(str(png_name), img)
                resized = self.resize_image(png_name)
                resized.save(png_name, 'PNG')
                print(f"FILE Converting Error: {dcm_name}")
                print(f"error index: {count}")
                print(e)

        else:
            print(f"skip file (already exists): {png_name}")


    def dcm2png(self):
        # to get instance list from othank
        self.get_list()

        # to get dicom file from the instance list
        t = tqdm(enumerate(self.list), total=len(self.list), desc="dicom converting to png", dynamic_ncols=True)
        png_list = []
        for  i, dcm in t:
            # step 1: file check + dcm download + making file name
            dcm_name, png_name = self.get_png(dcm)
            # step 2: image preprocessing
            if png_name is not None:
                self.preprocess(dcm_name, png_name, count=i)
                png_list.append(png_name.name)

        df = pd.read_csv(self.list_file.resolve())
        df['PATH'] = png_list
        df.to_csv(self.png_path.joinpath('images.csv').resolve(), index=False)


if __name__ == "__main__":
    import argparse

    p = DcmToPng()
    p.dcm2png()


