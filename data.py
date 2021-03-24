from pathlib import Path

import bisect
from PIL import Image
from tqdm import tqdm
import pandas as pd
import imageio

import torch
import torchvision.transforms as tfms
from torch.utils.data import Dataset, ConcatDataset, Subset

from utils import logger

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

DATA_BASE = Path('./data').resolve()
## doyun' dev computer
#MGH_DATA_BASE = DATA_BASE.joinpath('cxr/mgh/covid19_v4').resolve()
#MGH_DATA_BASE = DATA_BASE.joinpath('cxr/mgh/v4').resolve()
#MGH_DATA_BASE = DATA_BASE.joinpath('cxr/mgh/v4_crop').resolve()
#LOF_DATA_BASE = DATA_BASE.joinpath('cxr/mgh/outlier_cat329').resolve()
#ATLAS_DATA_BASE = DATA_BASE.joinpath('cxr/mgh/v4_crop').resolve()
## LMIC devbox computer
#MGH_DATA_BASE = DATA_BASE.joinpath('CheXpert-v1.0/external_data').resolve()
#MGH_DATA_BASE = DATA_BASE.joinpath('NIH/external_data_pa').resolve()
#MGH_DATA_BASE = DATA_BASE.joinpath('MIMIC_v1/external_data_pa').resolve()
#MGH_DATA_BASE = DATA_BASE.joinpath('covid19_v4').resolve()
#MGH_DATA_BASE = DATA_BASE.joinpath('v4').resolve()
MGH_DATA_BASE = DATA_BASE.joinpath('v4_crop').resolve()
LOF_DATA_BASE = DATA_BASE.joinpath('outlier').resolve()
ATLAS_DATA_BASE = DATA_BASE.joinpath('v4_crop').resolve()

label_name = ['Bone>Fracture>.', 'Bone>Non-fracture>.', 'Diaphragm>Diaphragm>.',
       'Foreign body>.>.', 'Hilar/mediastinum>Aorta>.',
       'Hilar/mediastinum>Cardiomegaly>.', 'Hilar/mediastinum>Hilar area>.',
       'Hilar/mediastinum>Mediastinum>.',
       'Lung density>Decreased density (Lucency)>Cavity/Cyst',
       'Lung density>Decreased density (Lucency)>Emphysema',
       'Lung density>Increased lung density>Atelectasis',
       'Lung density>Increased lung density>Nodule/mass',
       'Lung density>Increased lung density>Other interstitial opacity',
       'Lung density>Increased lung density>Pulmonary edema',
       'Lung density>Increased lung density>pneumonia',
       'Lung volume>Decreased lung volume>.',
       'Lung volume>Increased lung volume>.', 'Pleura>Other pleural lesions>.',
       'Pleura>Pleural effusion>.', 'Pleura>Pneumothorax>.']

folder_name = ['b_f', 'b_nf', 'd_d', 'fb', 'hm_a',
        'hm_c', 'hm_ha', 'hm_m', 'ld_dd_cc', 'ld_dd_e',
        'ld_ild_a', 'ld_ild_nm', 'ld_ild_oio', 'ld_ild_pe', 'ld_ild_p',
        'lv_dlv', 'lv_ilv', 'p_opl', 'p_pe', 'p_p']

Clean_Neg = False
Clean_Neg_list = [('Bone>Fracture>.','clean_negative3.2.1_Bone_Bone_Fracture_BLANK.csv'),
        ('Bone>Non-fracture>.','clean_negative3.2.1_Bone_Bone_Non-fracture_BLANK.csv'),
        ('Diaphragm>Diaphragm>.','clean_negative3.2.1_Below  diaphragm _Diaphragm_Diaphragm_BLANK.csv'),
        ('Foreign body>.>.','clean_negative3.2.1_Whole CXR_Foreign body_BLANK_BLANK.csv'),
        ('Hilar/mediastinum>Aorta>.','clean_negative3.2.1_Hilar mediastinum_Hilar mediastinum_Aorta_BLANK.csv'),
        ('Hilar/mediastinum>Cardiomegaly>.','clean_negative3.2.1_Hilar mediastinum_Hilar mediastinum_Cardiomegaly_BLANK.csv'),
        ('Hilar/mediastinum>Hilar area>.','clean_negative3.2.1_Hilar mediastinum_Hilar mediastinum_Hilar area_BLANK.csv'),
        ('Hilar/mediastinum>Mediastinum>.','clean_negative3.2.1_Hilar mediastinum_Hilar mediastinum_Mediastinum_BLANK.csv'),
        ('Lung density>Decreased density (Lucency)>Cavity/Cyst','clean_negative3.2.1_Lung_Lung density_Decreased density (Lucency)_Cavity Cyst.csv'),
        ('Lung density>Decreased density (Lucency)>Emphysema','clean_negative3.2.1_Lung_Lung density_Decreased density (Lucency)_Emphysema.csv'),
        ('Lung density>Increased lung density>Atelectasis','clean_negative3.2.1_Lung_Lung density_Increased lung density_Atelectasis.csv'),
        ('Lung density>Increased lung density>Nodule/mass','clean_negative3.2.1_Lung_Lung density_Increased lung density_Nodule mass.csv'),
        ('Lung density>Increased lung density>Other interstitial opacity','clean_negative3.2.1_Lung_Lung density_Increased lung density_Other interstitial opacity.csv'),
        ('Lung density>Increased lung density>Pulmonary edema','clean_negative3.2.1_Lung_Lung density_Increased lung density_Pulmonary edema.csv'),
        ('Lung density>Increased lung density>pneumonia','clean_negative3.2.1_Lung_Lung density_Increased lung density_pneumonia.csv'),
        ('Lung volume>Decreased lung volume>.','clean_negative3.2.1_Lung_Lung volume_Decreased lung volume_BLANK.csv'),
        ('Lung volume>Increased lung volume>.','clean_negative3.2.1_Lung_Lung volume_Increased lung volume_BLANK.csv'),
        ('Pleura>Other pleural lesions>.','clean_negative3.2.1_Pleura_Pleura_Other pleural lesions_BLANK.csv'),
        ('Pleura>Pleural effusion>.','clean_negative3.2.1_Pleura_Pleura_Pleural effusion_BLANK.csv'),
        ('Pleura>Pneumothorax>.','clean_negative3.2.1_Pleura_Pleura_Pneumothorax_BLANK.csv')]

def _tb_load_manifest(file_path, num_labels=31, name_labels=None, name_paths=None, mode='single', ext_data=False, fl_balance=False, r_seed=-1):
    if not file_path.exists():
        logger.error(f"manifest file {file_path} not found.")
        raise RuntimeError

    logger.debug(f"loading dataset manifest {file_path} ...")
    df = pd.read_csv(str(file_path)).fillna(0)

    if (not ext_data) and (True): # using the clean-set
        # cleanset
        if True:
            ## MGH validation set
            df = df.loc[(df['bad_age'] == 0) & (df['bad_quality'] == 0)]
            if r_seed != -1:
                df = df.sample(n=1000, replace=True, random_state=r_seed)

            if (False):
                df = df.loc[(df['Hilar/mediastinum>Cardiomegaly>.']==1) 
                        | (df['Lung density>Increased lung density>Atelectasis'] == 1)
                        | (df['Lung density>Increased lung density>Pulmonary edema'] == 1)
                        | (df['Lung density>Increased lung density>pneumonia'] == 1)
                        | (df['Pleura>Pleural effusion>.'] == 1)]
                df.reset_index(drop=True, inplace=True)

            ## MGH testset
            #df = df.loc[(df['bad_age'] == 0) & (df['bad_quality'] == 0)][0:250]
            #df = df.loc[(df['bad_age'] == 0) & (df['bad_quality'] == 0)][250:500]
            #df = df.loc[(df['bad_age'] == 0) & (df['bad_quality'] == 0)][500:750]
            #df = df.loc[(df['bad_age'] == 0) & (df['bad_quality'] == 0)][750:]
            
            ## CheXpert trainset
            #df = df.loc[(df['bad_age'] == 0) & (df['bad_quality'] == 0)][0:250]
            #df = df.loc[(df['bad_age'] == 0) & (df['bad_quality'] == 0)][250:500]
            #df = df.loc[(df['bad_age'] == 0) & (df['bad_quality'] == 0)][500:750]
            #df = df.loc[(df['bad_age'] == 0) & (df['bad_quality'] == 0)][750:1000]
            #df = df.loc[(df['bad_age'] == 0) & (df['bad_quality'] == 0)][1000:1250]
            #df = df.loc[(df['bad_age'] == 0) & (df['bad_quality'] == 0)][1250:1500]
            #df = df.loc[(df['bad_age'] == 0) & (df['bad_quality'] == 0)][1500:1750]
            #df = df.loc[(df['bad_age'] == 0) & (df['bad_quality'] == 0)][1750:2000]
            #df = df.loc[(df['bad_age'] == 0) & (df['bad_quality'] == 0)][2000:2250]
            #df = df.loc[(df['bad_age'] == 0) & (df['bad_quality'] == 0)][2250:2500]
            #df = df.loc[(df['bad_age'] == 0) & (df['bad_quality'] == 0)][2500:2750]
            #df = df.loc[(df['bad_age'] == 0) & (df['bad_quality'] == 0)][2750:3000]
            #df = df.loc[(df['bad_age'] == 0) & (df['bad_quality'] == 0)][3000:3250]
            #df = df.loc[(df['bad_age'] == 0) & (df['bad_quality'] == 0)][3250:3500]
            #df = df.loc[(df['bad_age'] == 0) & (df['bad_quality'] == 0)][3500:3750]
            #df = df.loc[(df['bad_age'] == 0) & (df['bad_quality'] == 0)][3750:4000]
            #df = df.loc[(df['bad_age'] == 0) & (df['bad_quality'] == 0)][4000:4250]
            #df = df.loc[(df['bad_age'] == 0) & (df['bad_quality'] == 0)][4250:4500]
            #df = df.loc[(df['bad_age'] == 0) & (df['bad_quality'] == 0)][4500:]

            ## NIH trainset
            #df = df.loc[(df['bad_age'] == 0) & (df['bad_quality'] == 0)][0:500]
            #df = df.loc[(df['bad_age'] == 0) & (df['bad_quality'] == 0)][500:1000]
            #df = df.loc[(df['bad_age'] == 0) & (df['bad_quality'] == 0)][1000:1500]
            #df = df.loc[(df['bad_age'] == 0) & (df['bad_quality'] == 0)][1500:2000]
            #df = df.loc[(df['bad_age'] == 0) & (df['bad_quality'] == 0)][2000:2500]
            #df = df.loc[(df['bad_age'] == 0) & (df['bad_quality'] == 0)][2500:3000]
            #df = df.loc[(df['bad_age'] == 0) & (df['bad_quality'] == 0)][3000:3500]
            #df = df.loc[(df['bad_age'] == 0) & (df['bad_quality'] == 0)][3500:4000]
            #df = df.loc[(df['bad_age'] == 0) & (df['bad_quality'] == 0)][4000:]

            ## MIMIC trainset
            #df = df.loc[(df['bad_age'] == 0) & (df['bad_quality'] == 0)][0:500]
            #df = df.loc[(df['bad_age'] == 0) & (df['bad_quality'] == 0)][500:1000]
            #df = df.loc[(df['bad_age'] == 0) & (df['bad_quality'] == 0)][1000:1500]
            #df = df.loc[(df['bad_age'] == 0) & (df['bad_quality'] == 0)][1500:2000]
            #df = df.loc[(df['bad_age'] == 0) & (df['bad_quality'] == 0)][2000:2500]
            #df = df.loc[(df['bad_age'] == 0) & (df['bad_quality'] == 0)][2500:]

            if (Clean_Neg):
                #hilar area special care
                for cl_feature, cl_file in Clean_Neg_list:
                    df_case = pd.read_csv(MGH_DATA_BASE.joinpath('clean_nagative_data_v5_deblank/'+cl_file), names=['ACC'])
                    #df_case = pd.read_csv(MGH_DATA_BASE.joinpath('clean_nagative_data_v5/'+cl_file))
                    #df_case = pd.read_csv(MGH_DATA_BASE.joinpath('clean_negative_data_v5_321/'+cl_file), names=['ACC'])
                    df[f'{cl_feature}'] = df[f'{cl_feature}'].replace(0, -2)
                    #df.loc[df.AccessionNumber.isin(df_case.ACC), f'{cl_feature}'] = 0
                    df.loc[(df.AccessionNumber.isin(df_case.ACC))&(df[f'{cl_feature}']==-2), f'{cl_feature}'] = 0

            if (fl_balance):
                for k, feature in enumerate(label_name):
                    num_p = df.loc[(df[f'{feature}'] == 1)].shape[0]
                    num_n = df.loc[(df[f'{feature}'] == 0)].shape[0]
                    ratio_pn = num_p / num_n
                    ratio_th = 5
                    if (ratio_pn < (1.0/ratio_th)):
                        df[f'{feature}'] = df[f'{feature}'].replace(0, -1)
                        df_n = df.loc[(df[f'{feature}'] == -1)].sample(n=(num_p*ratio_th), random_state=2020)
                        df[f'{feature}'].loc[df['AccessionNumber'].isin(df_n['AccessionNumber'])] = 0

                        pos = df[f'{feature}'].loc[df[f'{feature}']==1].shape[0]
                        neg = df[f'{feature}'].loc[df[f'{feature}']==0].shape[0]
                        dontcare = df[f'{feature}'].loc[df[f'{feature}']==-1].shape[0]

                        logger.info(f'[{k:02d}-{feature}] pos: {pos}, neg: {neg}, dont-care: {dontcare}')

                if name_labels == None:
                    df = df[~(df.iloc[:, -(num_labels+1):-1] == -1).all(1)]
                else:
                    df = df[~(df[name_labels] == -1).all(1)]
                df.reset_index(drop=True, inplace=True)

            if (Clean_Neg):
                for cl_feature, cl_file in Clean_Neg_list:
                    df[f'{cl_feature}'] = df[f'{cl_feature}'].replace(-2, -1)

            if False:
                for k, feature in enumerate(label_name):
                    num_p = df.loc[(df[f'{feature}'] == 1)].shape[0]
                    num_n = df.loc[(df[f'{feature}'] == 0)].shape[0]
                    num_i = df.loc[(df[f'{feature}'] == -1)].shape[0]

                    print(f'{feature}-{num_p}-{num_p/df.shape[0]}-{num_i}-{num_i/df.shape[0]}-{num_n}-{num_n/df.shape[0]}')
                exit(-1)

            if (True): # in order to add clinical information to network
                df['ScaledSex'] = df.sex.replace(0, -1)
                weight_gender = 10
                weight_age = 100
                min_age = 11.0
                max_age = 100.0
                #df.PatientAge = (df.PatientAge-min(df.PatientAge))/(max(df.PatientAge)-min(df.PatientAge))
                df['ScaledAge'] = (df.PatientAge-min_age)/(max_age-min_age)
                df.ScaledAge = weight_age * (df.ScaledAge - 0.5)
                df['ScaledSex'] = weight_gender * df.ScaledSex

            df.reset_index(drop=True, inplace=True)

    else:
        try:
            df['ScaledSex'] = df.sex.replace(0, -1)
            weight_gender = 10
            weight_age = 100
            min_age = 11.0
            max_age = 117.0
            #df.PatientAge = (df.PatientAge-min(df.PatientAge))/(max(df.PatientAge)-min(df.PatientAge))
            df['ScaledAge'] = (df.PatientAge-min_age)/(max_age-min_age)
            df.ScaledAge = weight_age * (df.ScaledAge - 0.5)
            df['ScaledSex'] = weight_gender * df.ScaledSex
        except:
            df['ScaledAge'] = 0
            df['ScaledSex'] = 0


    if (mode == 'single') | (mode == 'extd'):
        LABELS = df.columns[-(num_labels+1):-1] if name_labels == None else name_labels
        labels = df[LABELS].astype(int)
        paths = df['PATH'] if name_paths == None else df[name_paths]
        ages = df['ScaledAge'].astype(float)
        genders = df['ScaledSex'].astype(float)
        df_tmp = pd.concat([paths, ages, genders, labels], axis=1)
    elif mode == 'double':
        LABELS = df.columns[-(num_labels+2):-2] if name_labels == None else name_labels
        labels = df[LABELS].astype(int)
        paths = df[df.columns[-2:]] if name_paths == None else df[name_paths]
        df_tmp = pd.concat([paths, labels], axis=1)
    else:
        raise RuntimeError

    entries = df_tmp

    logger.debug(f"{len(entries)} entries are loaded.")
    return entries

# data augmentation - 512
train_transforms = tfms.Compose([
    tfms.ToPILImage(),
    tfms.Resize(562, Image.LANCZOS),
    tfms.RandomRotation((-10, 10)),
    tfms.RandomCrop((512, 512)),
    tfms.RandomHorizontalFlip(p=0.01), #with 1% horizontal flip
    tfms.ToTensor(),
])

test_transforms = tfms.Compose([
    tfms.ToPILImage(),
    tfms.Resize((512, 512), Image.LANCZOS),
    tfms.ToTensor(),
])

def get_image(img_path, transforms):
    image = imageio.imread(img_path)
    image_tensor = transforms(image)
    image_tensor = image_tensor[:1, :, :]
    #print(f'{img_path}-{image_tensor.shape}')
    return image_tensor


class CxrDataset(Dataset):
    transforms = train_transforms

    def __init__(self, base_path, manifest_file, num_labels=31, name_labels=None, name_paths=None, mode='single', ext_data=False, csv_path=None, fl_balance=False, r_seed=-1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        manifest_path = base_path.joinpath(manifest_file).resolve() if csv_path == None else csv_path.joinpath(manifest_file).resolve()
        self.entries = _tb_load_manifest(manifest_path, num_labels=num_labels, name_labels=name_labels, name_paths=name_paths, mode=mode, ext_data=ext_data, fl_balance=fl_balance, r_seed = r_seed)
        self.base_path = base_path
        self.mode = mode
        self.name_labels = name_labels

    def __getitem__(self, index):
        # need to debug
        def get_entries(index):
            df = self.entries.loc[index]
            if (self.mode == 'single') | (self.mode == 'extd'):
                paths = self.base_path.joinpath(df[0]).resolve()
                label = df[3:].tolist() if self.name_labels == None else df[self.name_labels].tolist()
                age = df[1]
                gender = df[2]
                return paths, label, age, gender
            else:
                paths = [self.base_path.joinpath(df[0]).resolve(), self.base_path.joinpath(df[1]).resolve()]
                label = df[2:].tolist() if self.name_labels == None else df[self.name_labels].tolist()
                return paths, label

        if (self.mode == 'single') | (self.mode == 'extd'):
            img_path, label, age, gender = get_entries(index)
            image_tensor = get_image(img_path, CxrDataset.transforms)
            target_tensor = torch.FloatTensor(label)
            clinic_tensor = torch.FloatTensor([age, gender])
            #clinic_tensor = torch.FloatTensor([age])
            return image_tensor, target_tensor, clinic_tensor
        elif self.mode == 'double':
            img_paths, label = get_entries(index)
            image_tensor0 = get_image(img_paths[0], CxrDataset.transforms)
            image_tensor1 = get_image(img_paths[1], CxrDataset.transforms)
            target_tensor = torch.FloatTensor(label)
            return image_tensor0, image_tensor1, target_tensor
        else:
            raise RuntimeError


    def __len__(self):
        return len(self.entries)

    def get_label_counts(self, indices=None):
        df = self.entries if indices is None else self.entries.loc[indices]
        counts = [df[x].value_counts() for x in self.labels]
        new_df = pd.concat(counts, axis=1).fillna(0).astype(int)
        return new_df

    @property
    def labels(self):
        #if self.mode == 'single':
        #    return self.entries.columns[1:].values.tolist()
        #elif self.mode == 'extd':
        if (self.mode == 'single') | (self.mode == 'extd'):
            return self.entries.columns[3:].values.tolist()
        else:
            return self.entries.columns[2:].values.tolist()

    @staticmethod
    def train():
        CxrDataset.transforms = train_transforms

    @staticmethod
    def eval():
        CxrDataset.transforms = test_transforms

class CxrConcatDataset(ConcatDataset):

    #def __init__(self, *args, **kwargs):
    #    super().__init__(*args, **kwargs)
    #    self.get_label_counts()

    def get_label_counts(self, indices=None):
        if indices is None:
            indices = list(range(self.__len__()))
        dataset_indices = [bisect.bisect_right(self.cumulative_sizes, idx) for idx in indices]
        sample_indices = [(i if d == 0 else i - self.cumulative_sizes[d - 1]) for i, d in zip(indices, dataset_indices)]
        nested_indices = [[] for d in self.datasets]
        for d, s in zip(dataset_indices, sample_indices):
            nested_indices[d].append(s)
        dfs = []
        for d, dataset in enumerate(self.datasets):
            dfs.append(dataset.get_label_counts(nested_indices[d]))
        df = pd.concat(dfs, sort=False).groupby(level=0).sum().astype(int)
        for dataset in self.datasets:
            assert len(df.columns) == len(dataset.labels), "label names should be matched!"
        return df

    @property
    def labels(self):
        return self.datasets[0].labels


class CxrSubset(Subset):

    #def __init__(self, *args, **kwargs):
    #    super().__init__(*args, **kwargs)
    #    self.get_label_counts()

    def get_label_counts(self, indices=None):
        if indices is None:
            indices = list(range(self.__len__()))

        df = self.dataset.get_label_counts([self.indices[x] for x in indices])
        return df

    @property
    def labels(self):
        return self.dataset.labels


def CxrRandomSplit(dataset, lengths):
    from torch._utils import _accumulate
    if sum(lengths) > len(dataset):
        raise ValueError("Sum of input lengths must less or equal to the length of the input dataset!")
    indices = torch.randperm(sum(lengths)).tolist()
    return [CxrSubset(dataset, indices[offset - length:offset]) for offset, length in zip(_accumulate(lengths), lengths)]



# Initiating dataset
def copy_mgh_dataset(src_path, csv_path, csv_file, t_view='AP', t_path='PATH', cont_op=False):
    if (cont_op):
        csvs = MGH_DATA_BASE.joinpath(csv_file).resolve()
    else:
        csvs = csv_path.joinpath(csv_file)

    for m in [csvs.resolve()]:
        print(f'>>> processing {m}')

        df = pd.read_csv(str(m))
        failures = []
        failed_files = []
        for i in tqdm(range(len(df)), total=len(df)):
            fs = [df.iloc[i][str(t_path)]]

            for k, f in enumerate(fs):
                r, df = anonymization(df, i, t_view, t_path)
                t = MGH_DATA_BASE.joinpath(r).resolve()

                if Path(t).is_file():
                    print(f'skip the existed file: {t}')
                else:
                    try:
                        resized = resize_image(f)
                        Path.mkdir(t.parent, parents=True, exist_ok=True)
                        resized.save(t, 'PNG')
                    except:
                        failures.append(i)
                        failed_files.append(r)
                        #breakpoint()

        df = df if cont_op else gen_labels(df)
        print(f'before failures: {df.shape}')
        df = df.drop(failures)
        print(f'after failures: {df.shape}')
        #breakpoint()
        t = MGH_DATA_BASE.joinpath(csv_file).resolve()
        #t = MGH_DATA_BASE.joinpath('post2015_mgh_cxr_all_dataset_v3').resolve()
        df.to_csv(t, float_format='%.0f', index=False)

        # 1. file make for two inputs is okay?
        # 2. make csv files for one or two inputs are okay?
        # 3. implementation of Data batch + augmentation

def resize_image(f_name):
    fp = src_path.joinpath(f_name).resolve()
    img = Image.open(fp)
    w, h = img.size
    rs = (512, int(h/w*512)) if w < h else (int(w/h*512), 512)
    resized = img.resize(rs, Image.LANCZOS)

    return resized

# anonymizing a file name
def anonymization(df, i, t_view, t_path):
    r = f'mgh_{t_view}_{df.iloc[i][0]:08d}.png'
    df.loc[i, t_path] = r

    return r, df

def gen_labels(df):
    # view positions
    df.insert(7, 'ap', 0)
    df['ap'].loc[df['ViewPosition'] == 'AP'] = 1
    df.insert(8, 'pa', 0)
    df['pa'].loc[df['ViewPosition'] == 'PA'] = 1
    df.insert(9, 'll', 0)
    df['ll'].loc[df['ViewPosition'] == 'LL'] = 1
    # sex
    df.insert(11, 'sex', 0)
    df['sex'].loc[df['PatientSex'] == 'M'] = 1
    # manufacturer
    df.insert(14, 'varian', 0)
    df['varian'].loc[df['Manufacturer'] == 'Varian'] = 1
    df.insert(15, 'agfa', 0)
    df['agfa'].loc[df['Manufacturer'] == 'Agfa'] = 1
    df.insert(16, 'ge', 0)
    df['ge'].loc[(df['Manufacturer'] == 'GE Healthcare') | (df['Manufacturer'] == '"GE Healthcare"') | (df['Manufacturer'] == 'GE MEDICAL SYSTEMS')] = 1
    df.insert(17, 'others', 0)
    df['others'].loc[(df['varian'] + df['agfa'] + df['ge']) == 0] = 1

    return df

if __name__ == "__main__":
    if (False):
        src_path = Path('/mnt/hdd/data_storage/mgh_cxr_img').resolve()
        csv_path = Path('/mnt/hdd/data_storage/mgh_cxr_list/clean-lists/20200316-dataset-mgh-v4').resolve()
        if src_path.exists():
            # for AP
            #csv_file = 'example-10-ap.csv'
            csv_file = 'post2015_mgh_cxr_ap_dataset_v4.csv'
            copy_mgh_dataset(src_path, csv_path, csv_file, t_view='AP', t_path='PATH')
            # for PA LL
            #csv_file = 'example-10-pa-ll.csv'
            csv_file = 'post2015_mgh_cxr_pa_ll_dataset_v4.csv'
            copy_mgh_dataset(src_path, csv_path, csv_file, t_view='PA', t_path='PATH1')
            copy_mgh_dataset(src_path, csv_path, csv_file, t_view='LL', t_path='PATH2', cont_op=True)
            #csv_file = 'example-10-pa.csv'
            csv_file = 'post2015_mgh_cxr_pa_dataset_v4.csv'
            copy_mgh_dataset(src_path, csv_path, csv_file, t_view='PA', t_path='PATH')
            #csv_file = 'example-10-ll.csv'
            csv_file = 'post2015_mgh_cxr_ll_dataset_v4.csv'
            copy_mgh_dataset(src_path, csv_path, csv_file, t_view='LL', t_path='PATH')
        else:
            assert False, (f'{src_path} is not existed.')
    else:
        inc_labels = [1, 2, 3, 6, 7, 8, 11, 13, 14, 15, 18, 19, 21, 22, 28]
        inc_rate   = [2, 8, 4, 2, 4, 2,  2,  8, 16,  8,  8,  4,  8,  8,  8]
        for k, feature in enumerate(label_name):
            num_p = df.loc[(df[f'{feature}'] == 1)].shape[0]
