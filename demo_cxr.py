import sys
from pathlib import Path
#sys.path.append(str(Path('./..').resolve()))

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchnet as tnt
import pandas as pd
import yaml

from utils import logger
from test import Tester
from environment import DemoEnvironment, initialize, print_label_name
from gradcam import GradCam, save_class_activation_images
from data import CxrDataset, MGH_DATA_BASE

from preprocess.dcm2png import DcmToPng


class Demo(Tester):
    def __init__(self, env, pt_runtime="test", fn_net=None, fl_gradcam=False, cls_gradcam=None, th_gradcam=0.7, fl_ensemble=False):
        super().__init__(env, pt_runtime=pt_runtime, fn_net=fn_net, fl_gradcam=fl_gradcam, cls_gradcam=cls_gradcam, fl_ensemble=fl_ensemble)
        self.th_gradcam = th_gradcam
        self.fl_gradcam_save = False
        self.fl_ensemble = fl_ensemble

    def demo_cxr_evaluation(self, epoch=1, fl_save=False, hmp_dims=None):
        if self.fn_net == None:
            pt_model = self.pt_runtime.joinpath('models/multitask_model_00.pth.tar')
        else:
            pt_model = self.pt_runtime.joinpath('models/'+str(self.fn_net)+'_00.pth.tar')

        self.env.load_model(pt_model)
        try:
            self.load()
        except:
            logger.debug('there is no pkl to load.')

        prob, pred = self.demo_cxr_test(epoch, self.env.test_loader, fl_save=fl_save)

        if self.fl_gradcam:
            self.gradcam_data(self.env.gradcam_loader, hmp_dims=hmp_dims)

        result ={
                'prob': prob.tolist(),
                'pred': pred.tolist(),
                }

        return result

    def demo_cxr_ensemble_evaluation(self, epoch=1, fl_save=False, hmp_dims=None, n_ens=1):

        probs  = []
        if self.fl_gradcam:
            cams = np.ones((len(self.env.gradcam_loader), len(self.cls_gradcam), 16, 16))

        for k in range(n_ens):
            pt_model = self.pt_runtime.joinpath('models/'+str(self.fn_net)+f'_{k:02d}.pth.tar')
            self.env.load_model(pt_model)
            #logger.info(f'network to test: {self.env.model}')
            try:
                self.load()
            except:
                logger.debug('there is no pkl to load.')

            prob, _ = self.demo_cxr_test(epoch, self.env.test_loader, fl_save=fl_save)
            probs.append(prob)


            if self.fl_gradcam:
                _, _, cam = self.gradcam_data(self.env.gradcam_loader, hmp_dims=hmp_dims)
                cams += cam

        # evaluate ensemble's performance
        prob, pred = self.demo_cxr_ensemble_test(probs, n_ens)

        if self.fl_gradcam:
            #[doyun] need to debugging
            _, _, cams = self.gradcam_data(self.env.gradcam_loader, ens_flg=True, cams_ens=cams, prob_ens=prob)

        result ={
                'prob': prob.tolist(),
                'pred': pred.tolist(),
                }

        return result



    def demo_cxr_test(self, epoch, test_loader, fl_save=False, fl_iter_save=False, fl_iter_target=0):
        test_set = test_loader.dataset
        out_dim = self.env.out_dim
        labels = self.env.labels

        CxrDataset.eval()
        self.env.model.eval()

        with torch.no_grad():
            tqdm_desc = f'testing '
            t = tqdm(enumerate(test_loader), total=len(test_loader), desc=tqdm_desc,
                    dynamic_ncols=True)

            pred = torch.zeros(len(test_loader), out_dim).int()
            prob = torch.zeros(len(test_loader), out_dim).float()
            for bt_idx, tp_data in t:
                output, _  = self.test_batch(tp_data)
                output[:, 0:-1] = torch.sigmoid(output[:, 0:-1])

                # view position decision
                pred[bt_idx, 0] = 1 if output[:, 0].item() >= output[:, 1].item() else 0
                pred[bt_idx, 1] = 0 if output[:, 0].item() >= output[:, 1].item() else 1
                # gender decision
                pred[bt_idx, 2] = 1 if output[:, 2].item() >= output[:, 3].item() else 0
                pred[bt_idx, 3] = 0 if output[:, 2].item() >= output[:, 3].item() else 1
                # gender decision
                pred[bt_idx, 4] = 1 if (output[:, 4].item() >= output[:, 5].item()) & (output[:, 4].item() >= output[:, 6].item()) else 0
                pred[bt_idx, 5] = 1 if (output[:, 5].item() >= output[:, 4].item()) & (output[:, 5].item() >= output[:, 6].item()) else 0
                pred[bt_idx, 6] = 1 if (output[:, 6].item() >= output[:, 5].item()) & (output[:, 6].item() >= output[:, 4].item()) else 0
                #  decision
                pred[bt_idx, 7] = 1 if output[:, 7].item() >= 0.7 else 0
                pred[bt_idx, 8] = 1 if output[:, 8].item() >= 0.7 else 0
                pred[bt_idx, 9] = 1 if output[:, 9].item() >= 0.7 else 0
                pred[bt_idx, 10] = 1 if output[:, 10].item() >= 0.7 else 0
                pred[bt_idx, 11] = 1 if output[:, 11].item() >= 0.7 else 0
                pred[bt_idx, 12] = 1 if output[:, 12].item() >= 0.7 else 0
                pred[bt_idx, 13] = 1 if output[:, 13].item() >= 0.7 else 0


                pred[bt_idx, 14] = int(np.round(output[:, 14].item()))
                prob[bt_idx, :] = output

        return prob, pred

    def demo_cxr_ensemble_test(self, probs, n_ens):
        output = torch.zeros(probs[0].shape)
        for i in range(n_ens):
            output += probs[i]

        output /= n_ens

        pred = torch.zeros(probs[0].shape).int()
        prob = torch.zeros(probs[0].shape).float()

        # view position decision
        for i in range(pred.shape[0]):
            pred[i, 0] = 1 if output[i, 0].item() >= output[i, 1].item() else 0
            pred[i, 1] = 0 if output[i, 0].item() >= output[i, 1].item() else 1
            # gender decision
            pred[i, 2] = 1 if output[i, 2].item() >= output[i, 3].item() else 0
            pred[i, 3] = 0 if output[i, 2].item() >= output[i, 3].item() else 1
            # gender decision
            pred[i, 4] = 1 if (output[i, 4].item() >= output[i, 5].item()) & (output[i, 4].item() >= output[i, 6].item()) else 0
            pred[i, 5] = 1 if (output[i, 5].item() >= output[i, 4].item()) & (output[i, 5].item() >= output[i, 6].item()) else 0
            pred[i, 6] = 1 if (output[i, 6].item() >= output[i, 5].item()) & (output[i, 6].item() >= output[i, 4].item()) else 0
            #  decision
            pred[i, 7] = 1 if output[i, 7].item() >= 0.7 else 0
            pred[i, 8] = 1 if output[i, 8].item() >= 0.7 else 0
            pred[i, 9] = 1 if output[i, 9].item() >= 0.7 else 0
            pred[i, 10] = 1 if output[i, 10].item() >= 0.7 else 0
            pred[i, 11] = 1 if output[i, 11].item() >= 0.7 else 0
            pred[i, 12] = 1 if output[i, 12].item() >= 0.7 else 0
            pred[i, 13] = 1 if output[i, 13].item() >= 0.7 else 0

            pred[i, 14] = int(np.round(output[i, 14].item()))
        prob = output
        return prob, pred


def cxr_predict(hmp_dims=None, dcm_file=None, cuda='0', fl_gradcam=True, Nens=3, th_gradcam=0.7, input_type='dicom'):

    print("\n-------------------------------------------------------------------")
    print("|                                                                 |")
    print("|                                                                 |")
    print("|     v1.0 MGH Age, View, Gender, Vendor, Abnormal Detection      |")
    print("|    (Copyright (c) 2021-2022, MGH LMIC. All rights reserved.)    |")
    print("|                                                                 |")
    print("|                                                                 |")
    print("-------------------------------------------------------------------\n")

    param_cuda     = cuda
    param_labels   = ['ap', 'pa', 'female', 'male', 'varian', 'agfa', 'ge',
            'Foreign body>.>.', 'Hilar/mediastinum>Cardiomegaly>.',
            'Lung density>Increased lung density>Atelectasis',
            'Lung density>Increased lung density>Pulmonary edema',
            'Lung density>Increased lung density>pneumonia',
            'Pleura>Pleural effusion>.', 'abnormal',
            'PatientAge']
    param_path     = None
    param_runDir   = ''
    param_type     = 0
    param_preModel = 'multitask_model'
    param_gradcam  = fl_gradcam
    param_arch     = None
    param_task     = 2
    param_clsGcam  = param_labels[-8:-2]
    param_Nens     = Nens
    fl_ensemble = False if param_Nens == 1 else True

    runtime_path, device = initialize(param_runDir, param_cuda)

    # image preprocessing
    d = DcmToPng(param_labels, dcm_path=runtime_path.joinpath('input_dir/DICOM').resolve(), png_path=runtime_path.joinpath('input_dir/IMAGEFILE').resolve(), ds=dcm_file, localOp=True, input_type=input_type)
    #d = DcmToPng(param_labels, dcm_path=runtime_path.joinpath('dicoms').resolve(), png_path=runtime_path.joinpath('pngs').resolve(), ds=dcm_file)
    if input_type == 'dicom':
        d.dcm2png()

    # start network inference
    env = DemoEnvironment(device, runtime_path, mtype=param_type, name_labels=param_labels, name_paths=param_path, name_model=param_arch, task_type=param_task)
    t = Demo(env, pt_runtime=runtime_path, fn_net=param_preModel, fl_gradcam=param_gradcam, cls_gradcam=param_clsGcam, th_gradcam=th_gradcam, fl_ensemble=fl_ensemble)

    if (fl_ensemble):
        result = t.demo_cxr_ensemble_evaluation(hmp_dims=hmp_dims, n_ens=param_Nens)
    else:
        result = t.demo_cxr_evaluation(hmp_dims=hmp_dims)

    df_prob = pd.DataFrame(result['prob'], columns=print_label_name)
    df_pred = pd.DataFrame(result['pred'], columns=print_label_name)
    df_file = pd.read_csv(runtime_path.joinpath('input_dir/IMAGEFILE/images.csv'))
    df_prob['file'] = df_file['PATH']
    df_pred['file'] = df_file['PATH']

    runtime_path.joinpath('output_dir/Classification').mkdir(parents=True, exist_ok=True)
    df_prob.to_csv(runtime_path.joinpath('output_dir/Classification/probability.txt'), header=True, index=True, sep=',', mode='w')
    df_pred.to_csv(runtime_path.joinpath('output_dir/Classification/prediction.txt'), header=True, index=True, sep=',', mode='w')

    #print(result)
    return result


if __name__ == "__main__":
    with open('config_dir/config.yaml', 'r') as f:
        config = yaml.load(f)

    result = cxr_predict(input_type=config['input'], cuda=config['cuda'], fl_gradcam=config['fl_gradcam'], hmp_dims=(config['hmp_dims'], config['hmp_dims']), Nens=config['Nens'])

