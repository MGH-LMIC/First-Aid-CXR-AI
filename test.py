import pickle
import numpy as np
import pandas as pd
from pathlib import Path

from utils import logger
from environment import print_label_name
from gradcam import GradCam, save_class_activation_images
from data import CxrDataset

ATLAS_GEN = False
atlas_name = 'fracture'
# 'diaphragm', 'cardiomegaly', 'atelectasis', 'pulmonary_edema', 'pneumonia', 'decreased_lung_volume',
# 'pleural_effusion', 'fracture', 'other_int_opacity', 'pneumothorax'

class Tester:
    def __init__(self, env, pt_runtime="test", fn_net=None, fl_gradcam=False, cls_gradcam=None, id_prob=None, fl_ensemble=False, fl_exai=False, f_name='sim'):
        self.env = env
        self.pt_runtime = pt_runtime
        self.fl_prob = False if id_prob == None else True
        self.id_prob = id_prob
        self.f_name = f_name
        self.fl_ensemble = fl_ensemble
        # for multiple class and binary label tasks
        self.pf_metric = {
                'loss': [],
                'accuracy': [],
                'sensitivity': [],
                'specificity': [],
                'auc_score': [],
                'ap_score': [],
                'mse_score': []
        }
        self.fn_net = fn_net
        self.fl_gradcam = fl_gradcam
        self.cls_gradcam = cls_gradcam
        self.th_gradcam = 0.9
        self.fl_gradcam_save = True

        #explainable methods
        self.fl_exai = fl_exai
        if self.fl_exai:
            self.fl_gradcam = True
            self.cls_gradcam = [
                        'Hilar/mediastinum>Cardiomegaly>.',
                        'Lung density>Increased lung density>Atelectasis',
                        'Lung density>Increased lung density>Pulmonary edema',
                        'Lung density>Increased lung density>pneumonia',
                        'Pleura>Pleural effusion>.'
                    ]
            self.th_gradcam = 0.5
            self.ex_method = EX_AI(env, pt_runtime=pt_runtime, thr=0.5, f_name=f_name)

    def load(self):
        pt_file = self.pt_runtime.joinpath(f'train.pkl')
        with open(pt_file, 'rb') as f:
            self.pf_metric = pickle.load(f)


    def test_batch(self, tp_data, fl_input=False):
        # to support different types of models.
        if self.env.type == 0:
            data = tp_data[0]
            target = tp_data[1]
            info = tp_data[2]
            data, target, info = data.to(self.env.device), target.to(self.env.device), info.to(self.env.device)
            #data, target = data.to(self.env.device), target.to(self.env.device)
            #network output
            output = self.env.model(data)
        elif self.env.type == 1:
            data1 = tp_data[0]
            data2 = tp_data[1]
            target = tp_data[2]
            data1, data2, target = data1.to(self.env.device), data2.to(self.env.device), target.to(self.env.device)
            #network output
            output = self.env.model(data1, data2)
        elif self.env.type == 3:
            data = tp_data[0]
            target = tp_data[1]
            info = tp_data[2]
            data, target, info = data.to(self.env.device), target.to(self.env.device), info.to(self.env.device)
            #network output
            output = self.env.model(data, info)

        if fl_input == False:
            return output, target
        else:
            return data, info, output


    def gradcam_data(self, test_loader, hmp_dims=(512,512), ens_flg=False, cams_ens=None, prob_ens=None):
        # threshold to draw a heatmap
        out_dim = self.env.out_dim

        CxrDataset.eval()
        self.env.model.eval()
        #with torch.no_grad():
        gradcam_res_list  = []
        gradcam_path_list = []

        cams = np.zeros((len(test_loader), len(self.cls_gradcam), 16, 16))

        for batch_idx, (data, target, info) in enumerate(test_loader):
            #data, target = data.to(self.env.device), target.to(self.env.device)
            data, target, info = data.to(self.env.device), target.to(self.env.device), info.to(self.env.device)
            # Grad CAM
            grad_cam = GradCam(self.env.model, self.env.type)
            if self.fl_ensemble:
                cam = self.gradcam_save_argcls_ens(grad_cam, data, test_loader, batch_idx, hmp_dims, info, ens_flg=ens_flg, cams_ens=cams_ens, prob_ens=prob_ens)
            else:
                gradcam_res, gradcam_path = self.gradcam_save_argcls(grad_cam, data, test_loader, batch_idx, hmp_dims, info)

            try:
                if self.fl_ensemble:
                    cams[batch_idx, :, :, :] = cam
                else:
                    gradcam_res_list.append(gradcam_res)
                    gradcam_path_list.append(gradcam_path)

            except AttributeError as e:
                print("No GradCam result?")

        return gradcam_res_list, gradcam_path_list, cams


    def gradcam_save_argcls(self, grad_cam, data, test_loader, batch_idx, hmp_dims, info):

        if self.cls_gradcam[0] == 'all':
            self.cls_gradcam = self.env.labels

        for i, nm_tcls in enumerate(self.cls_gradcam):
            ## need to implement to find index among self.env.labels from string of target class
            ## code start here!!!!
            id_tcls = self.env.labels.index(nm_tcls)
            if self.env.type == 3:
                cam, prob, tcls = grad_cam.generate_cam(data, info, target_class=id_tcls)
            else:
                cam_w = self.env.model.module.main.classifier.weight[id_tcls].cpu().detach().numpy()
                cam, prob, tcls, _ = grad_cam.generate_cam(data, target_class=id_tcls, cam_w=cam_w)
            noPlotflg = np.array([-1])
            # when we draw gradcam, we have to batch size as 1.
            file_name = test_loader.dataset.entries['PATH'][batch_idx]
            path_name = file_name.split(".")[0]+f'_{print_label_name[tcls]}'

            if prob >= self.th_gradcam:
                cam_rs = save_class_activation_images(data, cam, self.pt_runtime.joinpath('output_dir/PDF'), path_name, hmp_dims)
            #else:
            #    cam_rs = np.expand_dims(cam_rs, axis=0)
            #    cam_list = np.concatenate((cam_list, cam_rs), axis=0)
            cam_list=[]
            path_list=[]

            path_list.append(path_name)
        return cam_list, path_list

    def gradcam_save_argcls_ens(self, grad_cam, data, test_loader, batch_idx, hmp_dims, info, ens_flg=False, cams_ens=None, prob_ens=None):

        if self.cls_gradcam[0] == 'all':
            self.cls_gradcam = self.env.labels

        cams = np.zeros((len(self.cls_gradcam), 16, 16))
        for i, nm_tcls in enumerate(self.cls_gradcam):
            ## need to implement to find index among self.env.labels from string of target class
            ## code start here!!!!
            id_tcls = self.env.labels.index(nm_tcls)
            cam_w = self.env.model.module.main.classifier.weight[id_tcls].cpu().detach().numpy()

            if ens_flg == True:
                cam, prob, tcls, cam_low = grad_cam.generate_cam(data, target_class=id_tcls, cam_w=cam_w, ens_flg=True, ens_cam=cams_ens[batch_idx, i, :, :])
                cams[i, :, :] = cam_low

                noPlotflg = np.array([-1])
                # when we draw gradcam, we have to batch size as 1.
                file_name = test_loader.dataset.entries['PATH'][batch_idx]
                path_name = file_name.split(".")[0]+f'_{print_label_name[tcls]}'

                if prob_ens[batch_idx, id_tcls].item() >= self.th_gradcam:
                    cam_rs = save_class_activation_images(data, cam, self.pt_runtime.joinpath('output_dir/PDF'), path_name, hmp_dims)
            else:
                #review_cam
                cam, prob, tcls, cam_low = grad_cam.generate_cam(data, target_class=id_tcls, cam_w=cam_w, th_cam=0.5)
                cams[i, :, :] = cam_low

        return cams

