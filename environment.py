import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from pathlib import Path

from utils import logger, print_versions, get_devices
from model import BaseModel, TwoInputBaseModel, VaeModel, ExtdModel
from data import CxrDataset, MGH_DATA_BASE
from torch.utils.data import DataLoader

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

print_label_name   = ['ap', 'pa', 'female', 'male', 'varian', 'agfa', 'ge',
        'foreignbody', 'cardiomegaly',
        'atelectasis',
        'pulmonaryedema',
        'pneumonia',
        'pleuraleffusion',
        'abnormal',
        'age']

def initialize(param_runDir, cuda_id):

    runtime_path = Path(param_runDir).resolve()

    logger.handlers.clear()
    # set logger
    log_file = f"demo_log.log"
    logger.set_log_to_stream()
    logger.set_log_to_file(runtime_path.joinpath(log_file))

    # print versions after logger.set_log_to_file() to log them into file
    print_versions()
    #logger.info(f"runtime commit: {get_commit()}")
    logger.info(f"runtime path: {runtime_path}")

    random_seed = 20      #for Pytorch 1.2.0 + DenseNet121 #basic
    logger.info(f"random seed for reproducible: {random_seed}")
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # check gpu device
    device = get_devices(cuda_id)

    return runtime_path, device


class BaseEnvironment:

    def __init__(self, device, mtype=0, in_dim=1, out_dim=31, model_file=None, tf_learning=None, name_model=None):
        self.device = device[0]
        self.type = mtype
        self.num_gpu = len(device)
        pretrained = False if tf_learning == None else True

        if mtype == 0:
            self.model = BaseModel(in_dim, out_dim, name_model=name_model, pretrained=pretrained, tf_learning=tf_learning)
        elif mtype == 1:
            self.model = TwoInputBaseModel(in_dim, out_dim)
        elif mtype == 2:
            self.model = VaeModel(nm_ch=in_dim, dm_lat=out_dim)
        elif mtype == 3:
            self.model = ExtdModel(in_dim=in_dim, out_dim=out_dim, name_model=name_model, pretrained=pretrained, tf_learning=tf_learning)
        else:
            raise RuntimeError

        if model_file is not None:
            self.load_model(model_file)

        #if  len(device) > 1:
        #    print("Let's use", len(device), "GPUs!")
        #    self.model = nn.DataParallel(self.model, device_ids=device)
        self.model = nn.DataParallel(self.model, device_ids=device)

        self.model.to(self.device)

    def load_model(self, filename):
        filepath = Path(filename).resolve()
        logger.debug(f"loading the model from {filepath}")
        states = torch.load(filepath, map_location=self.device)
        try:
            self.model.load_state_dict(states, strict=True)
        except:
            model_dict = self.model.state_dict()
            # 1. filter out unnecessary keys
            states = {k: v for k, v in states.items() if k in model_dict}
            # 2. overwrite entries in the existing state dict
            model_dict.update(states)
            # 3. load the new state dict
            self.model.load_state_dict(model_dict)


class DemoEnvironment(BaseEnvironment):

    def __init__(self, device, pt_runtime, mtype=0, in_dim=1, name_labels=None, name_paths=None, nm_csv=None, name_model=None, task_type=0):
        out_dim = len(name_labels)
        self.name_labels = name_labels
        self.nm_csv = nm_csv
        self.arch = name_model
        self.task_type = task_type
        super().__init__(device, mtype=mtype, in_dim=in_dim, out_dim=out_dim, name_model=name_model)

        if (nm_csv == None):
            self.png_path = pt_runtime.joinpath('input_dir/IMAGEFILE').resolve()

            mode = 'single' if self.type == 0 else 'extd'
            self.demo_set = CxrDataset(self.png_path, 'images.csv', num_labels=out_dim, name_labels=name_labels, name_paths=name_paths, mode=mode, ext_data=True, fl_balance=False)
        else:
            #[doyun] starting from pngs
            self.png_path = MGH_DATA_BASE

            mode = 'single' if self.type == 0 else 'extd'
            self.demo_set = CxrDataset(self.png_path, nm_csv, num_labels=out_dim, name_labels=name_labels, name_paths=name_paths, mode=mode, fl_balance=False)

        self.setup_dataset()

    def setup_dataset(self):
        pin_memory = True if self.device.type == 'cuda' else False
        self.test_loader = DataLoader(self.demo_set, batch_size=1, num_workers=0, shuffle=False, pin_memory=pin_memory)
        self.gradcam_loader = DataLoader(self.demo_set, batch_size=1, num_workers=0, shuffle=False, pin_memory=pin_memory)

        self.labels = self.demo_set.labels
        self.out_dim = len(self.labels)

        nm_count = len(self.test_loader.dataset)
        logger.info(f"using ({nm_count}) cases for demo")

