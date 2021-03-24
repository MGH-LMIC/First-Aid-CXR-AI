import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm

# Base model from DenseNet-121
class BaseModel(nn.Module):
    def __init__(self, in_dim=1, out_dim=31, name_model=None, pretrained=False, tf_learning=None):
        super().__init__()

        if name_model == None:
            self.main = tvm.densenet121(pretrained=True)
            self.main.features.conv0 = nn.Conv2d(in_dim, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.main.classifier = nn.Linear(self.main.classifier.in_features, out_dim)
            if (pretrained==True):
                self.load_model(tf_learning)
        elif name_model == 'densenet161':
            self.main = tvm.densenet161(pretrained=True)
            self.main.features.conv0 = nn.Conv2d(in_dim, 96, kernel_size=7, stride=2, padding=3, bias=False)
            self.main.classifier = nn.Linear(self.main.classifier.in_features, out_dim)
        elif name_model == 'resnet152':
            self.main = tvm.resnet152(pretrained=True)
            self.main.conv1 = nn.Conv2d(in_dim, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.main.fc = nn.Linear(self.main.fc.in_features, out_dim)
        elif name_model == 'resnext101_32x8d':
            self.main = tvm.resnext101_32x8d(pretrained=True)
            self.main.conv1 = nn.Conv2d(in_dim, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.main.fc = nn.Linear(self.main.fc.in_features, out_dim)
        else:
            print("this architecture is not supported")
            exit(-1)

        # Find total parameters and trainable parameters
        total_params = sum(p.numel() for p in self.main.parameters())
        print(f'{total_params:,} total parameters.')
        total_trainable_params = sum(p.numel() for p in self.main.parameters() if p.requires_grad)
        print(f'{total_trainable_params:,} training parameters.')

    def load_model(self, path):
        # DataParallel make different layer names
        states = torch.load(path.resolve())
        new = list(states.items())
        my_states = self.main.state_dict()
        count=0
        for key, value in my_states.items():
            layer_name, weights = new[count]
            if my_states[key].shape == weights.shape:
                my_states[key] = weights
            else:
                print(f'pretrained weight skipping due to different shape: {key}')
            count+=1


    def forward(self, x):
        x = self.main(x)

        return x

# Base model with patient information from DenseNet-121
class ExtdModel(nn.Module):
    def __init__(self, in_dim=1, out_dim=31, name_model=None, pretrained=False, tf_learning=None):
        super().__init__()
        self.arch = name_model
        nm_clinic_input = 2
        if name_model == None:
            self.main = tvm.densenet121(pretrained=True)
            self.main.features.conv0 = nn.Conv2d(in_dim, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.main.classifier = StackClassifier(in_dim=(self.main.classifier.in_features+nm_clinic_input), out_dim=out_dim)
            if (pretrained==True):
                self.load_model(tf_learning)
        elif name_model == 'densenet161':
            self.main = tvm.densenet161(pretrained=True)
            self.main.features.conv0 = nn.Conv2d(in_dim, 96, kernel_size=7, stride=2, padding=3, bias=False)
            self.main.classifier = StackClassifier(in_dim=(self.main.classifier.in_features+nm_clinic_input), out_dim=out_dim)
            if (pretrained==True):
                self.load_model(tf_learning)
        elif name_model == 'resnet152':
            self.main = tvm.resnet152(pretrained=True)
            self.main.conv1 = nn.Conv2d(in_dim, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.main.fc = StackClassifier(in_dim=(self.main.fc.in_features+nm_clinic_input), out_dim=out_dim)
            if (pretrained==True):
                self.load_model(tf_learning)
        elif name_model == 'resnext101_32x8d':
            self.main = tvm.resnext101_32x8d(pretrained=True)
            self.main.conv1 = nn.Conv2d(in_dim, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.main.fc = StackClassifier(in_dim=(self.main.fc.in_features+nm_clinic_input), out_dim=out_dim)
            if (pretrained==True):
                self.load_model(tf_learning)
        else:
            print("this architecture is not supported")
            exit(-1)

        # Find total parameters and trainable parameters
        total_params = sum(p.numel() for p in self.main.parameters())
        print(f'{total_params:,} total parameters.')
        total_trainable_params = sum(p.numel() for p in self.main.parameters() if p.requires_grad)
        print(f'{total_trainable_params:,} training parameters.')

    def load_model(self, path):
        # DataParallel make different layer names
        states = torch.load(path.resolve())
        new = list(states.items())
        my_states = self.main.state_dict()
        count=0
        for key, value in my_states.items():
            try:
                layer_name, weights = new[count]
                if my_states[key].shape == weights.shape:
                    my_states[key] = weights
                else:
                    print(f'pretrained weight skipping due to different shape: {key}')
            except:
                continue

            count+=1

    def forward(self, x, z):
        if self.arch == 'resnext101_32x8d':
            x = self.main.conv1(x)
            x = self.main.bn1(x)
            x = self.main.relu(x)
            x = self.main.maxpool(x)

            x = self.main.layer1(x)
            x = self.main.layer2(x)
            x = self.main.layer3(x)
            x = self.main.layer4(x)
            x = self.main.avgpool(x)

            x = torch.flatten(x, 1)
            out = torch.cat((x, z), dim=1)
            out = self.main.fc(out)
        else:
            features = self.main.features(x)
            out = F.relu(features, inplace=True)
            out = F.adaptive_avg_pool2d(out, (1, 1))
            out = torch.flatten(out, 1)
            out = torch.cat((out, z), dim=1)
            out = self.main.classifier(out)

        return out

class StackClassifier(nn.Module):
    def __init__(self, in_dim=31, hid_dim=35, out_dim=31):
        super().__init__()

        if (True):
            self.main = nn.Sequential(
                    nn.Linear(in_dim, hid_dim, bias=True),
                    #nn.BatchNorm1d(hid_dim),
                    #nn.PReLU(hid_dim),
                    nn.ReLU(True),
                    #nn.BatchNorm1d(hid_dim),
                    #nn.Dropout(p=0.2),
                    nn.Linear(hid_dim, out_dim, bias=True),
            )
        else:
            self.main = nn.Sequential(
                    nn.Linear(in_dim,out_dim, bias=True),
            )
        print(self.main)


    def forward(self, x):
        x = self.main(x)

        return x

class ResidualClassifier(nn.Module):
    def __init__(self, in_dim=31, hid_dim=10, out_dim=31):
        super().__init__()

        self.classifier1 = nn.Sequential(
                nn.Linear(in_dim, hid_dim, bias=True),
                nn.BatchNorm1d(hid_dim),
                nn.ReLU(True),
        )
        self.classifier2 = nn.Linear((hid_dim+1), out_dim, bias=True)
        print(self.classifier1)
        print(self.classifier2)

    def forward(self, x, z):
        x = self.classifier1(x)
        x = torch.cat((x, (z/100.0)), dim=1)
        x = self.classifier2(x)
        return x

# two inputs Base model from DenseNet-121
class TwoInputBaseModel(nn.Module):
    def __init__(self, in_dim=1, out_dim=29):
        super().__init__()

        orig_model = tvm.densenet121(pretrained=True)
        self.part01 = orig_model.features
        self.part01.conv0 = nn.Conv2d(in_dim, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.part02 = orig_model.features
        self.part02.conv0 = nn.Conv2d(in_dim, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.classifier = nn.Linear(2*orig_model.classifier.in_features, out_dim)

    def forward(self, x1, x2):
        x1 = self.part01(x1)
        x1 = F.relu(x1, inplace=True)
        x1 = F.adaptive_avg_pool2d(x1, (1, 1))
        x1 = torch.flatten(x1, 1)

        x2 = self.part02(x2)
        x2 = F.relu(x2, inplace=True)
        x2 = F.adaptive_avg_pool2d(x2, (1, 1))
        x2 = torch.flatten(x2, 1)

        x = torch.cat((x1, x2), dim=1)
        x = self.classifier(x)

        return x

# in order to import any different types of pretrained network
class DiseaseModel(nn.Module):
    def __init__(self, in_dim=1, out_dim=1):
        super().__init__()

        self.main = tvm.densenet121(pretrained=True)
        self.main.features.conv0 = nn.Conv2d(in_dim, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.main.classifier = nn.Linear(self.main.classifier.in_features, out_dim)

    def forward(self, x):
        x = self.main(x)

        return x

class DiseaseClassifier(nn.Module):
    def __init__(self, in_dim=30, hid_dim=30, out_dim=1):
        super().__init__()

        # finding model from simulations
        self.main = nn.Sequential(
                nn.Linear(in_dim, hid_dim, bias=False),
                nn.ReLU(True),
                nn.Linear(hid_dim, hid_dim, bias=False),
                nn.ReLU(True),
                nn.Linear(hid_dim, out_dim, bias=False),
        )

    def forward(self, x):
        x = self.main(X)

        return x


class DiseaseNetwork(nn.Module):
    def __init__(self, net_type=0, base_out=29, dis_out=1, net_in=1, net_out=1):
        super().__init__()

        self.type = net_type
        if (self.type == 0):
            self.base = BaseModel(net_in, base_out)
        elif (self.type == 1):
            self.disease = DiseaseModel(net_in, dis_out)
        elif (self.type == 2): # To Fully deep learning model
            self.base = BaseModel(net_in, base_out)
            self.disease = DiseaseModel(net_in, dis_out)
            self.cls = DiseaseClassifier(base_out+dis_out, base_out+dis_out, net_out)
        elif (self.type == 3): # To Random Forest Classifier
            self.base = BaseModel(net_in, base_out)
            self.disease = DiseaseModel(net_in, dis_out)
        else:
            raise NameError(self.type)

    def forward(self, x):
        if (self.type == 0):
            x = self.base(x)
        elif (self.type == 1):
            x = self.disease(x)
        elif (self.type == 2): # To Fully deep learning model
            x1 = self.base(x)
            x2 = self.disease(x)
            x = torch.cat((F.sigmoid(x1), F.sigmoid(x2)), dim=1)
            x = self.cls(x)
        elif (self.type == 3): # To Random Forest Classifier
            x1 = self.base(x)
            x2 = self.disease(x)
            x = torch.cat((F.sigmoid(x1), F.sigmoid(x2)), dim=1)

        return x


class VaeModel(nn.Module):
    """Convolutional Variational Autoencoder (Conv VAE)

    Parameters
    ----------
    nm_ch : Number of channels of input image (default, 1)
    nm_flt: Number of filters used in first convolutional layer (default, 8)
    dm_lat: Depth of latent space (default, 20)
    ------
    x        : Reconstructed image
    z_mean   : Mean
    z_log_var: Variance
    """

    def __init__(self, nm_ch=1, nm_flt=8, dm_lat=128):
        super().__init__()

        self.conv1 = nn.Conv2d(nm_ch, nm_flt, kernel_size = 3, stride = 2, padding = 1)
        self.conv2 = nn.Conv2d(nm_flt, nm_flt * 2, kernel_size = 3, stride = 2, padding = 1)
        self.conv3 = nn.Conv2d(nm_flt * 2, nm_flt * 4, kernel_size = 3, stride = 2, padding = 1)
        self.conv4 = nn.Conv2d(nm_flt * 4, nm_flt * 8, kernel_size = 3, stride = 2, padding = 1)
        self.conv5 = nn.Conv2d(nm_flt * 8, nm_flt * 16, kernel_size = 3, stride = 2, padding = 1)

        # TODO: Change hardcoded dims below to more universal approach
        self.z_mean = nn.Linear(128 * 16 * 16, dm_lat)
        self.z_log_var = nn.Linear(128 * 16 * 16, dm_lat)

        self.decode_fc = nn.Linear(dm_lat, 128 * 16 * 16)

        self.convt1 = nn.ConvTranspose2d(nm_flt * 16, nm_flt * 8, kernel_size = 4, stride = 2, padding = 1)
        self.convt2 = nn.ConvTranspose2d(nm_flt * 8, nm_flt * 4, kernel_size = 4, stride = 2, padding = 1)
        self.convt3 = nn.ConvTranspose2d(nm_flt * 4, nm_flt * 2, kernel_size = 4, stride = 2, padding = 1)
        self.convt4 = nn.ConvTranspose2d(nm_flt * 2, nm_flt, kernel_size = 4, stride = 2, padding = 1)
        self.convt5 = nn.ConvTranspose2d(nm_flt, nm_ch, kernel_size = 4, stride = 2, padding = 1)

    def encoder(self, x):
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))
        x = F.leaky_relu(self.conv4(x))
        x = F.leaky_relu(self.conv5(x))

        x = x.view(x.size()[0], -1)

        z_mean, z_log_var = self.z_mean(x), self.z_log_var(x)

        x = self.reparametrize(z_mean, z_log_var)

        return x, z_mean, z_log_var

    def decoder(self, x):
        x = self.decode_fc(x)

        x = x.view(x.size()[0], 128, 16, 16)

        x = F.leaky_relu(self.convt1(x))
        x = F.leaky_relu(self.convt2(x))
        x = F.leaky_relu(self.convt3(x))
        x = F.leaky_relu(self.convt4(x))
        x = torch.sigmoid(self.convt5(x))

        return x

    def reparametrize(self, mu, log_var):
        """The reparametrization trick"""
        std = torch.exp(log_var/2)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        x, z_mean, z_log_var = self.encoder(x)
        x = self.decoder(x)
        return x, z_mean, z_log_var

