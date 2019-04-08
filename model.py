import torch.nn as nn
import torch.nn.functional as F
from lossfunc.l2_linear import L2Linear
from lossfunc.arcface import ArcfaceLoss, ArcfaceLinear
from lossfunc.regular_face_linear import RegularFaceLinear
from lossfunc.coco_loss import COCOLogit
from lossfunc.asoftmax import AngleLinear

class Net(nn.Module):
    def __init__(self, config, feature_dim=2):
        super(Net, self).__init__()
        self.config = config
        self.relu = nn.PReLU()
        # self.relu = nn.ReLU()
        self.conv1_1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.conv1_2 = nn.Conv2d(32, 32, kernel_size=5, padding=2)
        self.conv2_1 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.conv2_2 = nn.Conv2d(64, 64, kernel_size=5, padding=2)
        self.conv3_1 = nn.Conv2d(64, 128, kernel_size=5, padding=2)
        self.conv3_2 = nn.Conv2d(128, 128, kernel_size=5, padding=2)
        self.ip1 = nn.Linear(128*3*3, feature_dim)

        if 'MAIN_LOSS' in config:
            if config.MAIN_LOSS.TYPE == 'L2-softmax':
                self.ip2 = L2Linear(feature_dim, 10, config.MAIN_LOSS.ALPHA)
            elif config.MAIN_LOSS.TYPE == 'arcface':
                self.ip2 = ArcfaceLinear(feature_dim, 10, config.MAIN_LOSS.M)
            # elif config.MAIN_LOSS.TYPE == 'regularface':
            #     self.ip2 = RegularFaceLinear(feature_dim, 10)
            elif config.MAIN_LOSS.TYPE == 'coco_loss':
                self.ip2 = COCOLogit(feature_dim, 10, config.MAIN_LOSS.ALPHA)
            elif config.MAIN_LOSS.TYPE == 'a-softmax':
                self.ip2 = AngleLinear(feature_dim, 10, config.MAIN_LOSS.M)
            elif config.MAIN_LOSS.TYPE == 'softmax':
                self.ip2 = nn.Linear(feature_dim, 10, bias=False)
        else:
            self.ip2 = nn.Linear(feature_dim, 10, bias=False)

    def forward(self, x, label):
        x = self.relu(self.conv1_1(x))
        x = self.relu(self.conv1_2(x))
        x = F.max_pool2d(x,2)
        x = self.relu(self.conv2_1(x))
        x = self.relu(self.conv2_2(x))
        x = F.max_pool2d(x,2)
        x = self.relu(self.conv3_1(x))
        x = self.relu(self.conv3_2(x))
        x = F.max_pool2d(x,2)
        x = x.view(-1, 128*3*3)
        ip1 = self.ip1(x)
        ip2 = self.ip2(ip1)

        return ip1, ip2