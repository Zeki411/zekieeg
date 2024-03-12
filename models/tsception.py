import torch 
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel: int, stride: int, pool_kernel: int):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel, stride=stride)
        self.leakyrelu = nn.LeakyReLU()
        self.avgpool = nn.AvgPool2d(kernel_size=(1, pool_kernel), stride=(1, pool_kernel))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.leakyrelu(x)
        x = self.avgpool(x)
        return x
    
class BackBone(nn.Module):
    def __init__(self, num_electrodes: int, num_T: int, num_S: int, in_channels: int, hid_channels: int, num_classes: int, sampling_rate: int, dropout: float):
        super(BackBone, self).__init__()
        self.num_electrodes = num_electrodes
        self.num_T = num_T
        self.num_S = num_S
        self.in_channels = in_channels
        self.hid_channels = hid_channels
        self.num_classes = num_classes
        self.sampling_rate = sampling_rate
        self.dropout = dropout

        self.inception_window = [0.5, 0.25, 0.125]
        self.pool = 8
        # by setting the convolutional kernel being (1,lenght) and the strids being 1 we can use conv2d to
        # achieve the 1d convolution operation
        self.Tception1 = ConvBlock(in_channels, num_T, (1, int(self.inception_window[0] * sampling_rate)), 1, self.pool)
        self.Tception2 = ConvBlock(in_channels, num_T, (1, int(self.inception_window[1] * sampling_rate)), 1, self.pool)
        self.Tception3 = ConvBlock(in_channels, num_T, (1, int(self.inception_window[2] * sampling_rate)), 1, self.pool)

        self.BN_t = nn.BatchNorm2d(num_T)

        self.Sception1 = ConvBlock(num_T, num_S, (int(num_electrodes), 1), 1, int(self.pool * 0.25))
        self.Sception2 = ConvBlock(num_T, num_S, (int(num_electrodes * 0.5), 1), (int(num_electrodes * 0.5), 1), int(self.pool * 0.25))

        self.BN_s = nn.BatchNorm2d(num_S)
 

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.Tception1(x)
        out = y
        y = self.Tception2(x)
        out = torch.cat((out, y), dim=-1)
        y = self.Tception3(x)
        out = torch.cat((out, y), dim=-1)
        out = self.BN_t(out)
        z = self.Sception1(out)
        out_ = z
        z = self.Sception2(out)
        out_ = torch.cat((out_, z), dim=2)
        out = self.BN_s(out_)

        return out
    
class ClassifierHead(nn.Module):
    def __init__(self, num_S: int, hid_channels: int, num_classes: int, dropout: float):
        super(ClassifierHead, self).__init__()
        self.fusion_layer = ConvBlock(num_S, num_S, (3, 1), 1, 4)
        self.BN_fusion = nn.BatchNorm2d(num_S)
        # self.GloAvgPool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(nn.Linear(num_S, hid_channels), nn.ReLU(), nn.Dropout(dropout),
                                nn.Linear(hid_channels, num_classes))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.fusion_layer(x)
        out = self.BN_fusion(out)
        out = torch.squeeze(torch.mean(out, dim=-1), dim=-1)
        out = self.fc(out)
        return out
    
class TSCeption(nn.Module):
    def __init__(self, num_classes: int = 2, num_electrodes: int = 32, num_T: int = 15, num_S: int = 15, in_channels: int = 1, hid_channels: int = 32, sampling_rate: int = 128, dropout: float = 0.5):
        # input_size: 1 x EEG channel x datapoint
        super(TSCeption, self).__init__()
        self.backbone = BackBone(num_electrodes, num_T, num_S, in_channels, hid_channels, num_classes, sampling_rate, dropout)
        self.classifier_head = ClassifierHead(num_S, hid_channels, num_classes, dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.backbone(x)
        out = self.classifier_head(out)
        return out
    

'''-----------------------------------------------------------------------------------------------------------------'''
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class SEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None,
                 *, reduction=16):
        super(SEBasicBlock, self).__init__()
        self.conv1 = nn.Conv1d(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(planes, planes, 1)
        self.bn2 = nn.BatchNorm1d(planes)
        self.se = SELayer(planes, reduction)
        self.downsample = downsample
        self.stride = stride
        

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class AFR(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, blocks: int = 1):
        super(AFR, self).__init__()
        self.inplanes = in_channels
        self.AFR = self._make_layer(SEBasicBlock, out_channels, blocks)


    def _make_layer(self, block, planes, blocks, stride=1):  # makes residual SE block
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.squeeze()
        x = self.AFR(x)
        return x


class Exp1ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel: int, stride: int, pool_kernel: int):
        super(Exp1ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel, stride=stride)
        # self.activation = nn.GELU()
        self.activation = nn.LeakyReLU()
        if pool_kernel == 0:
            self.avgpool = nn.Identity()
        else:
            self.avgpool = nn.AvgPool2d(kernel_size=(1, pool_kernel), stride=(1, pool_kernel))
        
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.activation(x)
        x = self.avgpool(x)
        return x
    
class Exp1BackBone(nn.Module):
    def __init__(self, num_electrodes: int, num_T: int, num_S: int, in_channels: int, sampling_rate: int):
        super(Exp1BackBone, self).__init__()
        self.num_electrodes = num_electrodes
        self.num_T = num_T
        self.num_S = num_S
        self.in_channels = in_channels
        self.sampling_rate = sampling_rate

        self.inception_window = [0.5, 0.25, 0.125]
        self.pool = 8
        # by setting the convolutional kernel being (1,lenght) and the strids being 1 we can use conv2d to
        # achieve the 1d convolution operation
        self.Tception1 = Exp1ConvBlock(in_channels, num_T, (1, int(self.inception_window[0] * sampling_rate)), 1, self.pool)
        self.Tception2 = Exp1ConvBlock(in_channels, num_T, (1, int(self.inception_window[1] * sampling_rate)), 1, self.pool)
        self.Tception3 = Exp1ConvBlock(in_channels, num_T, (1, int(self.inception_window[2] * sampling_rate)), 1, self.pool)

        self.BN_t = nn.BatchNorm2d(num_T)

        self.Sception1 = Exp1ConvBlock(num_T, num_S, (int(num_electrodes), 1), 1, int(self.pool * 0.25))
        self.Sception2 = Exp1ConvBlock(num_T, num_S, (int(num_electrodes * 0.5), 1), (int(num_electrodes * 0.5), 1), int(self.pool * 0.25))

        self.BN_s = nn.BatchNorm2d(num_S)

        self.fusion_layer = Exp1ConvBlock(num_S, num_S, (3, 1), 1, 4)
        self.BN_fusion = nn.BatchNorm2d(num_S)

        self.afr = AFR(num_S, num_S, 5)
        self.dropout = nn.Dropout(0.5)
        self.BN_afr = nn.BatchNorm2d(num_S)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.Tception1(x)
        out = y
        y = self.Tception2(x)
        out = torch.cat((out, y), dim=-1)
        y = self.Tception3(x)
        out = torch.cat((out, y), dim=-1)
        out = self.BN_t(out)
        z = self.Sception1(out)
        out_ = z
        z = self.Sception2(out)
        out_ = torch.cat((out_, z), dim=2)
        out = self.BN_s(out_)

        out = self.fusion_layer(out)
        out = self.BN_fusion(out)

        out = self.afr(out)
        # out = self.dropout(out)
        out = out.unsqueeze(2)
        out = self.BN_afr(out)

        return out

class Exp1ClassifierHead(nn.Module):
    def __init__(self, num_S: int, hid_channels: int, num_classes: int, dropout: float):
        super(Exp1ClassifierHead, self).__init__()

        self.fc = nn.Sequential(nn.Linear(num_S, hid_channels), nn.ReLU(), nn.Dropout(dropout),
                                nn.Linear(hid_channels, num_classes))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        out = torch.squeeze(torch.mean(out, dim=-1), dim=-1)
        out = self.fc(out)
        return out
       
class ExpClassifierHead(nn.Module):
    def __init__(self, num_S: int, hid_channels: int, num_classes: int, dropout: float):
        super(ExpClassifierHead, self).__init__()
        self.fc = nn.Sequential(nn.Linear(num_S, hid_channels), nn.ReLU(), nn.Dropout(dropout),
                        nn.Linear(hid_channels, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x.reshape(x.shape[0], x.shape[1], 1, -1)
        out = torch.squeeze(torch.mean(out, dim=-1), dim=-1)
        out = self.fc(out)
        return out


class ExpTSCeption(nn.Module):
    def __init__(self, num_classes: int = 2, num_electrodes: int = 32, num_T: int = 15, num_S: int = 15, 
                 in_channels: int = 1, hid_channels: int = 32, sampling_rate: int = 128, dropout: float = 0.5):
        # input_size: 1 x EEG channel x datapoint
        super(ExpTSCeption, self).__init__()
        self.backbone = Exp1BackBone(num_electrodes, num_T, num_S, in_channels, sampling_rate)
        self.classifier_head = Exp1ClassifierHead(num_S, hid_channels, num_classes, dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.backbone(x)
        out = self.classifier_head(out)
        return out

