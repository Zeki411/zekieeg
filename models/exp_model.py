import torch
from torch import nn
import torch.nn.functional as F
from typing import Dict, Tuple, Union
from mmengine.model import BaseModel


class TemporalInceptionBlock(nn.Module):
    def __init__(self,
                 in_channels: int = 128,
                 num_T: int = 8,
                 sampling_rate: int = 128,
                 ):
        super(TemporalInceptionBlock, self).__init__()
        
        self.inception_window = [0.5, 0.25, 0.125]
        self.pool = 8
        # by setting the convolutional kernel being (1,lenght) and the strids being 1 we can use conv2d to
        # achieve the 1d convolution operation
        self.TConv1 = self.tconv_block(in_channels, num_T, (1, int(self.inception_window[0] * sampling_rate)), 1, self.pool)
        self.TConv2 = self.tconv_block(in_channels, num_T, (1, int(self.inception_window[1] * sampling_rate)), 1, self.pool)
        self.TConv3 = self.tconv_block(in_channels, num_T, (1, int(self.inception_window[2] * sampling_rate)), 1, self.pool)
        self.TBatchNorm = nn.BatchNorm2d(num_T)
    
    def tconv_block(self, in_channels: int, out_channels: int, kernel: int, stride: int, pool_kernel: int) -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel, stride=stride),
            nn.LeakyReLU(), nn.AvgPool2d(kernel_size=(1, pool_kernel), stride=(1, pool_kernel)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.TConv1(x)
        out = y
        y = self.TConv2(x)
        out = torch.cat((out, y), dim=-1)
        y = self.TConv3(x)
        out = torch.cat((out, y), dim=-1)
        out = self.TBatchNorm(out)
        return out

class SpatialInceptionBlock(nn.Module):
    def __init__(self, 
                in_channels: int = 1,
                num_S: int = 8,
                feature_chn_width: int = 1):
        
        super(SpatialInceptionBlock, self).__init__()

        self.pool = 8
        self.scales = [3, 5]

        self.SConv1 = self.sconv_block(in_channels, num_S, (1*self.scales[0], feature_chn_width*self.scales[0],), 
                                       (1, feature_chn_width), int(self.pool * 0.25))
        self.SConv2 = self.sconv_block(in_channels, num_S, (1*self.scales[1], feature_chn_width*self.scales[1],),
                                        (1, feature_chn_width), int(self.pool * 0.25))
        self.SBatchNorm = nn.BatchNorm2d(num_S)

    def sconv_block(self, in_channels: int, out_channels: int, kernel: int, stride: int, pool_kernel: int) -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel, stride=stride),
            nn.LeakyReLU(), nn.AvgPool2d(kernel_size=(1, pool_kernel), stride=(1, pool_kernel)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.SConv1(x)
        out = y
        y = self.SConv2(x)
        out = torch.cat((
            out.reshape(out.shape[0], out.shape[1], 1, -1), 
            y.reshape(y.shape[0], y.shape[1], 1, -1)
        ), dim=-1)
        out = self.SBatchNorm(out)
        return out

class ExpModel(nn.Module):
    def __init__(self,
                 sampling_rate: int = 128,
                 num_electrodes: int = 28,
                 chunk_size: int = 128,
                 channel_location_dict: Dict[str, Tuple[int, int]] = None,
                 in_channels: int = 1,
                 num_classes: int = 2,
                 num_T: int = 8,
                 num_S: int = 8,
                 hid_channels: int = 32,
                 dropout: float = 0.5
                 ):
        
        super(ExpModel, self).__init__()

        self.num_electrodes = num_electrodes
        self.sampling_rate = sampling_rate
        self.in_channels = in_channels
        self.chunk_size = chunk_size
        self.channel_loc = channel_location_dict
        self.num_classes = num_classes
        self.num_T = num_T
        self.num_S = num_S
        self.hid_channels = hid_channels
        self.dropout = dropout

        self.channel_location_dict = channel_location_dict
        loc_x_list = []
        loc_y_list = []
        for _, locs in channel_location_dict.items():
            if locs is None:
                continue
            (loc_y, loc_x) = locs
            loc_x_list.append(loc_x)
            loc_y_list.append(loc_y)
        self.grid_width = max(loc_x_list) + 1
        self.grid_height = max(loc_y_list) + 1

        # define blocks

        self.TIBlock = TemporalInceptionBlock(in_channels, num_T, sampling_rate)
        # Temporarily create a mock input to determine the dimensionality
        # required for the SpatialInceptionBlock, without calling self.tiblock_dim()
        # which would prematurely invoke self.forward()
        with torch.no_grad():
            mock_input = torch.zeros(1, self.in_channels, num_electrodes, self.chunk_size)
            mock_output = self.TIBlock(mock_input)
            tiblock_dim = mock_output.shape[-1]  # Assuming you want the last dimension

        self.SIBlock = SpatialInceptionBlock(num_T, num_S, tiblock_dim)

        self.fc = nn.Sequential(nn.Linear(num_S, hid_channels), nn.ReLU(), nn.Dropout(dropout),
                                nn.Linear(hid_channels, num_classes))



    def conv_block(self, in_channels: int, out_channels: int, kernel: int, stride: int, pool_kernel: int) -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel, stride=stride),
            nn.LeakyReLU(), nn.AvgPool2d(kernel_size=(1, pool_kernel), stride=(1, pool_kernel)))


    def to_grid(self, features: torch.Tensor) -> torch.Tensor:

        batch_size = features.shape[0]
        num_filters = features.shape[1]
        # num_electrodes = features.shape[2]
        num_ftpoints = features.shape[3]
        
        # reshape
        features = features.permute(2, 0, 1, 3)

        # filtes x num_electrodes x feature_points
        outputs = torch.zeros([self.grid_height, self.grid_width, batch_size, num_filters, num_ftpoints], device=features.device, dtype=features.dtype)

        # 9 x 9 x feature_points
        for i, locs in enumerate(self.channel_location_dict.values()):
            if locs is None:
                continue
            (loc_y, loc_x) = locs
            outputs[loc_y][loc_x] = features[i]

        # reshape to batch_size x num_filters x 9 x (num_ftpoints x 9)
        outputs = outputs.permute(2, 3, 0, 1, 4).reshape(batch_size, num_filters, self.grid_height, num_ftpoints * self.grid_width)
        return outputs

    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.TIBlock(x)
        out = self.to_grid(out)
        out = self.SIBlock(out)
        out = torch.squeeze(torch.mean(out, dim=-1), dim=-1)
        out = self.fc(out)

        return out



class MMExpModel(BaseModel):
    def __init__(self,
                 sampling_rate: int = 128,
                 num_electrodes: int = 28,
                 chunk_size: int = 128,
                 channel_location_dict: Dict[str, Tuple[int, int]] = None,
                 in_channels: int = 1,
                 num_classes: int = 2,
                 num_T: int = 8,
                 num_S: int = 8,
                 hid_channels: int = 32,
                 dropout: float = 0.5
                 ):
        super().__init__()

        self.exp_model = ExpModel(sampling_rate, 
                              num_electrodes, 
                              chunk_size, 
                              channel_location_dict, 
                              in_channels, 
                              num_classes, 
                              num_T, 
                              num_S, 
                              hid_channels, 
                              dropout).double()
    
    def forward(self, eeg_epochs, labels, mode):
        outputs = self.exp_model(eeg_epochs)
        if mode == 'loss':
            return {'loss': F.cross_entropy(outputs, labels)}
        elif mode == 'predict':
            return outputs, labels
        






    


        