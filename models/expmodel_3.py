import torch
from torch import nn

class BasicConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel: int, stride: int, pool_kernel: int):
        super(BasicConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel, stride=stride)
        self.gelu = nn.GELU()
        self.avgpool = nn.AvgPool2d(kernel_size=(1, pool_kernel), stride=(1, pool_kernel))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.gelu(x)
        x = self.avgpool(x)
        return x
    
    
class SEBlock(nn.Module):
    def __init__(self, channel, reduction=8):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ResidualSEBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, se_reduction: int = 8, downsample=None):
        super(ResidualSEBlock, self).__init__()

        # Approximate 'same' padding for a 3x1 kernel with stride 1x1
        padding = (1, 0)  # This padding works with the assumption of stride=(1,1)
        
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 1), stride=(1, 1), padding=padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(3, 1), stride=(1, 1), padding=padding)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.se = SEBlock(out_channels, reduction=se_reduction)

        # Adjusting the shortcut connection if necessary
        self.downsample = downsample if downsample is not None else (
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels)
            ) if in_channels != out_channels else None
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
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

class BottleneckResidualSEBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, se_reduction: int = 8, downsample=None):
        super(BottleneckResidualSEBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1), stride=(1, 1))
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0))
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(1, 1), stride=(1, 1))
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.se = SEBlock(out_channels, reduction=se_reduction)

        # Adjusting the shortcut connection if necessary
        self.downsample = downsample if downsample is not None else (
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels)
            ) if in_channels != out_channels else None
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out



class expAFR(nn.Module):
    def __init__(self, in_channels: int, block = ResidualSEBlock,
                 num_blocks: int = 1, deep_scale: int = 1, se_reduction_base: int = 16):
        super(expAFR, self).__init__()

        out_channels = 0
        se_reduction = se_reduction_base  

        blocks = []
        for _ in range(num_blocks):
            out_channels = in_channels * deep_scale
            blocks.append(block(in_channels, out_channels, se_reduction=se_reduction))
            in_channels = out_channels
            se_reduction = se_reduction * deep_scale

        self.out_channels = out_channels
        self.afr_blocks = nn.Sequential(*blocks)  

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.afr_blocks(x)
        return x
    


#  Multi-Resolution Temporal Spatial Convolutional Neural Network (MR-TSCNN)
class MRTSCNN(nn.Module):
    '''
    Multi-Resolution Temporal Spatial Convolutional Neural Network (MR-TSCNN)
    '''
    def __init__(self, 
                 in_channels: int,
                 sampling_rate: int,
                 num_electrodes: int,
                 num_T: int,
                 num_S: int,
                 se_reduction: list = [0, 0]):
        super(MRTSCNN, self).__init__()
        
        self.in_channels = in_channels
        self.sampling_rate = sampling_rate
        self.num_electrodes = num_electrodes
        self.num_T = num_T
        self.num_S = num_S
        self.se_reduction = se_reduction

        self.pool = 8
        self.Twindow_ratios = [0.5, 0.25, 0.125]
        self.Tconv1 = BasicConvBlock(in_channels, num_T, (1, int(self.Twindow_ratios[0] * sampling_rate)), 1, self.pool)
        self.Tconv2 = BasicConvBlock(in_channels, num_T, (1, int(self.Twindow_ratios[1] * sampling_rate)), 1, self.pool)
        self.Tconv3 = BasicConvBlock(in_channels, num_T, (1, int(self.Twindow_ratios[2] * sampling_rate)), 1, self.pool)
        self.T_BN = nn.BatchNorm2d(num_T)

        if self.se_reduction[0] > 0:  # if se_reduction is 0, SEBlock will not be used
            self.T_SE = SEBlock(num_T, reduction=self.se_reduction[0])


        # Please make sure the electrodes organized in the order left to right or right to left
        self.Sconv1 = BasicConvBlock(num_T, num_S, (int(num_electrodes), 1), 1, int(self.pool * 0.25))
        self.Sconv2 = BasicConvBlock(num_T, num_S, (int(num_electrodes * 0.5), 1), (int(num_electrodes * 0.5), 1), int(self.pool * 0.25))
        self.S_BN = nn.BatchNorm2d(num_S)

        if self.se_reduction[1] > 0:  # if se_reduction is 0, SEBlock will not be used
            self.S_SE = SEBlock(num_S, reduction=self.se_reduction[1])

        self.TS_Fusion = BasicConvBlock(num_S, num_S, (3, 1), 1, 1)
        self.TS_BN = nn.BatchNorm2d(num_S)
        self.out_channels = num_S

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.Tconv1(x)
        x2 = self.Tconv2(x)
        x3 = self.Tconv3(x)
        x = torch.cat([x1, x2, x3], dim=-1)
        x = self.T_BN(x)
        if self.T_SE is not None:
            x = self.T_SE(x)

        x1 = self.Sconv1(x)
        x2 = self.Sconv2(x)
        x = torch.cat([x1, x2], dim=-2)
        x = self.S_BN(x)
        if self.S_SE is not None:
            x = self.S_SE(x)

        x = self.TS_Fusion(x)
        x = self.TS_BN(x)
        return x


class FeedForwardBlock(nn.Sequential):
    def __init__(self,
                 in_channels: int,
                 expansion: int = 4,
                 dropout: float = 0.):
        super().__init__(
            nn.Linear(in_channels, expansion * in_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(expansion * in_channels, in_channels),
        )
    
class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int = 1, mha_dropout_rate: float = 0.0, ff_expansion: int = 4,  ff_dropout_rate: float = 0.0):
        super(TransformerEncoderBlock, self).__init__()
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, dropout=mha_dropout_rate)
        self.ff = FeedForwardBlock(embed_dim, ff_expansion, ff_dropout_rate)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mha(x, x, x)[0] + x
        x = self.ff(x) + x
        return x
      
        
class TransformerEncoder(nn.Module):
    """
    
    """
    def __init__(self, depth: int, embed_dim: int, num_heads: int, mha_dropout_rate: float = 0.5, ff_expansion: int = 4, ff_dropout_rate: float = 0.5):
        super(TransformerEncoder, self).__init__()
        self.encoder_blocks = nn.ModuleList([
            TransformerEncoderBlock(embed_dim=embed_dim, 
                                    num_heads=num_heads, 
                                    mha_dropout_rate=mha_dropout_rate, 
                                    ff_expansion=ff_expansion, 
                                    ff_dropout_rate=ff_dropout_rate)
            for _ in range(depth)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.squeeze().permute(2, 0, 1)  # (B, C, T) -> (T, B, C
        for encoder_block in self.encoder_blocks:
            x = encoder_block(x)
        x = x.permute(1, 2, 0)      # (T, B, C) -> (B, C, T)
        x = x.unsqueeze(-2)         # (B, C, T) -> (B, C, 1, T)
        return x

class ClassifierHead(nn.Module):
    def __init__(self, in_channels: int, hid_channels: int, num_classes: int, dropout_rate: float = 0.5):
        super(ClassifierHead, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, hid_channels), 
            nn.ReLU(), 
            nn.Dropout(dropout_rate),
            nn.Linear(hid_channels, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

class ExpModel3(nn.Module):
    def __init__(self,
                 in_channels: int = 1,
                 sampling_rate: int = 128,
                 num_electrodes: int = 32,
                 num_T: int = 16,
                 num_S: int = 16,
                 num_classes: int = 2,
                 hid_channels: int = 32,
                 dropout_rate: float = 0.5):
        super(ExpModel3, self).__init__()
        self.mr_tscnn = MRTSCNN(in_channels=in_channels, 
                                sampling_rate=sampling_rate, 
                                num_electrodes=num_electrodes, 
                                num_T=num_T, 
                                num_S=num_S,
                                se_reduction=[num_T//4, num_S//4])
        
        self.afr = expAFR(in_channels=self.mr_tscnn.out_channels, 
                          block=BottleneckResidualSEBlock,
                          num_blocks=1, 
                          deep_scale=2, 
                          se_reduction_base=16)
        
        self.transformer = TransformerEncoder(depth=1,
                                              embed_dim=self.afr.out_channels,
                                              num_heads=2,
                                              mha_dropout_rate=0.5,
                                              ff_expansion=4,
                                              ff_dropout_rate=0.5)

        self.classifier = ClassifierHead(in_channels=self.afr.out_channels, 
                                         hid_channels=hid_channels, 
                                         num_classes=num_classes, 
                                         dropout_rate=dropout_rate)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mr_tscnn(x)
        x = self.afr(x)
        x = self.transformer(x)
        x = self.classifier(x)
        return x






        



