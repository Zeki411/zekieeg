from mmengine.model import BaseModel
import torch.nn.functional as F
from .tsception import TSCeption

class MMTSCeption(BaseModel):
    def __init__(self,
                 num_classes=2,
                 num_electrodes=32,
                 sampling_rate=128,
                 num_T=15,
                 num_S=15,
                 hid_channels=32,
                 dropout=0.5):
        super(MMTSCeption, self).__init__()

        self.tsception = TSCeption(
            num_classes=num_classes,
            num_electrodes=num_electrodes,
            num_T=num_T,
            num_S=num_S,
            in_channels=1,
            hid_channels=hid_channels,
            sampling_rate=sampling_rate,
            dropout=dropout
        )
        
    def forward(self, eeg_epochs, labels, mode):
        outputs = self.tsception(eeg_epochs)
        if mode == 'loss':
            return {'loss': F.cross_entropy(outputs, labels)}
        elif mode == 'predict':
            return outputs, labels



