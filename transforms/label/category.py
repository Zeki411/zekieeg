from typing import List, Union, Dict

from torcheeg.transforms import LabelTransform

class CategoryRange(LabelTransform):
    r'''
    Args:
        threshold_ranges: list of list of float
            threshold range for each attribute of sample original label to binarize
            (from high to low)
    '''

    def __init__(self, threshold_ranges: Union[List]):
        super(CategoryRange, self).__init__()
        self.threshold_ranges = threshold_ranges

        # self.num_binaries = len(threshold_ranges) #TODO

    def __call__(self, *args,  y: Union[int, float, List],
                 **kwargs) -> Union[int, List]:
        r'''
        Args:
            label (int, float, or list): The input label or list of labels.
            
        Returns:
            int, float, or list: The output label or list of labels after binarization.
        '''
        return super().__call__(*args, y=y, **kwargs)
    
    def apply(self, y: Union[int, float, List], **kwargs) -> List:
        
        for lvl_idx, lvl in enumerate(self.threshold_ranges):
            satistfied = False
            for att_idx, att in enumerate(lvl):
                print(y)

                if att[0] < y[att_idx] <= att[1]:
                    satistfied = True
                else:
                    satistfied = False
            if satistfied:
                return int(lvl_idx)
        
        return int(len(self.threshold_ranges)) # the last one is which is not satisfied by any range
                

