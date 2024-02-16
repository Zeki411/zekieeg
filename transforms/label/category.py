from typing import List, Union, Dict

from torcheeg.transforms import LabelTransform

class CategoryRange(LabelTransform):
    r'''
    Args:
        threshold_ranges: list of threshold range for each attribute of sample original label to categorize
    '''

    def __init__(self, threshold_ranges: Union[List, Dict]):
        super(CategoryRange, self).__init__()
        

        if isinstance(threshold_ranges, dict):
            self.categories = list(threshold_ranges.keys())
            self.threshold_ranges = [threshold_ranges[cat] for cat in self.categories]
            self.categories.append('others')
        else:
            self.threshold_ranges = threshold_ranges
            self.categories = list(range(len(threshold_ranges)+1))

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
                for rg in att:
                    if rg[0] < y[att_idx] <= rg[1]:
                        satistfied = True
                        break
                    else:
                        satistfied = False
            if satistfied:
                return self.categories[int(lvl_idx)]
        
        return self.categories[int(len(self.threshold_ranges))] # the last one is which is not satisfied by any range
                

