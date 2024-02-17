from typing import List, Union, Dict

from torcheeg.transforms import LabelTransform

class CategoryRange(LabelTransform):
    r'''
    Categorize the label according to a certain threshold range. \
    Labels within the range are set to the corresponding category, and labels outside the range are set to the last category.
    
    :obj:`CategoryRange` allows simultaneous categorization using the same threshold range for multiple labels. \
    The threshold range can also be defined as a dictionary, where the key is the category name and the value is the threshold range. \
    The last category is the category for labels that do not satisfy any threshold range ('others' by default if dict provided).
    The range is defined as a list of two elements, where the first element is the lower bound and the second element is the upper bound. 

    The threshold range can also be defined as a list of threshold ranges, where the last category is the category for labels that do not satisfy any threshold range.

    .. code-block:: python
    
        transform = ztransforms.CategoryRange(
            threshold_ranges = { 
                'High Stress' : [ [ [7.5, 9] ] ],
                'Low Stress'  : [ [ [5, 7.5] ] ],
            }
        )
        transform(y=[8])['y']
        >>> High Stress
    

    .. code-block:: python

        transform = ztransforms.CategoryRange(
            threshold_ranges = { 
                'High Stress' : [ [ [7.5, 9] ], [ [0, 2.5]   ] ],
                'Low Stress'  : [ [ [5, 7.5] ], [ [2.5, 5.0] ] ],
            }
        )
        transform(y=[8, 2])['y']
        >>> High Stress

    .. code-block:: python

        transform = ztransforms.CategoryRange(
            threshold_ranges = { 
                'High Stress' : [ [ [7.5, 9] ], [ [0, 2.5]   ] ],
                'Low Stress'  : [ [ [5, 7.5] ], [ [2.5, 5.0] ] ],
            }
        )
        transform(y=[3, 2])['y']
        >>> others

    .. code-block:: python

        transform = ztransforms.CategoryRange(
            threshold_ranges = { 
                'High Stress' : [ [ [7.5, 9] ], [ [0, 2.5]   ] ],
                'Low Stress'  : [ [ [5, 7.5] ], [ [2.5, 5.0] ] ],
                'Remain'      : -1
            }
        )
        transform(y=[3, 2])['y']
        >>> Remain

    .. code-block:: python

        transform = ztransforms.CategoryRange(
            threshold_ranges = [
                [ [ [7.5, 9] ], [ [0, 2.5]   ] ],
                [ [ [5, 7.5] ], [ [2.5, 5.0] ] ],
            ]
        )
        transform(y=[8, 2])['y']
        >>> 0

    .. code-block:: python

        transform = ztransforms.CategoryRange(
            threshold_ranges = [
                [ [ [7.5, 9] ], [ [0, 2.5]   ] ],
                [ [ [5, 7.5] ], [ [2.5, 5.0] ] ],
            ]
        )
        transform(y=[3, 2])['y']
        >>> 2

    Args:
        threshold_ranges (list or dict): The threshold range used during categorization. \
                                        The range is defined as a list of two elements, \
                                        where the first element is the lower bound and the second element is the upper bound. \
                                        The last category is the category for labels that do not satisfy any threshold range.

    '''

    def __init__(self, threshold_ranges: Union[List, Dict]):
        super(CategoryRange, self).__init__()
        

        if isinstance(threshold_ranges, dict):
            self.categories = list(threshold_ranges.keys())
            self.remain_category = None
            self.threshold_ranges = []
            
            for cat in self.categories:
                if threshold_ranges[cat] == -1:
                    self.remain_category = cat
                    self.categories.remove(cat)
                    break

                self.threshold_ranges.append(threshold_ranges[cat])


            if self.remain_category:
                self.categories.append(self.remain_category)
            else:
                self.categories.append('others')
        else:
            self.threshold_ranges = threshold_ranges
            self.categories = list(range(len(threshold_ranges)+1))

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
                

