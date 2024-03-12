from mmengine.evaluator import BaseMetric
import torch
from torch.nn import functional as F
from typing import Optional, Sequence, List
import numpy as np
from itertools import product


class Accuracy(BaseMetric):
    def process(self, data_batch, data_samples):
        score, gt = data_samples
        
        # save the middle result of a batch to `self.results`
        self.results.append({
            'batch_size': len(gt),
            'correct': (score.argmax(dim=1) == gt).sum().cpu(),
        })
    
    def compute_metrics(self, results):
        total_correct = sum(item['correct'] for item in results)
        total_size = sum(item['batch_size'] for item in results)
        # return the dict containing the eval results
        # the key is the name of the metric name
        return dict(accuracy=100 * total_correct / total_size)
    

class AccuracyWithLoss(BaseMetric):
    def process(self, data_batch, data_samples):
        score, gt = data_samples
        # save the middle result of a batch to `self.results`
        self.results.append({
            'eval_batch_size': len(gt),
            'eval_correct': (score.argmax(dim=1) == gt).sum().cpu(),
            'eval_loss': F.cross_entropy(score, gt).cpu(),
        })
    
    def compute_metrics(self, results):
        total_correct = sum(item['eval_correct'] for item in results)
        total_size = sum(item['eval_batch_size'] for item in results)
        total_loss = sum(item['eval_loss'] for item in results)
        # return the dict containing the eval results
        # the key is the name of the metric name
        return dict(mean_eval_accuracy=100 * total_correct / total_size,
                    mean_eval_loss=total_loss / total_size)
    

class ExpAccuracy(BaseMetric):
    def __init__(self, process_name: str):
        super().__init__()
        self.process_name = process_name

    def process(self, data_batch, data_samples):
        score, gt = data_samples
        
        # save the middle result of a batch to `self.results`
        self.results.append({
            f'{self.process_name}_batch_size': len(gt),
            f'{self.process_name}_correct': (score.argmax(dim=1) == gt).sum().cpu(),
        })
    
    def compute_metrics(self, results):
        total_correct = sum(item[f'{self.process_name}_correct'] for item in results)
        total_size = sum(item[f'{self.process_name}_batch_size'] for item in results)
        # return the dict containing the eval results
        # the key is the name of the metric name
        return dict({f'{self.process_name}_accuracy': 100 * total_correct / total_size})
    

    
    


class ConfusionMatrix(BaseMetric):
    default_prefix = 'confusion_matrix'
    def __init__(self,
                num_classes: Optional[int] = None,
                collect_device: str = 'cpu',
                prefix: Optional[str] = None) -> None:
        super().__init__(collect_device, prefix)

        self.num_classes = num_classes

    def process(self, data_batch, data_samples) -> None:

        score, gt = data_samples  
        pred_label = score.argmax(dim=1).flatten()

        self.results.append({
            'pred_label': pred_label,
            'gt_label': gt,
        })

    def compute_metrics(self, results: list) -> dict:
        pred_labels = []
        gt_labels = []
        for result in results:
            pred_labels.append(result['pred_label'])
            gt_labels.append(result['gt_label'])
        confusion_matrix = ConfusionMatrix.calculate(
            torch.cat(pred_labels),
            torch.cat(gt_labels),
            num_classes=self.num_classes)
        return {'result': confusion_matrix}

    @staticmethod
    def calculate(pred, target, num_classes=None) -> dict:
        """Calculate the confusion matrix for single-label task.

        Args:
            pred (torch.Tensor | np.ndarray | Sequence): The prediction
                results. It can be labels (N, ), or scores of every
                class (N, C).
            target (torch.Tensor | np.ndarray | Sequence): The target of
                each prediction with shape (N, ).
            num_classes (Optional, int): The number of classes. If the ``pred``
                is label instead of scores, this argument is required.
                Defaults to None.

        Returns:
            torch.Tensor: The confusion matrix.
        """
        # pred = to_tensor(pred)
        # target_label = to_tensor(target).int()
        target_label = target.int()

        assert pred.size(0) == target_label.size(0), \
            f"The size of pred ({pred.size(0)}) doesn't match "\
            f'the target ({target_label.size(0)}).'
        assert target_label.ndim == 1

        if pred.ndim == 1:
            assert num_classes is not None, \
                'Please specify the `num_classes` if the `pred` is labels ' \
                'intead of scores.'
            pred_label = pred
        else:
            num_classes = num_classes or pred.size(1)
            pred_label = torch.argmax(pred, dim=1).flatten()

        with torch.no_grad():
            indices = num_classes * target_label + pred_label
            matrix = torch.bincount(indices, minlength=num_classes**2)
            matrix = matrix.reshape(num_classes, num_classes)


        return matrix

    @staticmethod
    def plot(confusion_matrix: torch.Tensor,
             include_values: bool = False,
             cmap: str = 'viridis',
             classes: Optional[List[str]] = None,
             colorbar: bool = True,
             show: bool = True,
             normalize: bool = False):
        """Draw a confusion matrix by matplotlib.

        Modified from `Scikit-Learn
        <https://github.com/scikit-learn/scikit-learn/blob/dc580a8ef/sklearn/metrics/_plot/confusion_matrix.py#L81>`_

        Args:
            confusion_matrix (torch.Tensor): The confusion matrix to draw.
            include_values (bool): Whether to draw the values in the figure.
                Defaults to False.
            cmap (str): The color map to use. Defaults to use "viridis".
            classes (list[str], optional): The names of categories.
                Defaults to None, which means to use index number.
            colorbar (bool): Whether to show the colorbar. Defaults to True.
            show (bool): Whether to show the figure immediately.
                Defaults to True.
        """  # noqa: E501
        import matplotlib.pyplot as plt

        if normalize:
            row_sums = confusion_matrix.sum(axis=1)
            confusion_matrix = confusion_matrix / row_sums[:, np.newaxis]

        fig, ax = plt.subplots(figsize=(10, 10))

        num_classes = confusion_matrix.size(0)

        im_ = ax.imshow(confusion_matrix, interpolation='nearest', cmap=cmap)
        text_ = None
        cmap_min, cmap_max = im_.cmap(0), im_.cmap(1.0)

        if include_values:
            text_ = np.empty_like(confusion_matrix, dtype=object)

            # print text with appropriate color depending on background
            thresh = (confusion_matrix.max() + confusion_matrix.min()) / 2.0

            for i, j in product(range(num_classes), range(num_classes)):
                color = cmap_max if confusion_matrix[i,
                                                     j] < thresh else cmap_min

                text_cm = format(confusion_matrix[i, j], '.2g')

                if normalize:
                    text_d = format(confusion_matrix[i, j], '.2f')
                else:
                    text_d = format(confusion_matrix[i, j], 'd')
                
                if len(text_d) < len(text_cm):
                    text_cm = text_d

                text_[i, j] = ax.text(
                    j, i, text_cm, ha='center', va='center', color=color)

        display_labels = classes or np.arange(num_classes)

        if colorbar:
            fig.colorbar(im_, ax=ax)
        ax.set(
            xticks=np.arange(num_classes),
            yticks=np.arange(num_classes),
            xticklabels=display_labels,
            yticklabels=display_labels,
            ylabel='True label',
            xlabel='Predicted label',
        )
        ax.invert_yaxis()
        ax.xaxis.tick_top()

        ax.set_ylim((num_classes - 0.5, -0.5))
        # Automatically rotate the x labels.
        fig.autofmt_xdate(ha='center')

        if show:
            plt.show()
        return fig

