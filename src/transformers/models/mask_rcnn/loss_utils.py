# Copyright (c) OpenMMLab. All rights reserved.

import warnings

import torch
from torch import nn


def reduce_loss(loss, reduction):
    """Reduce loss as specified.

    Args:
        loss (`torch.Tensor`):
            Elementwise loss tensor.
        reduction (`str`):
            Options are `"none"`, `"mean"` and `"sum"`.

    Return:
        `torch.Tensor`: Reduced loss tensor.
    """
    reduction_enum = nn.functional._Reduction.get_enum(reduction)
    # none: 0, elementwise_mean:1, sum: 2
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.mean()
    elif reduction_enum == 2:
        return loss.sum()


def weight_reduce_loss(loss, weight=None, reduction="mean", avg_factor=None):
    """Apply element-wise weight and reduce loss.

    Args:
        loss (`torch.Tensor`):
            Element-wise loss.
        weight (`torch.Tensor`, *optional*):
            Element-wise weights.
        reduction (`str`, *optional*, defaults to `"mean"`):
            Same as built-in losses of PyTorch.
        avg_factor (`float`, *optional*):
            Average factor when computing the mean of losses.

    Returns:
        `torch.Tensor`: Processed loss values.
    """
    # if weight is specified, apply element-wise weight
    if weight is not None:
        loss = loss * weight

    # if avg_factor is not specified, just reduce the loss
    if avg_factor is None:
        loss = reduce_loss(loss, reduction)
    else:
        # if reduction is mean, then average the loss by avg_factor
        if reduction == "mean":
            # Avoid causing ZeroDivisionError when avg_factor is 0.0,
            # i.e., all labels of an image belong to ignore index.
            eps = torch.finfo(torch.float32).eps
            loss = loss.sum() / (avg_factor + eps)
        # if reduction is 'none', then do nothing, otherwise raise an error
        elif reduction != "none":
            raise ValueError('avg_factor can not be used with reduction="sum"')
    return loss


def accuracy(pred, target, topk=1, thresh=None):
    """Calculate accuracy according to the prediction and target.

    Args:
        pred (`torch.Tensor`):
            The model prediction, shape (N, num_class).
        target (`torch.Tensor`):
            The target of each prediction, shape (N, ).
        topk (`int` or `Tuple[int]`, *optional*, defaults to 1):
            If the predictions in `topk` matches the target, the predictions will be regarded as correct ones.
        thresh (`float`, *optional*):
            If not None, predictions with scores under this threshold are considered incorrect.

    Returns:
        `float` or `Tuple[float]`: If the input `topk` is a single integer,
            the function will return a single float as accuracy. If `topk` is a tuple containing multiple integers, the
            function will return a tuple containing accuracies of each `topk` number.
    """
    if not isinstance(topk, (int, tuple)):
        raise ValueError(f"topk should be int or tuple of int, but got {type(topk)}")
    if isinstance(topk, int):
        topk = (topk,)
        return_single = True
    else:
        return_single = False

    maxk = max(topk)
    if pred.size(0) == 0:
        accu = [pred.new_tensor(0.0) for i in range(len(topk))]
        return accu[0] if return_single else accu
    if not (pred.ndim == 2 and target.ndim == 1):
        raise ValueError(
            f"pred and target should be of shape (N, C) and (N, ), but got {pred.shape} and {target.shape}"
        )
    if pred.size(0) != target.size(0):
        raise ValueError(f"pred and target should be of same size, but got {pred.size(0)} and {target.size(0)}")
    if maxk > pred.size(1):
        raise ValueError(f"maxk {maxk} exceeds pred dimension {pred.size(1)}")
    pred_value, pred_label = pred.topk(maxk, dim=1)
    pred_label = pred_label.t()  # transpose to shape (maxk, N)
    correct = pred_label.eq(target.view(1, -1).expand_as(pred_label))
    if thresh is not None:
        # Only prediction values larger than thresh are counted as correct
        correct = correct & (pred_value > thresh).t()
    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / pred.size(0)))
    return res[0] if return_single else res


def cross_entropy(
    pred,
    label,
    weight=None,
    reduction="mean",
    avg_factor=None,
    class_weight=None,
    ignore_index=-100,
    avg_non_ignore=False,
):
    """Calculate the CrossEntropy loss.

    Args:
        pred (`torch.Tensor`):
            The prediction with shape (N, C), C is the number of classes.
        label (`torch.Tensor`):
            The learning label of the prediction.
        weight (`torch.Tensor`, *optional*):
            Sample-wise loss weight.
        reduction (`str`, *optional*, defaults to `"mean"`):
            The method used to reduce the loss.
        avg_factor (`int`, *optional*):
            Average factor that is used to average the loss.
        class_weight (`List[float]`, *optional*):
            The weight for each class.
        ignore_index (`int`, *optional*, defaults to -100):
            The label index to be ignored.
        avg_non_ignore (`bool`, *optional*, defaults to `False`):
            The flag decides to whether the loss is only averaged over non-ignored targets.

    Returns:
        `torch.Tensor`: The calculated loss
    """
    # The default value of ignore_index is the same as nn.functional.cross_entropy
    ignore_index = -100 if ignore_index is None else ignore_index
    # element-wise losses
    loss = nn.functional.cross_entropy(pred, label, weight=class_weight, reduction="none", ignore_index=ignore_index)

    # average loss over non-ignored elements
    # pytorch's official cross_entropy average loss over non-ignored elements
    # refer to https://github.com/pytorch/pytorch/blob/56b43f4fec1f76953f15a627694d4bba34588969/torch/nn/functional.py#L2660  # noqa
    if (avg_factor is None) and avg_non_ignore and reduction == "mean":
        avg_factor = label.numel() - (label == ignore_index).sum().item()

    # apply weights and do the reduction
    if weight is not None:
        weight = weight.float()
    loss = weight_reduce_loss(loss, weight=weight, reduction=reduction, avg_factor=avg_factor)

    return loss


def _expand_onehot_labels(labels, label_weights, label_channels, ignore_index):
    """Expand onehot labels to match the size of prediction."""
    bin_labels = labels.new_full((labels.size(0), label_channels), 0)
    valid_mask = (labels >= 0) & (labels != ignore_index)
    inds = torch.nonzero(valid_mask & (labels < label_channels), as_tuple=False)

    if inds.numel() > 0:
        bin_labels[inds, labels[inds]] = 1

    valid_mask = valid_mask.view(-1, 1).expand(labels.size(0), label_channels).float()
    if label_weights is None:
        bin_label_weights = valid_mask
    else:
        bin_label_weights = label_weights.view(-1, 1).repeat(1, label_channels)
        bin_label_weights *= valid_mask

    return bin_labels, bin_label_weights, valid_mask


def binary_cross_entropy(
    pred,
    label,
    weight=None,
    reduction="mean",
    avg_factor=None,
    class_weight=None,
    ignore_index=-100,
    avg_non_ignore=False,
):
    """Calculate the binary CrossEntropy loss.

    Args:
        pred (`torch.Tensor`):
            The prediction with shape (N, 1) or (N, ). When the shape of pred is (N, 1), label will be expanded to
            one-hot format, and when the shape of pred is (N, ), label will not be expanded to one-hot format.
        label (`torch.Tensor`):
            The learning label of the prediction, with shape (N, ).
        weight (`torch.Tensor`, *optional*):
            Sample-wise loss weight.
        reduction (`str`, *optional*, defaults to `"mean"`):
            The method used to reduce the loss. Options are `"none"`, `"mean"` and `"sum"`.
        avg_factor (`int`, *optional*):
            Average factor that is used to average the loss. Defaults to None.
        class_weight (`List[float]`, *optional*):
            The weight for each class.
        ignore_index (`int`, *optional*, defaults to -100):
            The label index to be ignored.
        avg_non_ignore (`bool`, *optional*, defaults to `False`):
            The flag decides to whether the loss is only averaged over non-ignored targets.

    Returns:
        `torch.Tensor`: The calculated loss.
    """
    # The default value of ignore_index is the same as nn.functional.cross_entropy
    ignore_index = -100 if ignore_index is None else ignore_index

    if pred.dim() != label.dim():
        label, weight, valid_mask = _expand_onehot_labels(label, weight, pred.size(-1), ignore_index)
    else:
        # should mask out the ignored elements
        valid_mask = ((label >= 0) & (label != ignore_index)).float()
        if weight is not None:
            # The inplace writing method will have a mismatched broadcast
            # shape error if the weight and valid_mask dimensions
            # are inconsistent such as (B,N,1) and (B,N,C).
            weight = weight * valid_mask
        else:
            weight = valid_mask

    # average loss over non-ignored elements
    if (avg_factor is None) and avg_non_ignore and reduction == "mean":
        avg_factor = valid_mask.sum().item()

    # weighted element-wise losses
    weight = weight.float()
    loss = nn.functional.binary_cross_entropy_with_logits(
        pred, label.float(), pos_weight=class_weight, reduction="none"
    )
    # do the reduction for the weighted loss
    loss = weight_reduce_loss(loss, weight, reduction=reduction, avg_factor=avg_factor)

    return loss


def mask_cross_entropy(
    pred, target, label, reduction="mean", avg_factor=None, class_weight=None, ignore_index=None, **kwargs
):
    """Calculate the CrossEntropy loss for masks.

    Args:
        pred (`torch.Tensor`):
            The prediction with shape (N, C, *), C is the number of classes. The trailing * indicates arbitrary shape.
        target (`torch.Tensor`):
            The learning label of the prediction.
        label (`torch.Tensor`):
            Indicates the class label of the mask corresponding object. This will be used to select the mask in the of
            the class which the object belongs to when the mask prediction if not class-agnostic.
        reduction (`str`, *optional*, defaults to `"mean"`):
            The method used to reduce the loss. Options are `"none"`, `"mean"` and `"sum"`.
        avg_factor (`int`, *optional*):
            Average factor that is used to average the loss.
        class_weight (`List[float]`, *optional*):
            The weight for each class.
        ignore_index (`int`, *optional*):
            Placeholder, to be consistent with other loss.

    Returns:
        `torch.Tensor`: The calculated loss
    """
    if ignore_index is not None:
        raise ValueError("BCE loss does not support ignore_index")
    # TODO: handle these two reserved arguments
    if reduction != "mean":
        raise ValueError("BCE loss only supports reduction == 'mean'")
    if avg_factor is not None:
        raise ValueError("BCE loss does not support avg_factor")
    num_rois = pred.size()[0]
    inds = torch.arange(0, num_rois, dtype=torch.long, device=pred.device)
    pred_slice = pred[inds, label].squeeze(1)
    return nn.functional.binary_cross_entropy_with_logits(pred_slice, target, weight=class_weight, reduction="mean")[
        None
    ]


class CrossEntropyLoss(nn.Module):
    def __init__(
        self,
        use_sigmoid=False,
        use_mask=False,
        reduction="mean",
        class_weight=None,
        ignore_index=None,
        loss_weight=1.0,
        avg_non_ignore=False,
    ):
        """
        Args:
            use_sigmoid (`bool`, *optional*, defaults to `False`):
                Whether the prediction uses sigmoid instead of softmax.
            use_mask (`bool`, *optional*, defaults to `False`):
                Whether to use mask cross entropy loss.
            reduction (`str`, *optional*, defaults to `"mean"`):
                Loss reduction to use. Options are `"none"`, `"mean"` and `"sum"`.
            class_weight (`List[float]`, *optional*):
                Optional weight of each class.
            ignore_index (`int`, *optional*):
                Optional label index to be ignored.
            loss_weight (`float`, *optional*, defaults to 1.0):
                Weight of the loss.
            avg_non_ignore (`bool`, *optional*, defaults to `False`):
                The flag decides to whether the loss is only averaged over non-ignored targets.
        """
        super(CrossEntropyLoss, self).__init__()
        if not ((use_sigmoid is False) or (use_mask is False)):
            raise ValueError("Only one of `use_sigmoid` and `use_mask` can be True.")
        self.use_sigmoid = use_sigmoid
        self.use_mask = use_mask
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.class_weight = class_weight
        self.ignore_index = ignore_index
        self.avg_non_ignore = avg_non_ignore
        if (ignore_index is not None) and not self.avg_non_ignore and self.reduction == "mean":
            warnings.warn(
                "Default ``avg_non_ignore`` is False, if you would like to "
                "ignore the certain label and average loss over non-ignore "
                "labels, which is the same with PyTorch official "
                "cross_entropy, set ``avg_non_ignore=True``."
            )

        if self.use_sigmoid:
            self.cls_criterion = binary_cross_entropy
        elif self.use_mask:
            self.cls_criterion = mask_cross_entropy
        else:
            self.cls_criterion = cross_entropy

    def forward(
        self, cls_score, label, weight=None, avg_factor=None, reduction_override=None, ignore_index=None, **kwargs
    ):
        """
        Args:
            cls_score (`torch.Tensor`):
                The prediction.
            label (`torch.Tensor`):
                The learning label of the prediction.
            weight (`torch.Tensor`, *optional*):
                Sample-wise loss weight.
            avg_factor (int, *optional*)
                Average factor that is used to average the loss.
            reduction_override (`str`, *optional*):
                The method used to reduce the loss. Options are `"none"`, `"mean"` and `"sum"`.
            ignore_index (`int`, *optional*):
                The label index to be ignored. If not None, it will override the default value.

        Returns:
            `torch.Tensor`: The calculated loss.
        """
        if reduction_override not in (None, "none", "mean", "sum"):
            raise ValueError(f"invalid reduction_override, got {reduction_override}")
        reduction = reduction_override if reduction_override else self.reduction
        if ignore_index is None:
            ignore_index = self.ignore_index

        if self.class_weight is not None:
            class_weight = cls_score.new_tensor(self.class_weight, device=cls_score.device)
        else:
            class_weight = None
        loss_cls = self.loss_weight * self.cls_criterion(
            cls_score,
            label,
            weight,
            class_weight=class_weight,
            reduction=reduction,
            avg_factor=avg_factor,
            ignore_index=ignore_index,
            avg_non_ignore=self.avg_non_ignore,
            **kwargs,
        )
        return loss_cls
