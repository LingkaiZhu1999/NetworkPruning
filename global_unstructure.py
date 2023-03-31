import torch.nn.utils.prune as prune
from torch.nn.utils.prune import _compute_nparams_toprune, _validate_pruning_amount, _validate_pruning_amount_init, BasePruningMethod, PruningContainer
import torch
import numbers
from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Tuple

class CustomFromMask(BasePruningMethod):

    PRUNING_TYPE = "global"

    def __init__(self, mask):
        self.mask = mask

    def compute_mask(self, t, default_mask):
        assert default_mask.shape == self.mask.shape
        mask = default_mask * self.mask.to(dtype=default_mask.dtype)
        return mask

    @classmethod
    def apply(cls, module, name, mask):
        r"""Adds the forward pre-hook that enables pruning on the fly and
        the reparametrization of a tensor in terms of the original tensor
        and the pruning mask.

        Args:
            module (nn.Module): module containing the tensor to prune
            name (str): parameter name within ``module`` on which pruning
                will act.
        """
        return super(CustomFromMask, cls).apply(module, name, mask=mask)
    
def global_unstructured(parameters, pruning_method, importance_scores=None, **kwargs):
    r"""
    Globally prunes tensors corresponding to all parameters in ``parameters``
    by applying the specified ``pruning_method``.
    Modifies modules in place by:

    1) adding a named buffer called ``name+'_mask'`` corresponding to the
       binary mask applied to the parameter ``name`` by the pruning method.
    2) replacing the parameter ``name`` by its pruned version, while the
       original (unpruned) parameter is stored in a new parameter named
       ``name+'_orig'``.

    Args:
        parameters (Iterable of (module, name) tuples): parameters of
            the model to prune in a global fashion, i.e. by aggregating all
            weights prior to deciding which ones to prune. module must be of
            type :class:`nn.Module`, and name must be a string.
        pruning_method (function): a valid pruning function from this module,
            or a custom one implemented by the user that satisfies the
            implementation guidelines and has ``PRUNING_TYPE='unstructured'``.
        importance_scores (dict): a dictionary mapping (module, name) tuples to
            the corresponding parameter's importance scores tensor. The tensor
            should be the same shape as the parameter, and is used for computing
            mask for pruning.
            If unspecified or None, the parameter will be used in place of its
            importance scores.
        kwargs: other keyword arguments such as:
            amount (int or float): quantity of parameters to prune across the
            specified parameters.
            If ``float``, should be between 0.0 and 1.0 and represent the
            fraction of parameters to prune. If ``int``, it represents the
            absolute number of parameters to prune.

    Raises:
        TypeError: if ``PRUNING_TYPE != 'unstructured'``

    Note:
        Since global structured pruning doesn't make much sense unless the
        norm is normalized by the size of the parameter, we now limit the
        scope of global pruning to unstructured methods.

    Examples:
        >>> from torch.nn.utils import prune
        >>> from collections import OrderedDict
        >>> net = nn.Sequential(OrderedDict([
        ...     ('first', nn.Linear(10, 4)),
        ...     ('second', nn.Linear(4, 1)),
        ... ]))
        >>> parameters_to_prune = (
        ...     (net.first, 'weight'),
        ...     (net.second, 'weight'),
        ... )
        >>> prune.global_unstructured(
        ...     parameters_to_prune,
        ...     pruning_method=prune.L1Unstructured,
        ...     amount=10,
        ... )
        >>> print(sum(torch.nn.utils.parameters_to_vector(net.buffers()) == 0))
        tensor(10)

    """
    # ensure parameters is a list or generator of tuples
    if not isinstance(parameters, Iterable):
        raise TypeError("global_unstructured(): parameters is not an Iterable")

    importance_scores = importance_scores if importance_scores is not None else {}
    if not isinstance(importance_scores, dict):
        raise TypeError("global_unstructured(): importance_scores must be of type dict")

    # flatten importance scores to consider them all at once in global pruning
    relevant_importance_scores = torch.nn.utils.parameters_to_vector(
        [
            importance_scores.get((module, name), getattr(module, name))
            for (module, name) in parameters
        ]
    )
    # similarly, flatten the masks (if they exist), or use a flattened vector
    # of 1s of the same dimensions as t
    default_mask = torch.nn.utils.parameters_to_vector(
        [
            getattr(module, name + "_mask", torch.ones_like(getattr(module, name)))
            for (module, name) in parameters
        ]
    )

    # use the canonical pruning methods to compute the new mask, even if the
    # parameter is now a flattened out version of `parameters`
    container = PruningContainer()
    container._tensor_name = "temp"  # to make it match that of `method`
    method = pruning_method(**kwargs)
    method._tensor_name = "temp"  # to make it match that of `container`
    if method.PRUNING_TYPE != "unstructured":
        raise TypeError(
            'Only "unstructured" PRUNING_TYPE supported for '
            "the `pruning_method`. Found method {} of type {}".format(
                pruning_method, method.PRUNING_TYPE
            )
        )

    container.add_pruning_method(method)

    # use the `compute_mask` method from `PruningContainer` to combine the
    # mask computed by the new method with the pre-existing mask
    final_mask = container.compute_mask(relevant_importance_scores, default_mask)

    # Pointer for slicing the mask to match the shape of each parameter
    pointer = 0
    for module, name in parameters:

        param = getattr(module, name)
        # The length of the parameter
        num_param = param.numel()
        # Slice the mask, reshape it
        param_mask = final_mask[pointer : pointer + num_param].view_as(param)
        # Assign the correct pre-computed mask to each parameter and add it
        # to the forward_pre_hooks like any other pruning method
        custom_from_mask(module, name, mask=param_mask)
        for k in list(module._forward_pre_hooks):
            hook = module._forward_pre_hooks[k]
            if isinstance(hook, PruningContainer):
                if isinstance(hook[-1], CustomFromMask):
                    hook[-1].mask = None
            elif isinstance(hook, CustomFromMask):
                hook.mask = None

        # Increment the pointer to continue slicing the final_mask
        pointer += num_param


def custom_from_mask(module, name, mask):
    r"""Prunes tensor corresponding to parameter called ``name`` in ``module``
    by applying the pre-computed mask in ``mask``.
    Modifies module in place (and also return the modified module)
    by:

    1) adding a named buffer called ``name+'_mask'`` corresponding to the
       binary mask applied to the parameter ``name`` by the pruning method.
    2) replacing the parameter ``name`` by its pruned version, while the
       original (unpruned) parameter is stored in a new parameter named
       ``name+'_orig'``.

    Args:
        module (nn.Module): module containing the tensor to prune
        name (str): parameter name within ``module`` on which pruning
            will act.
        mask (Tensor): binary mask to be applied to the parameter.

    Returns:
        module (nn.Module): modified (i.e. pruned) version of the input module

    Examples:
        >>> from torch.nn.utils import prune
        >>> m = prune.custom_from_mask(
        ...     nn.Linear(5, 3), name='bias', mask=torch.tensor([0, 1, 0])
        ... )
        >>> print(m.bias_mask)
        tensor([0., 1., 0.])

    """
    CustomFromMask.apply(module, name, mask)
    return module
        