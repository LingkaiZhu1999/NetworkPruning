import torch.nn.utils.prune as prune
from torch.nn.utils.prune import _compute_nparams_toprune, _validate_pruning_amount, _validate_pruning_amount_init, BasePruningMethod
import torch

class Prune_and_Reconnect(BasePruningMethod):
    r"""Reconnect (currently unpruned) units in a tensor by assiging one to the mask of the ones
    with the lowest L1-norm.

    Args:
        amount (int or float): quantity of parameters to prune.
            If ``float``, should be between 0.0 and 1.0 and represent the
            fraction of parameters to reconnect. If ``int``, it represents the
            absolute number of parameters to reconnect.
    """

    PRUNING_TYPE = "unstructured"

    def __init__(self, amount_prune, amount_add):
        # Check range of validity of pruning amount
        _validate_pruning_amount_init(amount_prune)
        _validate_pruning_amount_init(amount_add)
        self.amount_prune = amount_prune
        self.amount_add = amount_add

    def compute_mask(self, t, default_mask):
        # Check that the amount of units to prune is not > than the number of
        # parameters in t
        tensor_size = t.nelement()
        # Compute number of units to prune: amount if int,
        # else amount * tensor_size
        nparams_toprune = _compute_nparams_toprune(self.amount_prune, tensor_size)
        nparams_toadd = _compute_nparams_toprune(self.amount_add, tensor_size)
        # This should raise an error if the number of units to prune is larger
        # than the number of units in the tensor
        _validate_pruning_amount(nparams_toprune, tensor_size)
        _validate_pruning_amount(nparams_toadd, tensor_size)
        mask = default_mask.clone(memory_format=torch.contiguous_format)

        if nparams_toprune != 0 and nparams_toadd != 0:  # k=0 not supported by torch.kthvalue
            # largest=True --> top k; largest=False --> bottom k
            # Prune the smallest k
            topk = torch.topk(torch.abs(t).view(-1), k=nparams_toprune, largest=False)
            # topk will have .indices and .values
            mask_bef_pru = mask.clone(memory_format=torch.contiguous_format)
            mask.view(-1)[topk.indices] = 0

            zero_indices = (mask_bef_pru.view(-1) == 0).nonzero() if torch.count_nonzero(mask_bef_pru.view(-1)==0) >=1 else (mask.view(-1) == 0).nonzero()
            try: 
                del mask_bef_pru
            except:
                None
            zero_indices = zero_indices.view(-1)[torch.randperm(zero_indices.nelement())].view(zero_indices.size())[0:nparams_toadd]
            mask.view(-1)[zero_indices] = 1


        return mask

    @classmethod
    def apply(cls, module, name, amount_prune, amount_add, importance_scores=None):
        r"""Adds the forward pre-hook that enables pruning on the fly and
        the reparametrization of a tensor in terms of the original tensor
        and the pruning mask.

        Args:
            module (nn.Module): module containing the tensor to prune
            name (str): parameter name within ``module`` on which pruning
                will act.
            amount (int or float): quantity of parameters to prune.
                If ``float``, should be between 0.0 and 1.0 and represent the
                fraction of parameters to prune. If ``int``, it represents the
                absolute number of parameters to prune.
            importance_scores (torch.Tensor): tensor of importance scores (of same
                shape as module parameter) used to compute mask for pruning.
                The values in this tensor indicate the importance of the corresponding
                elements in the parameter being pruned.
                If unspecified or None, the module parameter will be used in its place.
        """
        return super(Prune_and_Reconnect, cls).apply(
            module, name, amount_prune=amount_prune, amount_add=amount_add, importance_scores=importance_scores
        )
    
def prune_and_connect(module, name, amount_prune, amount_add, importance_scores=None):
    Prune_and_Reconnect.apply(module, name, amount_add=amount_add, amount_prune=amount_prune, importance_scores=importance_scores)
    return module
class Prune_L1(BasePruningMethod):
    r"""Reconnect (currently unpruned) units in a tensor by assiging one to the mask of the ones
    with the lowest L1-norm.

    Args:
        amount (int or float): quantity of parameters to prune.
            If ``float``, should be between 0.0 and 1.0 and represent the
            fraction of parameters to reconnect. If ``int``, it represents the
            absolute number of parameters to reconnect.
    """

    PRUNING_TYPE = "unstructured"

    def __init__(self, amount):
        # Check range of validity of pruning amount
        _validate_pruning_amount_init(amount)
        self.amount = amount

    def compute_mask(self, t, default_mask):
        # Check that the amount of units to prune is not > than the number of
        # parameters in t
        tensor_size = t.nelement()
        # Compute number of units to prune: amount if int,
        # else amount * tensor_size
        nparams_toprune = _compute_nparams_toprune(self.amount, tensor_size)
        # This should raise an error if the number of units to prune is larger
        # than the number of units in the tensor
        _validate_pruning_amount(nparams_toprune, tensor_size)

        mask = default_mask.clone(memory_format=torch.contiguous_format)

        if nparams_toprune != 0:  # k=0 not supported by torch.kthvalue
            # largest=True --> top k; largest=False --> bottom k
            # Prune the smallest k
            topk = torch.topk(torch.abs(t).view(-1), k=nparams_toprune, largest=False)
            # topk will have .indices and .values
            mask.view(-1)[topk.indices] = 0

        return mask

    @classmethod
    def apply(cls, module, name, amount, importance_scores=None):
        r"""Adds the forward pre-hook that enables pruning on the fly and
        the reparametrization of a tensor in terms of the original tensor
        and the pruning mask.

        Args:
            module (nn.Module): module containing the tensor to prune
            name (str): parameter name within ``module`` on which pruning
                will act.
            amount (int or float): quantity of parameters to prune.
                If ``float``, should be between 0.0 and 1.0 and represent the
                fraction of parameters to prune. If ``int``, it represents the
                absolute number of parameters to prune.
            importance_scores (torch.Tensor): tensor of importance scores (of same
                shape as module parameter) used to compute mask for pruning.
                The values in this tensor indicate the importance of the corresponding
                elements in the parameter being pruned.
                If unspecified or None, the module parameter will be used in its place.
        """
        return super(Prune_and_Reconnect, cls).apply(
            module, name, amount=amount, importance_scores=importance_scores
        )
    


