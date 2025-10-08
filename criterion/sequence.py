import numpy as np
from .. import utils

torch = utils.LazyImport("torch")

class Criterion ():
    def __init__ (self, dataset, device):
        self.device = device
        self.loss = torch.nn.NLLLoss(
            ignore_index=0)#, reduction='none')

    def __call__ (self, output, target,
                  weights=None, **kwargs):
        # calculating NLL loss for all tokens
        # output has shape (Batch, Tokens, Embdim)
        # PyTorch NLLloss requires (Batch, d1, d2, .. Tokens)
        # so moving Tokens axis to end
        output_ = output.moveaxis(1,-1)
        loss = self.loss(output_, target)
        loss = self.apply_weights(loss, weights)
        n_tokens = (target!=0).sum()
        loss_norm =  loss.sum()/n_tokens
        # loss_norm is already normalized by the non-pad tokens number
        # BUT the trainer applies own normalization by number of samples
        # multiplying at num samples is the compensation
        return loss_norm * len(output)
        
    def apply_weights (self, loss, weights):
        if weights is None:
            return loss
        
        if len(weights.shape) != 1:
            raise NotImplementedError("Currently supporting only batch weights")
        weights_T = weights.reshape(1,-1).T
        return loss * weights_T
    
    def postproc (self, pred):
        return torch.exp(pred[:,-1])

    def state_dict (self):
        return {}

    def load_state_dict (self, state_dict):
        pass

class CriterionPortable ():
    def __init__ (self):
        pass

    def postproc (self, pred):
        return np.exp(pred[:,-1])

    def load_state_dict (self, state_dict):
        pass

