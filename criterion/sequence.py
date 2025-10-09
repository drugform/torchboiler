import numpy as np
from .. import utils

torch = utils.LazyImport("torch")

def calc_accuracy (probs, target):
    match = (probs.argmax(dim=2) == target) | (target == 0)
    n_tokens = (target != 0).sum()
    n_pads = (target == 0).sum()
    n_correct_tokens = match.sum()-n_pads
    n_correct_seqs = (match.sum(dim=1) == match.shape[1]).sum()
    acc_tok = float(n_correct_tokens/n_tokens)
    acc_seq = float(n_correct_seqs/len(target))
    return acc_tok, acc_seq

class Criterion ():
    def __init__ (self, dataset, device):
        self.device = device
        self.loss = torch.nn.NLLLoss(
            ignore_index=0, reduction='none')

    def __call__ (self, output, target,
                  weights=None):
        # calculating NLL loss for all tokens
        # output has shape (Batch, Tokens, Embdim)
        # PyTorch NLLloss requires (Batch, d1, d2, .. Tokens)
        # so moving Tokens axis to end
        output_ = output.moveaxis(1,-1)
        loss = self.loss(output_, target)
        loss = self.apply_weights(loss, weights)
        n_tokens = (target!=0).sum()
        torch_loss = loss.sum()/n_tokens
        acc_tok, acc_seq = calc_accuracy(output, target)
        # __call__ returns batch-averaged loss for backward pass
        # and batch_averaged metrics, including 'Loss'
        # 'Loss' metric is unscaled loss (if scaling is applicable)
        metrics = {'Loss' : float(torch_loss.detach()),
                   'Acc(tok)' : acc_tok,
                   'Acc(seq)' : acc_seq}
        return torch_loss, metrics

        
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

