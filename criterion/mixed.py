import os
import numpy as np

from ..utils import fprint
from .. import utils

torch = utils.LazyImport("torch")
tqdm = utils.LazyImport("tqdm")
    
def multiclass_loss (output, target, weights=None):
    nans = torch.isnan(target)
    target = target.clone()
    output = output.clone()
    target[nans] = 0
    output[nans] = 0

    res = binary_cross_entropy_with_logits(output, target, reduction='none')
    res[nans] = 0
    if weights is not None:
        res *= weights

    res = res.sum()/len(target)
    return res
    
def mse_loss (output, target, weights=None):
    target = target.clone()
    target[torch.isnan(target)] = output[torch.isnan(target)]
    if weights is not None:
        target = target * weights
        output = output * weights
    return torch.nn.functional.mse_loss(output, target)

def mixed_loss (output, target, regression, weights=None):
        
    losses = []
    
    outdim = len(target[0])
    batch_size = len(target)

    if weights is None:
        weights = torch.ones(batch_size,
                             dtype=torch.float32).to(output.device)
    
    assert output.shape == target.shape, 'Output and target shape mismatch'
    assert len(regression) == outdim, 'Regression flags shape mismatch'
    assert weights.shape[0] == batch_size, 'Weights shape and batch size mismatch'
    if len(weights.shape) > 1:
        assert weights.shape[1] == outdim, 'Weights shape and outdim mismatch'
        batch_only_weight = False
    else:
        batch_only_weight = True
    
    for i in range(len(regression)):
        if batch_only_weight:
            w = weights
        else:
            w = weights[:,i]
        
        if regression[i] == 1:
            losses.append(mse_loss(output[:,i],
                                   target[:,i],
                                   w))
        else:
            losses.append(multiclass_loss(output[:,i],
                                          target[:,i],
                                          w))
    return torch.mean(torch.stack(losses))

class Criterion ():
    def __init__ (self, dataset, device):
        self.device = device
        try:
            self.regression_flags = utils.dataset_attr(dataset, 'regression_flags')
        except:
            fprint("Mixed criterion: Guessing regression flags from dataset...")
            Y = []
            ids = np.arange(len(dataset))
            np.random.shuffle(ids)
            for i in tqdm.tqdm(ids[:1000]):
                item = dataset[i]
                Y.append([float(e) for e in item[-1]])

            self.regression_flags = utils.get_regression_flags(np.array(Y))
            fprint(self.regression_flags)

        self.regression_flags = np.array(self.regression_flags)

        try:
            self.meanY, self.stdY = utils.dataset_attr(dataset, 'scale')
        except:
            if self.regression_flags.max() == 0:
                fprint("Mixed criterion: Classification only dataset, no scaling required...")
                self.meanY = np.zeros(len(self.regression_flags))
                self.stdY = np.ones(len(self.regression_flags))
            else:
                fprint("Mixed criterion: Guessing scaling constants from dataset...")
                Y = []
                ids = np.arange(len(dataset))
                rng = np.random.default_rng(42)
                rng.shuffle(ids)
                for i in tqdm.tqdm(ids[:1000]):
                    item = dataset[i]
                    Y.append([float(e) for e in item[-1]])

                self.fit_scale(Y)
                fprint(f"Mixed criterion: Got constants: {self.meanY}, {self.stdY}")
            
        self.meanY = torch.FloatTensor(self.meanY).to(device)
        self.stdY = torch.FloatTensor(self.stdY).to(device)
        
    def __call__ (self, output, target, weights=None):
        scaled_target = self.scale(target)
        unscaled_output = self.unscale(output)
        torch_loss = mixed_loss(output, scaled_target,
                                self.regression_flags,
                                weights)
        
        unscaled_loss = mixed_loss(unscaled_output, target,
                                   self.regression_flags,
                                   weights).detach()
        metrics = {'Loss' : float(torch_loss.detach()),
                   'Unscaled' : float(unscaled_loss.detach())}
        return torch_loss, metrics

    def state_dict (self):
        return {'regression_flags' : self.regression_flags,
                'meanY' : self.meanY.cpu().detach().numpy(),
                'stdY' : self.stdY.cpu().detach().numpy()}

    def load_state_dict (self, state_dict):
        self.regression_flags = state_dict['regression_flags']
        self.meanY = state_dict['meanY']
        self.stdY = state_dict['stdY']
        self.meanY = torch.FloatTensor(self.meanY).to(self.device)
        self.stdY = torch.FloatTensor(self.stdY).to(self.device)
        
    def postproc (self, pred):
        for i,r in enumerate(self.regression_flags):
            if r==0:
                pred[:,i] = torch.sigmoid(pred[:,i])
        
        pred = self.unscale(pred)
        return pred

    def fit_scale (self, Y):
        self.meanY = np.nanmean(Y, axis=0) * self.regression_flags
        Y = Y-self.meanY
        self.stdY = np.nanstd(Y, axis=0) * self.regression_flags + (1-self.regression_flags)
        Y = Y/self.stdY
        return Y

    def scale (self, Y):
        Y = Y-self.meanY
        Y = Y/self.stdY
        return Y

    def unscale (self, Y):
        Y = Y*self.stdY
        Y = Y+self.meanY
        return Y

def sigmoid(x):
    return np.where(x >= 0, 
                    1 / (1 + np.exp(-x)), 
                    np.exp(x) / (1 + np.exp(x)))
    
class CriterionPortable ():
    def __init__ (self):
        pass
    
    def load_state_dict (self, state_dict):
        self.regression_flags = state_dict['regression_flags']
        self.meanY = state_dict['meanY']
        self.stdY = state_dict['stdY']

    def unscale (self, Y):
        Y = Y*self.stdY
        Y = Y+self.meanY
        return Y
            
    def postproc (self, pred):
        for i,r in enumerate(self.regression_flags):
            if r==0:
                pred[:,i] = sigmoid(pred[:,i])
        
        pred = self.unscale(pred)
        return pred
