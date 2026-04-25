import numpy as np
from .. import utils

torch = utils.LazyImport("torch")

def softmax(X, theta = 1.0, axis = None):
    # make X at least 2d
    y = np.atleast_2d(X)
    # find axis
    if axis is None:
        axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)
    # multiply y against the theta parameter, 
    y = y * float(theta)
    # subtract the max for numerical stability
    y = y - np.expand_dims(np.max(y, axis = axis), axis)
    # exponentiate y
    y = np.exp(y)
    # take the sum along the specified axis
    ax_sum = np.expand_dims(np.sum(y, axis = axis), axis)
    # finally: divide elementwise
    p = y / ax_sum
    # flatten if X was 1D
    if len(X.shape) == 1: p = p.flatten()
    return p

class Criterion ():
    def __init__ (self, dataset, device):
        self.device = device
        self.class_weights = utils.dataset_attr(
            dataset,
            'class_weights',
            ignore_missing=True)
        if self.class_weights is not None:
            self.class_weights = torch.tensor(class_weights).to(device)
        self.loss = torch.nn.CrossEntropyLoss(
            weight=self.class_weights,
            reduction='none',
            label_smoothing=0.01)

    def __call__ (self, output, target,
                  weights=None, **kwargs):
        loss = self.loss(output, target)
        if weights is not None:
            loss = loss*weights
        return loss.mean()

    def postproc (self, pred):
        return torch.softmax(pred, dim=1)

    def state_dict (self):
        return {'class_weights' : self.class_weights.cpu().numpy()}

    def load_state_dict (self, state_dict):
        self.class_weights = torch.FloatTensor(
            state_dict['class_weights']).to(self.device)


class CriterionPortable ():
    def __init__ (self):
        pass
    
    def load_state_dict (self, state_dict):
        self.class_weights = state_dict['class_weights']
            
    def postproc (self, pred):
        return softmax(pred, axis=1)
