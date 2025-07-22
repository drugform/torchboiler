import numpy as np
from ..utils import LazyImport 

torch = LazyImport('torch')

class Collator ():
    def __init__ (self, allow_padding=False):
        self.train = True
        self.collate_to = 'numpy'
        self.allow_padding = allow_padding
        # TODO: get rid of train arg:
        # make it work with list(list(array)) and list(dict(array))
        # OR:
        # move collation into parse_batch => no need for train mode
        # here check if loosing any dataloader functionality
        #
        # TODO: make it work with pure numbers and list of pure numbers
        # at the place of array
    """
    def train (self):
        self.train = True
        return self

    def eval (self):
        self.train = False
        return self

    def torch (self):
        self.collate_to = 'torch'
        return self

    def numpy (self):
        self.collate_to = 'numpy'
        return self
    """
    def get_dtype (self, array):
        # efficiency nightmare, requires rework
        name = type(array).__name__
        if name == 'Tensor':
            if self.collate_to == 'torch':
                return array.dtype
            else: # torch -> numpy 
                return array.numpy().dtype
        if name == 'ndarray':
            if self.collate_to == 'numpy':
                return array.dtype
            else: # numpy -> torch 
                return torch.from_numpy(array).dtype

    def stack_fn (self, samples):
        if self.collate_to == 'torch':
            zeros = torch.zeros
            tensor = torch.tensor
            stack = torch.stack
        elif self.collate_to == 'numpy':
            zeros = np.zeros
            tensor = np.array
            stack = np.stack
        else:
            raise Exception(f"collate_to must be `numpy` or `torch`; got {self.collate_to}")

        if not self.allow_padding:
            return stack(samples)


        if len(samples[0].shape) == 0:
            return stack(samples)
        
        lens = [s.shape[0] for s in samples]
        if len(np.unique(lens)) == 1: # no padding required
            return stack(samples)
        
        maxlen = max(lens)
        rest_dims = samples[0].shape[1:]
        dims = [len(samples), maxlen] + list(rest_dims)
            
        packed = zeros(dims, dtype=self.get_dtype(samples[0]))
        for i,s in enumerate(samples):
            s_len = s.shape[0]
            packed[i, :s_len] = tensor(s)

        return packed
        
    def collate_item (self, samples):
        name = type(samples[0]).__name__        
        if name in ['ndarray', 'Tensor']:
            return self.stack_fn(samples)
        elif name in ['list', 'tuple']:
            return [self.stack_fn([s[i]
                                   for s in samples])
                    for i in range(len(samples[0]))]
        elif name in ['dict']:
            return {k:self.stack_fn([s[k]
                                     for s in samples])
                    for k in samples[0].keys()}
        else:
            raise Exception(f'Unknown sample format: {name}')

    def collate_train (self, samples):
        n_items = len(samples[0])
        if n_items not in [2,3]:
            raise Exception(f'Train dataset samples should be lists of len==2 ([input, output]) or len==3 ([input, weight, output]). Got len=={n_items}')
        inputs = [s[0] for s in samples]
        targets = [s[-1] for s in samples]
        if len(samples[0]) == 2:
            return (self.collate_item(inputs),
                    self.collate_item(targets))
        else:
            weights = np.array([s[1] for s in samples])
            if self.collate_to == 'torch':
                weights = torch.from_numpy(weights)
            
            return (self.collate_item(inputs),
                    weights,
                    self.collate_item(targets))
        
    def __call__ (self, samples):
        if self.train:
            return self.collate_train(samples)
        else:
            return self.collate_item(samples)

        

