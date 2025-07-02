import sys
import os
import io
import numpy as np
import base64
import zlib
from importlib import import_module

class LazyImport () :
    'thin shell class to wrap modules.  load real module on first access and pass thru'
    def __init__ (self, modname):
        self._modname = modname
        self._mod = None
   
    def __getattr__ (self, attr):
        'import module on first attribute access'
        try :
            return getattr(self._mod, attr)
        except Exception as e :
            if self._mod is None :
                # module is unset, load it
                self._mod = import_module(self._modname)
            else :
                # module is set, got different exception from getattr(). reraise it
                raise e
        # retry getattr if module was just loaded for first time
        # call this outside exception handler in case it raises new exception
        return getattr(self._mod, attr)

torch = LazyImport("torch")
tqdm = LazyImport("tqdm")
    
def fprint (*args):
    print(*args)
    sys.stdout.flush()


class Progress ():
    def __init__ (self, iterator, verbose):
        self.verbose = verbose
        self.iterator = tqdm.tqdm(iterator) if verbose else iterator

    def __iter__ (self):
        return iter(self.iterator)

    def update (self, value):
        if self.verbose:
            if type(value) is float:
                value = round(value, 3)
            self.iterator.set_description(str(value))

    def manual (self, step):
        if self.verbose:
            self.iterator.update(step)
            
    def finish (self):
        if self.verbose:
            self.manual( len(self.iterator) - self.iterator.n )

def set_seed (seed):
    np.random.seed(seed)
    try:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except: pass

def dataset_attr (ds, name, ignore_missing=False):
    try: return ds.__getattribute__(name)
    except:
        try: return ds.dataset.__getattribute__(name)
        except:
            if ignore_missing: return None
            else: raise Exception(f"No attribute {name} in dataset")

def get_regression_flags (Y):
    regression = []
    Y = np.nan_to_num(Y.copy(), 0)
    for i in range(Y.shape[1]):
        regression.append(
            len((Y[:,i][Y[:,i].nonzero()]-1).nonzero()[0]) > 0)
    return regression 

def input2device (item, device):
    # currently torch only
    if type(item) == torch.Tensor:
        bs = len(item)
        return item.to(device), bs
    elif type(item) in (list, tuple):
        bs = len(item[0])
        return [elem.to(device) for elem in item], bs
    elif type(item) == dict:
        bs = len(next(iter(item.values())))
        return {k:v.to(device) for k,v in item.items()}, bs
    else:
        raise Exception(f'Unsupported batch item type: {type(item)}')

def parse_batch (batch, device):
    # currently torch only
    inpt,bs = input2device(batch[0], device)
    tgt,_ = input2device(batch[-1], device)
 
    if len(batch) == 2:
        w = torch.ones(bs).to(device)
    elif len(batch) == 3:
        w = batch[1].to(device)
    else:
        raise Exception('batch must have 2 (inpt, tgt) or 3 (inpt, w, tgt) elems')
    return inpt, w, tgt

def forward (net, inpt):
    # currently torch only
    if type(inpt) in [torch.Tensor,
                      np.ndarray]:
        return net(inpt)
    elif type(inpt) in (list, tuple):
        return net(*inpt)
    elif type(inpt) == dict:
        return net(**inpt)
    else:
        raise Exception(f'Unsupported network input type: {type(inpt)}')
