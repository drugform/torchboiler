import sys
import os
import io
import numpy as np
import base64
import zlib
import decimal
import sigfig
from importlib import import_module

class LazyImport () :
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
mpt = LazyImport("torch.multiprocessing")
mp = LazyImport("multiprocessing")
tqdm = LazyImport("tqdm")
    
def fprint (*args):
    print(*args)
    sys.stdout.flush()

def round (num, prec, significant=False, with_zeros=False):
    # обходим эпическое округление в питоне
    if with_zeros:
        num = str(num)
    if significant:
        return sigfig.round(num, sigfigs=prec)
    else:
        return sigfig.round(num, decimals=prec, warn=False)

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

# IDEA: implement recursion logic from default collator
def convert_sample (sample, to):
    name = type(sample).__name__
    if name in ['Tensor', 'ndarray']:
        return convert_item(sample, to)
    elif name in ['list', 'tuple']:
        return tuple([convert_item(item, to)
                      for item in sample])
    elif name in ['dict']:
        return {k:convert_item(v, to)
                for k,v in item.items()}
    else:
        Exception(f'Unsupported batch item type: {name}')

def convert_item (item, to):
    name = type(item).__name__
    if to == 'numpy':
        if name == 'Tensor':
            return item.detach().cpu().numpy()
        elif name == 'ndarray':
            return item
        else:
            raise Exception(f'Item type must be torch.Tensor or np.ndarray. Got {name}')
    elif to == 'cpu':
        if name == 'Tensor':
            return item.detach().cpu()
        elif name == 'ndarray':
            return torch.from_numpy(item)
        else:
            raise Exception(f'Item type must be torch.Tensor or np.ndarray. Got {name}')
    elif to.startswith('cuda'):
        if name == 'Tensor':
            return item.to(to)
        elif name == 'ndarray':
            return torch.from_numpy(item).to(to)
        else:
            raise Exception(f'Item type must be torch.Tensor or np.ndarray. Got {name}')
    else:
        raise Exception(f'Unknown conversion target: {to}. Expecting `numpy`,  `cpu` (torch), `cuda`, `cuda:X`')


def forward (net, inpt):
    name = type(inpt).__name__
    if name in ['Tensor', 'ndarray']:
        return net(inpt)
    elif name in ['list', 'tuple']:
        return net(*inpt)
    elif name in ['dict']:
        return net(**inpt)
    else:
        Exception(f'Unsupported network input type: {name}')

class DataLoader ():
    def __init__ (self, dataset, batch_size, collate_fn):
        self.dataset = dataset
        self.batch_size = batch_size
        self.collate_fn = collate_fn
        
        self.pos = 0
        self.limit = len(dataset)

    def __iter__ (self):
        return self
        
    def __next__(self):
        to = min(self.pos + self.batch_size,
                 self.limit)
        if self.pos >= self.limit:
            raise StopIteration
        
        samples = [self.dataset[idx] for idx
                   in range(self.pos, to)]
        self.pos += len(samples)
        return self.collate_fn(samples)


def get_sample_shapes (sample, prefix):
    name = type(sample).__name__
    if name in ['Tensor', 'ndarray']:
        return {f'{prefix}_0' : tuple(sample.shape)}
    elif name in ['list', 'tuple']:
        return {f'{prefix}_{i}' : tuple(sample[i].shape)
                for i in range(len(sample))}
    elif name in ['dict']:
        return {f'{prefix}_{k}' : tuple(v.shape)
                for k,v in sample.items()}
    else:
        Exception(f'Unsupported batch item type: {name}')

################################################################


def run_parallel (fn, args):
    procs = []
    for arg in args:
        p = mp.Process(target=fn, args=(arg,))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()

def par_map (fn, args):
    def worker (i, send_end):
        result = fn(args[i])
        send_end.send((i, result))

    procs = []
    pipe_list = []
    for i in range(len(args)):
        recv_end, send_end = mp.Pipe(False)
        p = mp.Process(target=worker, args=(i, send_end))
        procs.append(p)
        pipe_list.append(recv_end)
        p.start()

    for p in procs:
        p.join()

    ret = [x.recv() for x in pipe_list]
    return ret

