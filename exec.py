import os
import io
from copy import deepcopy
import numpy as np

from . import utils
from .utils import fprint
from . import criterion
from . import serialize
from . import collator

torch = utils.LazyImport("torch")
onnxruntime = utils.LazyImport("onnxruntime")

class ExecBoiler ():
    def __init__ (self, model_path, device,
                  caller_globals={}, **args):
        self.device = device
        ckpt = serialize.unpack(model_path)
        self.cfg = ckpt['cfg']
        self.load_criterion(ckpt['criterion'],
                            caller_globals)
        self.set_attributes()
        self.format = ckpt['format']

        if self.format == 'torch':
            self.net = load_net_torch(ckpt, device,
                                      net=args['net'])
            self.convert_to = self.device
        elif self.format == 'torchscript':
            self.net = load_net_torchscript(ckpt, device)
            self.convert_to = self.device            
        elif self.format == 'onnx':
            self.net = load_net_onnx(ckpt, device,
                                     backend=args.get('backend'))
            self.convert_to = 'numpy'
        else:
            raise Exception(f'Unknown format: {self.format}')

        self.init_collator(caller_globals)
        del ckpt

    def init_collator (self, caller_globals={}):
        # copy from train.py
        col_cfg = deepcopy(self.cfg.collator.__dict__)
        col_name = col_cfg.pop('name')
        
        col_module = caller_globals.get(col_name)
        if col_module is None:
            col_module = getattr(globals()['collator'],
                                  col_name)

        self.collator = col_module.Collator(**col_cfg)

    def set_attributes (self):
        for k,v in self.cfg.attributes.__dict__.items():
            if getattr(self, k, None) is None:
                setattr(self, k, v)
            else:           
                fprint(f'Warning: cannot set top-level attribute: {k}. ExecBoiler already has an attr with this name. You can still access it as self.cfg.attributes["{k}"]')
    
    def load_criterion (self, state_dict, caller_globals):
        # duplicate with train.py
        crit_cfg = deepcopy(self.cfg.criterion.__dict__)
        crit_name = crit_cfg.pop('name')

        crit_module = caller_globals.get(crit_name)
        if crit_module is None:
            crit_module = getattr(globals()['criterion'],
                                  crit_name)

        self.criterion = crit_module.CriterionPortable(**crit_cfg)
        self.criterion.load_state_dict(state_dict)

    def forward (self, inpt):
        if 'torch' in self.format:
            out = forward_torch(self.net, inpt)
        else:
            out = forward_onnx(self.net, inpt)
        pred = self.criterion.postproc(out)
        return pred
            
    def __call__ (self, dataset,
                  batch_size=None, with_targets=False):
        collate_fn = utils.dataset_attr(
            dataset, 'collate_fn', ignore_missing=True)
        data_loader = utils.DataLoader(
            dataset,
            batch_size = (batch_size or self.cfg.batch_size),
            collate_fn = collate_fn or self.collator)

        """
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size = (batch_size or self.cfg.batch_size),
            shuffle = False,
            num_workers = 0,
            pin_memory = True,
            collate_fn = collate_fn or self.collator)
        """

        preds,targets = [],[]
        if with_targets:
            for batch in data_loader:
                inpt = utils.convert_sample(batch[0], self.convert_to)
                pred = self.forward(inpt)
                preds += list(pred)

                tgt = utils.convert_sample(batch[-1], 'numpy')

                targets += list(tgt)
            return preds, targets
        
        else:
            for batch in data_loader:
                inpt = utils.convert_sample(batch, self.convert_to)
                pred = self.forward(inpt)
                preds += list(pred)
            return preds

    
################################################################
### load
    
def load_net_torch (ckpt, device, net):
    net.load_state_dict(ckpt['net'])
    net.to(device)
    net.eval()
    return net

def load_net_torchscript (ckpt, device):
    net = ckpt['net']
    net.to(device)
    net.eval()
    return net

def load_net_onnx (ckpt, device, backend=None):
    if backend is None:
        if device == 'cpu':
            provider = "CPUExecutionProvider"
        elif device == 'cuda':
            provider = "CUDAExecutionProvider"
        else:
            device_id = int(device.split('cuda:')[1])
            provider = ("CUDAExecutionProvider",
                        {"device_id" : device_id})
    else:
        raise NotImplementedError(
            f'ONNX backend {backend} not supported')
    
    with io.BytesIO(ckpt['net']) as fp:
        net_bytes = fp.read()
        net = onnxruntime.InferenceSession(
            net_bytes, providers=[provider])
    return net

### forward

def forward_torch (net, input_):
    with torch.no_grad():
        yp = utils.forward(net, input_)
        yp = yp.detach().cpu().numpy()
    return yp

def forward_onnx (net, input_):
    if type(input_) is dict:
        ort_input = input_        
    else:
        if type(input_) in (list, tuple):
            pass
        elif type(input_) == np.ndarray:
            input_ = (input_,)
        else:
            Exception(f'Unsupported network input type: {type(input_)}')
        ort_input = {input_arg.name: input_value
                     for input_arg, input_value
                     in zip(net.get_inputs(), input_)}
        # todo: check all inputs are np.ndarray
        # todo: auto convert input type to get_input().type
    ort_output = net.run(None, ort_input)
    if len(ort_output) == 1:
        return ort_output[0]
    else:
        return ort_output
    
