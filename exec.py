import os
import io
from copy import deepcopy
import numpy as np

from . import utils
from .utils import fprint
from . import criterion
from . import serialize

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
            self.runner = ExecTorch(ckpt, device, **args)
        elif self.format == 'torchscript':
            self.runner = ExecTorchScript(ckpt, device, **args)
        elif self.format == 'onnx':
            self.runner = ExecONNX(ckpt, device, **args)
        else:
            raise Exception(f'Unknown format: {self.format}')

        del ckpt

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

    def __call__ (self, input_):
        output = self.runner(input_)
        return self.criterion.postproc(output)

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

class ExecTorch ():
    def __init__ (self, ckpt, device, **kwargs):
        self.net = load_net_torch(ckpt, device,
                                  net=kwargs['net'])
    def __call__ (self, inpt):
        return forward_torch(self.net, inpt)

class ExecTorchScript ():
    def __init__ (self, ckpt, device, **kwargs):
        self.net = load_net_torchscript(ckpt, device)
    def __call__ (self, inpt):
        return forward_torch(self.net, inpt)

class ExecONNX ():
    def __init__ (self, ckpt, device, **kwargs):
        self.net = load_net_onnx(ckpt, device,
                                 backend=kwargs.get('backend'))
    def __call__ (self, inpt):
        return forward_onnx(self.net, inpt)


    
"""    
class ExecBoiler ():
    def __init__ (self, model_path, device,
                  caller_globals={}, **args):
        self.device = device
        ckpt = serialize.unpack(model_path)

        self.cfg = ckpt['cfg']
        self.load_criterion(ckpt['criterion'],
                            caller_globals)

        self.format = ckpt['format']
        
        if self.format == 'torch':
            self.load_net_torch(ckpt, args['net'])
        elif self.format in ['torchscript',
                             'torchtrace']:
            self.load_net_torchscript(ckpt)
        elif self.format == 'onnx':
            self.load_net_onnx(ckpt)
        else:
            raise Exception(f"Unknown network format: {self.format}")
        
        del ckpt

        self.set_attributes()
        
    def set_attributes (self):
        for k,v in self.cfg.attributes.items():
            if getattr(self, k, None) is None:
                setattr(self, k, v)
            else:           
                fprint(f'Warning: cannot set top-level attribute: {k}. ExecBoiler already has an attr with this name. You can still access it as self.cfg.attributes["{k}"]')
        
    def load_net_torch (self, ckpt, net):
        self.net = net
        self.net.load_state_dict(ckpt['net'])
        self.net.to(self.device)
        self.net.eval()

    def load_net_torchscript (self, ckpt):
        self.net = ckpt['net']
        self.net.to(self.device)
        self.net.eval()

    def load_net_onnx (self, ckpt):
        with io.BytesIO(ckpt['net']) as fp:
            net_bytes = fp.read()
        if self.device == 'cpu':
            prov = "CPUExecutionProvider"
        else:
            prov = "CUDAExecutionProvider"
        self.net = onnxruntime.InferenceSession(
            net_bytes, providers=[prov])        
                
    def load_criterion (self, state_dict, caller_globals):
        # duplicate with train.py
        crit_cfg = deepcopy(self.cfg.criterion)
        crit_name = crit_cfg.pop('name')

        crit_module = caller_globals.get(crit_name)
        if crit_module is None:
            crit_module = getattr(globals()['criterion'],
                                  crit_name)

        self.criterion = crit_module.CriterionPortable(**crit_cfg)
        self.criterion.load_state_dict(state_dict)

    def forward_torch (self, input_):
        with torch.no_grad():
            if self.cfg.recurrent:
                raise NotImplementedError()
                yp, hidden_state = utils.forward(
                    self.net, *input_, hidden_state)
                hidden_state = hidden_state.detach()
            else:
                yp = utils.forward(self.net, input_)
            yp = yp.detach().cpu().numpy()
            return yp

    def forward_onnx (self, input_):
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
                         in zip(self.net.get_inputs(), input_)}
            # todo: check all inputs are np.ndarray
            # todo: auto convert input type to get_input().type
        ort_output = self.net.run(None, ort_input)
        if len(ort_output) == 1:
            return ort_output[0]
        else:
            return ort_output

    def forward_onnx_cuda (self, input_):
        DEVICE_NAME='cuda'
        DEVICE_INDEX=0
        x_ortvalue = onnxruntime.OrtValue.ortvalue_from_numpy(input_, DEVICE_NAME, DEVICE_INDEX)
        io_binding = self.net.io_binding()
        io_binding.bind_input(name='x', device_type=x_ortvalue.device_name(), device_id=0, element_type=input_.dtype, shape=x_ortvalue.shape(), buffer_ptr=x_ortvalue.data_ptr())
        io_binding.bind_output(name='linear', device_type=DEVICE_NAME, device_id=DEVICE_INDEX, element_type=input_.dtype, shape=x_ortvalue.shape())
        self.net.run_with_iobinding(io_binding)
        z = io_binding.get_outputs()
        return z[0].numpy()
        
    def __call__ (self, input_):
        if self.format in ['torch',
                           'torchscript',
                           'torchtrace']:
            input_,bs = utils.input2device(input_, self.device)
            yp = self.forward_torch(input_)
        elif self.format == 'onnx':
            yp = self.forward_onnx(input_)
        else:
            raise Exception(f'Unknown model format: {self.format}')
        yp = self.criterion.postproc(yp)
        return yp
        
    def predict_dataset (self, dataset, batch_size=1, verbose=True):
        if self.format not in ['torch',
                               'torchscript',
                               'torchtrace']:
            raise Exception(f"Will not run {self.format} model at torch dataset")
        collate_fn = utils.dataset_attr(
            dataset, 'collate_fn', ignore_missing=True)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn)
        
        if self.cfg.recurrent:
            raise NotImplementedError()
            hidden_state = self.net.init_hidden().to(self.device)

        t = utils.Progress(loader, verbose)
        pred = []
        for batch in t:
            input_,w,y = utils.parse_batch(batch, self.device)
            yp = self(input_)
            pred.append(yp)
            
        pred = np.vstack(pred)
        return pred
        
"""
