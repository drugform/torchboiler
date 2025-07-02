import io
import os
from copy import deepcopy

from . import utils
from . import serialize

torch = utils.LazyImport("torch")

class ExportBoiler ():
    def __init__ (self, model_path, net=None):
        self.model_path = model_path
        self.dict = serialize.unpack(model_path)

        if self.dict['format'] == 'torch':
            self.net = net.cpu()
            self.net.load_state_dict(self.dict['net'])
        elif self.dict['format'] in ['torch.trace',
                                    'torch.script']:
            assert net is None, 'Net arg not required for {format} format'
            self.net = self.dict['net']
        else:
            raise Exception(f'Expecting a torch model as export source. Got format: {format}')

    def to_torchscript (self, trace=False):
        dict_ = deepcopy(self.dict)
        self.net.eval()
        if trace:
            dict_['format'] = 'torchtrace'
            sample_input = torch.from_numpy(
                dict_['sample_input'])
            torch_input = sample_input
            #with torch.no_grad():
            model = torch.jit.trace(self.net, torch_input)
            out_file = os.path.splitext(self.model_path)[0]+'.tt.bin'
        else:
            dict_['format'] = 'torchscript'
            model = torch.jit.script(self.net)
            out_file = os.path.splitext(self.model_path)[0]+'.ts.bin'
        
        dict_['net'] = model
        serialize.pack(dict_, out_file)

    def to_onnx (self, dynamo=True, dynamic_axes=None):
        dict_ = deepcopy(self.dict)
        self.net.eval()
        sample_input = torch.from_numpy(
            dict_['sample_input'])
        out_file = os.path.splitext(self.model_path)[0]+'.onnx.bin'
        onnx_tmp_file = "tmp.onnx" # use tmpfile

        if dynamic_axes is None:
            dynamic_axes = {'input' : [0]}

        # if dynamo=True, param name is dynamic_shapes,
        # not dynaic_axes
            
        onnx_program = torch.onnx.export(
            self.net,
            (sample_input,),
            onnx_tmp_file,
            dynamic_axes=dynamic_axes,
            dynamo=dynamo)

        if dynamo:
            onnx_program.optimize()
            onnx_program.save(onnx_tmp_file)

        with open(onnx_tmp_file, 'rb') as fp:
            onnx_bytes = fp.read()

        os.remove(onnx_tmp_file)
        dict_['format'] = 'onnx'
        dict_['net'] = onnx_bytes
        serialize.pack(dict_, out_file)

        
