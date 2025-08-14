import io
import os
from copy import deepcopy
import tempfile

from . import utils
from . import serialize

torch = utils.LazyImport("torch")

class ExportBoiler ():
    def __init__ (self, model_path, net=None):
        self.model_path = model_path
        self.dict = serialize.unpack(model_path)
        self.sample_input = self.dict['sample_input']
        if self.dict['format'] == 'torch':
            self.net = net.cpu()
            self.net.load_state_dict(self.dict['net'])
        elif self.dict['format'] == 'torchscript':
            assert net is None, 'Net arg not required for {format} format'
            self.net = self.dict['net']
        else:
            raise Exception(f'Expecting a torch model as export source. Got format: {format}')

    def __call__ (self, name, **params):
        if name == 'torchscript':
            return self.to_torchscript(**params)
        elif name == 'onnx':
            return self.to_onnx(**params)
        
    def to_torchscript (self, trace=False, keep_sample_input=False):
        dict_ = deepcopy(self.dict)
        self.net.eval()
        dict_['format'] = 'torchscript'
        out_file = os.path.splitext(self.model_path)[0]+'.ts.bin'
        if trace:
            torch_input = utils.convert_sample(
                self.sample_input.copy(), 'cpu')
            model = torch.jit.trace(self.net, torch_input)
        else:
            model = torch.jit.script(self.net)
        
        dict_['net'] = model
        if not keep_sample_input:
            del dict_['sample_input']
        serialize.pack(dict_, out_file)
        return out_file
        
    def to_onnx (self, dynamic_axes=None, keep_sample_input=False):
        dict_ = deepcopy(self.dict)
        self.net.eval()
        sample_input = utils.convert_sample(
            self.sample_input, 'cpu')
        out_file = os.path.splitext(self.model_path)[0]+'.onnx.bin'

        artifacts_dir = os.path.dirname(self.model_path)
        tmp_name = next(tempfile._get_candidate_names())
        onnx_tmp_file = os.path.join(artifacts_dir, f"{tmp_name}.onnx")
        onnx_args = {'model' : self.net,
                     'dynamo' : True,
                     'report' : True,
                     'artifacts_dir' : artifacts_dir}
        
        if type(sample_input) is dict:
            onnx_args['kwargs'] = sample_input
        else:
            onnx_args['args'] = sample_input
        
        if type(dynamic_axes) is str:
            onnx_args.update(
                onnx_format_dynamic_axes(
                    self.net, sample_input, dynamic_axes))
        else:
            onnx_args.update(dynamic_axes)

        onnx_program = torch.onnx.export(**onnx_args)
        onnx_program.optimize()
        onnx_program.save(onnx_tmp_file)

        with open(onnx_tmp_file, 'rb') as fp:
            onnx_bytes = fp.read()

        dict_['format'] = 'onnx'
        dict_['net'] = onnx_bytes
        if not keep_sample_input:
            del dict_['sample_input']
        serialize.pack(dict_, out_file)
        try:
            os.remove(onnx_tmp_file)
        except Exception as e:
            print(f'Failed to delete onnx tmp file `{onnx_tmp_file}` : {e}')
        return out_file

def onnx_format_dynamic_axes (net, sample_input, mode):
    if mode == 'auto_batch':
        sample_output = utils.forward(net, sample_input)
        # TODO: let sample_output be dict
        
        if type(sample_input) is dict:
            onnx_program = torch.onnx.export(
                model=net,
                kwargs=sample_input,
                dynamo=True)
        else:
            onnx_program = torch.onnx.export(
                model=net,
                args=sample_input,
                dynamo=True)
            
        input_names = [inp.name for inp
                       in onnx_program.model.graph.inputs]
        
        if type(sample_output) is not tuple:
            sample_output = (sample_output,)
        
        output_names = [f'output_{i}' for i
                        in range(len(sample_output))]
        
        dynamic_axes = {k:[0] for k
                        in input_names+output_names}
        return {'input_names' : input_names,
                'output_names' : output_names,
                'dynamic_axes' : dynamic_axes}
    else:
        raise Exception(f'Unknown dynamic axes mode: {mode}')
