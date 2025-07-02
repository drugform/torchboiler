import io
import numpy as np
from types import SimpleNamespace
import msgpack

from . import utils

torch = utils.LazyImport("torch")

EXT_TYPES = ['numpy', 'torch', 'torchscript', 'namespace']
_EXT_CODES_REV = dict(enumerate(EXT_TYPES))
_EXT_CODES = {v:k for k,v in enumerate(EXT_TYPES)}

def _packer (obj):
    name = obj.__class__.__name__
    if isinstance(obj, np.ndarray):
        return msgpack.ExtType(
            *encode_numpy(obj))
    elif name == 'SimpleNamespace':
        return msgpack.ExtType(
            *encode_namespace(obj))
    elif name in ['RecursiveScriptModule',
                  'TopLevelTracedModule']:
        return msgpack.ExtType(
            *encode_torchscript(obj))
    elif name == 'Tensor':
        return msgpack.ExtType(
            *encode_torch(obj))
    else:
        raise Exception(f"no encoder for {obj} (class name: {name})")

def packb (obj):
    return msgpack.packb(obj,
                         default=_packer,
                         use_bin_type=True)

def pack (obj, fname):
    with open(fname, 'wb') as fp:
        return msgpack.pack(obj, fp,
                            default=_packer,
                            use_bin_type=True)
    

def encode_namespace (obj):
    obj_type = 'namespace'
    code = _EXT_CODES[obj_type]
    return code, packb(obj.__dict__)

def encode_numpy (obj):
    obj_type = 'numpy'
    code = _EXT_CODES[obj_type]
    allowed_dtypes = {'i', 'b', 'f', 'u', 'c', 'S'}
    if obj.dtype.kind not in allowed_dtypes:
        raise Exception(f'Cannot encode numpy {obj.dtype} array')
    with io.BytesIO() as f:
        np.save(f, obj, allow_pickle=False)
        dump = f.getvalue()
    return code, dump

def encode_torch (obj):
    obj_type = 'torch'
    code = _EXT_CODES[obj_type]
    with io.BytesIO() as f:
        torch.save(obj, f)
        dump = f.getvalue()
    return code, dump

def encode_torchscript (obj):
    obj_type = 'torchscript'
    code = _EXT_CODES[obj_type]
    with io.BytesIO() as f:
        torch.jit.save(obj, f)
        dump = f.getvalue()
    return code, dump

################################################################

def _unpacker (code, data):
    obj_type = _EXT_CODES_REV[code]
    if obj_type == 'numpy':
        return decode_numpy(data)
    elif obj_type == 'torch':
        return decode_torch(data)
    elif obj_type == 'torchscript':
        return decode_torchscript(data)
    elif obj_type == 'namespace':
        return decode_namespace(data)
    else:
        raise Exception(f'unknown type: {obj_type}')

def unpackb (obj):
    return msgpack.unpackb(obj,
                           ext_hook=_unpacker,
                           raw=False)

def unpack (fname):
    with open(fname, 'rb') as fp:
        return msgpack.unpack(fp,
                              ext_hook=_unpacker,
                              strict_map_key=False,
                              raw=False)


def decode_namespace (data):
    return SimpleNamespace(
        **unpackb(data))
    

def decode_numpy (data):
    with io.BytesIO(data) as f:
        arr = np.load(f, allow_pickle=False)
    return arr

def decode_torch (data):
    with io.BytesIO(data) as f:
        tensor = torch.load(f,
                            weights_only=True,
                            map_location='cpu')
    return tensor

def decode_torchscript (data):
    with io.BytesIO(data) as f:
        module = torch.jit.load(f,
                            map_location='cpu')
    return module
