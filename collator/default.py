import numpy as np
import collections.abc


# TODO: return contiguous arrays
class Collator ():
    def __init__ (self, depth=3, pad_axes=None, pad_value=0, **kwargs):
        self.depth = depth
        self.pad_axes = pad_axes
        self.pad_value = pad_value
        
        self.collate_fn_map = {np.ndarray: self.collate_numpy_array_fn}
        for type_ in np.ScalarType:
            self.collate_fn_map[type_] = self.collate_numpy_scalar_fn

    def collate_numpy_scalar_fn (self, batch):
        ret = np.asarray(batch)
        if (ret.dtype == np.float64 and
            type(batch[0]) is float):
            # fixing default numpy float type: float64->float32
            ret = ret.astype(np.float32)
        return ret
    
    def collate_numpy_array_fn (self, batch):
        if self.pad_axes is None:
            return np.stack(batch)

        if self.pad_axes == 'all':
            n_axes = len(batch[0].shape)
            pad_axes = list(range(n_axes))
        else:
            pad_axes = self.pad_axes
        
        shapes = np.array([sample.shape for sample in batch])
        uniq_shapes_to_pad = np.unique(shapes[:, pad_axes])
        if len(uniq_shapes_to_pad) == 1:
            return np.stack(batch)
    
        pack_shape = [len(batch)] + shapes.max(axis=0).tolist()
        packed = np.zeros(pack_shape,
                      dtype=batch[0].dtype)
        packed.fill(self.pad_value)
        for sample_id,sample in enumerate(batch):
            slices = [sample_id]+[slice(0,d) for d in sample.shape]
            for axis_id,dim in enumerate(sample.shape):
                if axis_id in pad_axes:
                    slices.append(slice(0, dim))
                else:
                    max_dim = packed.shape[axis_id+1]
                    slices.append(slice(0, max_dim))
                    slices.append(...)
            packed[tuple(slices)] = sample
    
        return packed
            
    def collate (self, batch, depth):
        if depth < 0:
            raise RuntimeError('Collation recursion depth reached. try less data comlicated structure')
        
        elem = batch[0]
        elem_type = type(elem)
        
        if elem_type in self.collate_fn_map:
            return self.collate_fn_map[elem_type](batch)

        if isinstance(elem, collections.abc.Mapping):
            return elem_type(
                {key: self.collate(
                    [d[key] for d in batch],
                    depth-1)
                 for key in elem})

        elif isinstance(elem, collections.abc.Sequence):
            #it = iter(batch)
            #elem_size = len(next(it))
            #if not all(len(elem) == elem_size for elem in it):
            #    raise RuntimeError("each element in list of batch should be of equal size")
            transposed = list(zip(*batch))
            return elem_type([self.collate(elem, depth-1)
                              for elem in transposed])
        else:
            raise TypeError(f'Cannot collate data type: {elem_type}')

    def __call__ (self, batch):
        return self.collate(batch, self.depth)
