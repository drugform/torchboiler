import os
import json
import shutil
import numpy as np
from copy import deepcopy

from ..version import __version__
from ..utils import LazyImport, set_seed, fprint

torch = LazyImport('torch')
cuda_available = LazyImport('cuda_available')
from .. import TrainBoiler, ExecBoiler, ExportBoiler

class HoldOut ():
    def __init__ (self, name, device,
                  model_root, train_root,
                  dataset, net_builder, net_cfg,
                  ensemble_cfg, train_cfg, export_cfg,
                  hooks={}, caller_globals={}):

        self.name = name
        self.make_config(ensemble_cfg)
        self.train_cfg = train_cfg
        self.export_cfg = export_cfg
        self.net_builder = net_builder
        self.net_cfg = net_cfg
        self.dataset = dataset
        self.device = device

        self.hooks = hooks
        self.caller_globals = caller_globals

        self.train_dir = os.path.join(train_root,
                                       name)
        self.model_dir = os.path.join(model_root,
                                       name)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        
        self.tb_cfg = TrainBoiler.make_config(self.train_cfg)
        self.train()
                
    def make_config (self, ensemble_cfg):
        # check nothing more given
        self.ensemble_cfg = {'name' : 'holdout'}
        self.ensemble_cfg.update(ensemble_cfg)
            
    def save (self, savedata={}):
        savedata = deepcopy(savedata)
        savedata['ensemble'] = self.ensemble_cfg
        savefile = os.path.join(self.model_dir,
                                'model.json')
        with open(savefile, 'w') as fp:
            json.dump(savedata, fp, indent=4)
            
    def load_ (self):
        fname = os.path.join(self.model_dir,
                             f"holdout.bin")
        assert os.path.exists(fname), f'Holdout model not finished'
        self.model = ExecBoiler(fname, self.device)

    def test (self, batch_size=None):
        train_split = int(len(self.dataset)*self.tb_cfg.train_prop)
        # obtaining validation subset
        if (not self.tb_cfg.recurrent
            and self.tb_cfg.shuffle):
            train_ds, valid_ds = torch.utils.data.random_split(
                self.dataset, (train_split, len(self.dataset)-train_split),
                generator=torch.Generator().manual_seed(42))
        else:
            ids = np.arange(len(self.dataset))
            train_ds = torch.utils.data.Subset(self.dataset, ids[ :train_split])
            valid_ds = torch.utils.data.Subset(self.dataset, ids[train_split: ])

        self.load_()
        pred, tgt = self.model(valid_ds,
                               batch_size=batch_size,
                               with_targets=True)
        return [[pred, tgt]]

    def test_external (self, test_dataset, batch_size=None):
        self.load_()
        pred, tgt = self.model(test_dataset,
                               batch_size=batch_size,
                               with_targets=True)
        return pred, tgt
    
    def train (self):
        set_seed(42)
        net = self.net_builder(**self.net_cfg)
        TB = TrainBoiler('holdout', self.tb_cfg,
                         self.dataset, net, self.device,
                         root=self.train_dir,
                         hooks=self.hooks,
                         caller_globals=self.caller_globals)

        best_file = TB.get_result()
        exp = ExportBoiler(best_file, net=net)
        fname = exp(**self.export_cfg)
        dst_fname = os.path.join(self.model_dir,
                                 f"holdout.bin")
        shutil.copyfile(fname, dst_fname)
        return dst_fname
    
    def __call__ (self, dataset, batch_size=None, value_only=True):
        bs = batch_size or self.model.cfg.batch_size
        pred = self.model(dataset, bs)
        if value_only:
            return pred
        return {'value' : pred,
                'ci' : np.nan}
