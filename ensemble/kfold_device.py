import os
import json
import shutil
import numpy as np
from copy import deepcopy

from ..version import __version__
from ..utils import LazyImport, set_seed, run_parallel

torch = LazyImport('torch')
from .. import TrainBoiler, ExecBoiler, ExportBoiler


class FoldView ():
    def __init__ (self, ds, fold_id, n_folds, mode='all'):
        if hasattr(ds, 'regression_flags'): 
            self.regression_flags = ds.regression_flags
        if hasattr(ds, 'class_weights'): 
            self.class_weights = ds.class_weights
            
        assert fold_id < n_folds

        all_ids = np.arange(len(ds))
        step = len(all_ids) / n_folds
        test_from = int(step * fold_id)
        test_to = int(step * (fold_id+1))

        test_ids = all_ids[test_from:test_to]
        train_ids = np.setdiff1d(all_ids, test_ids)

        self.use_ids = {'all' : np.concatenate((train_ids, test_ids)),
                        'test_only' : test_ids,
                        'train_only' : train_ids}[mode]
        self.ds = ds
        self.mode = mode
        
    def __len__ (self):
        return len(self.use_ids)

    def __getitem__ (self, idx):
        real_idx = self.use_ids[idx]
        return self.ds[real_idx]

    
class KFold ():
    def __init__ (self, name, device,
                  model_root, train_root,
                  dataset, net_builder, net_cfg,
                  ensemble_cfg, train_cfg, export_cfg):
        self.name = name
        self.make_config(ensemble_cfg)
        self.train_cfg = train_cfg
        self.export_cfg = export_cfg
        self.net_builder = net_builder
        self.net_cfg = net_cfg
        self.dataset = dataset
        self.device = device

        self.train_dir = os.path.join(train_root,
                                       name)
        self.model_dir = os.path.join(model_root,
                                       name)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        
        self.n_folds = ensemble_cfg['n_folds']
        self.tb_cfg = TrainBoiler.make_config(self.train_cfg)
        self.tb_cfg.train_prop = (self.n_folds-1)/self.n_folds
        self.tb_cfg.shuffle = False

        if type(self.device) is str:
            for fold_id in range(self.n_folds):
                self.train_fold(fold_id)            
        else:
            run_parallel(self.train_fold,
                    list(range(self.n_folds)))
                
    def make_config (self, ensemble_cfg):
        self.ensemble_cfg = {'name' : 'kfold',
                             'n_folds' : 4,
                             'parallel' : False}
        self.ensemble_cfg.update(ensemble_cfg)
            
    def save (self, savedata={}):
        savedata = deepcopy(savedata)
        savedata['ensemble'] = self.ensemble_cfg
        savefile = os.path.join(self.model_dir,
                                'model.json')
        with open(savefile, 'w') as fp:
            json.dump(savedata, fp, indent=4)
            
    def load_ (self):
        self.n_folds = self.ensemble_cfg['n_folds']
        self.submodels = []
        for fold_id in range(self.n_folds):
            fname = os.path.join(self.model_dir,
                                 f"fold{fold_id}.bin")
            assert os.path.exists(fname), f'Fold{fold_id} model not finished'
            device = self.fold_id2device(fold_id)
            self.submodels.append(
                ExecBoiler(fname, device))

    def fold_id2device (self, fold_id):
        if type(self.device) is str:
            return self.device

        n_devices = len(self.device)
        device_id = fold_id % n_devices
        return self.device[device_id]
    
    def train_fold (self, fold_id):
        set_seed(42)
        net = self.net_builder(**self.net_cfg)
        data_view = FoldView(self.dataset,
                             fold_id,
                             self.n_folds,
                             mode='all')

        device = self.fold_id2device(fold_id)
        subname = f"fold{fold_id}"
        TB = TrainBoiler(subname, self.tb_cfg,
                         data_view, net, device,
                         root=self.train_dir)

        best_file = TB.get_result()
        exp = ExportBoiler(best_file, net=net)
        fname = exp(**self.export_cfg)
        dst_fname = os.path.join(self.model_dir,
                                 f"fold{fold_id}.bin")
        shutil.copyfile(fname, dst_fname)
        return dst_fname

    def test_fold (self, fold_id):
        data_view = FoldView(self.dataset,
                             fold_id,
                             self.n_folds,
                             mode='test_only')
        submodel = self.submodels[fold_id]
        pred, tgt = submodel(data_view, with_targets=True)
        return pred, tgt        

    def test (self):
        self.load_()
        results = []
        for fold_id in range(self.n_folds):
            results.append(
                self.test_fold(fold_id))
        return results
    
    def __call__ (self, dataset, batch_size=None, value_only=True):
        bs = batch_size or self.submodels[0].cfg.batch_size

        sub_preds = [submodel(dataset, bs)
                     for submodel in self.submodels]
            
        pred = np.mean(sub_preds, axis=0)
        if value_only:
            return pred
        
        stds = np.std(sub_preds, ddof=1, axis=0)
        n = len(self.submodels)
        student_coef = scipy.stats.t.ppf((1 + 0.95) / 2., n-1)
        ci_norm = student_coef * stds / np.sqrt(n)
        return {'value' : pred,
                'ci' : ci_norm}
