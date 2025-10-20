import os
import json

def make_import (ens_type):
    if ens_type == "kfold":
        from .kfold import KFold as Ens
        return Ens
    else:
        raise Exception(f"Unknown model type: {ens_type}")

class EnsembleBoiler ():
    @staticmethod
    def train (name, device,
               model_root, train_root,
               dataset, net_builder, net_cfg,
               ensemble_cfg, train_cfg, export_cfg):

        ens_type = ensemble_cfg.pop('name')
        Ens = make_import(ens_type)
        
        return Ens(name, device,
                   model_root, train_root,
                   dataset, net_builder, net_cfg,
                   ensemble_cfg, train_cfg, export_cfg)

    @staticmethod
    def load (model_dir, device):
        model_file = os.path.join(model_dir,
                                  'model.json')
        with open(model_file, 'r') as fp:
            info = json.load(fp)
        
        ensemble_cfg = info["ensemble"]
        ens_type = ensemble_cfg.pop('name')
        Ens = make_import(ens_type)
        ens = Ens.__new__(Ens)
        ens.model_dir = model_dir
        ens.device = device
        ens.ensemble_cfg = ensemble_cfg
        ens.info = info
        ens.load_()
        return ens
