import numpy as np
import json
import math
import time
import copy
import os
import shutil
import joblib
from copy import deepcopy
from types import SimpleNamespace


from . import utils
from .utils import fprint
from . import criterion
from . import serialize
from . import collator

torch = utils.LazyImport("torch")
tensorboard = utils.LazyImport("torch.utils.tensorboard")        
        
class TrainBoiler ():
    def __init__ (self, name, config,
                  dataset, net, device,
                  start_from_checkpoint=None,
                  root='/tmp/torchboiler', hooks={},
                  caller_globals={}):
        self.version = '0.1.0'
        self.name = name
        utils.set_seed(42)
        self.device = device
        self.net = net
        if start_from_checkpoint is not None:
            self.load_net_from_checkpoint(
                start_from_checkpoint)

        self.net.to(device)

        self.cfg = config
        self.init_info()
        self.init_state()
        self.init_optimizer()
        self.init_criterion(dataset, caller_globals)
        self.init_collator(caller_globals)
        self.init_workdir(root)
        self.log_writer = tensorboard.SummaryWriter(
            log_dir=self.workdir)

        self.state.start_moment = time.time()
        self.train_loop(dataset, hooks)

    @staticmethod
    def make_config (args):
        cfg = {'batch_size' : 16,
               'criterion' : {'name' : 'mixed'},
               'collator'  : {'name' : 'default'},
               'optimizer' : {'name' : 'RAdam',
                              'lr' : 1e-3},
               'scheduler' : {'name' : 'ReduceLROnPlateau',
                              'mode' : 'min',
                              'factor' : 2./(1+math.sqrt(5)),
                              'patience' : 3,
                              'min_lr' : 1e-4},
               'n_epochs' : 50,
               'train_prop' : 0.9,
               'n_workers' : 0,
               'verbose' : True,
               'repeat_train' : 1,
               'repeat_valid' : 1,
               'with_restarts' : True,
               'shuffle' : True,
               'recurrent' : False,
               'maximize' : False,
               'attributes' : {}}
        
        for k,v in args.items():
            if type(cfg[k]) is dict:
                cfg[k].update(v)
            else:
                cfg[k] = v
        for k,v in cfg.items():
            if type(cfg[k]) is dict:
                cfg[k] = SimpleNamespace(**cfg[k])
        return SimpleNamespace(**cfg)
        
    def train_loop (self, dataset, hooks):
        train_loader, valid_loader = self.make_loaders(dataset)
        
        for ep in range(self.state.cur_epoch,
                        self.cfg.n_epochs):
            if self.state.early_stopped:
                break

            if os.path.exists(self.stop_file):
                os.remove(self.stop_file)
                fprint("Stopped on demand")
                break

            self.net.train()
            train_loss = self.epoch_step(ep, train_loader)
            self.log_writer.add_scalar('Loss/train', train_loss, ep)

            self.net.eval()
            with torch.no_grad():
                valid_loss = self.epoch_step(
                    ep, valid_loader, valid=True)            
            self.log_writer.add_scalar('Loss/valid', valid_loss, ep)
                
            if np.isnan(train_loss) or np.isnan(valid_loss):
                fprint('Got nan and stopped')
                break

            self.scheduler.step(valid_loss)
            status_msg = self.check_progress(valid_loss)
            
            if hooks.get('post_epoch_hook') is not None:
                hooks['post_epoch_hook'](self)

            self.state.cur_epoch = ep+1
            self.report_epoch(train_loss, valid_loss, status_msg)

            if ep == 0:
                self.format_dynamic_shapes()        

            self.save_checkpoint()

    def collect_sample_shapes (self, sample, prefix):
        # TODO: проверять полное соответствие ключей в collect_shapes
        shapes = utils.get_sample_shapes(sample)
        if self.state.shapes.get(prefix) is None:
            self.state.shapes[prefix] = {
                k:set() for k in shapes.keys()}

        for k,v in shapes.items():
            self.state.shapes[prefix][k].add(v)

    def collect_batch_shapes (self, ep, batch):
        if ep == 0:
            inpt, tgt = batch[0], batch[-1]
            self.collect_sample_shapes(inpt, 'in')
            self.collect_sample_shapes(out, 'out')

    def format_dynamic_shapes (self):
        def infer_dyn (self, shapes):
            dyn = {}
            for k,v in shapes:
                axis_diff_shapes = ( (v==v[0]).sum(axis=0)!=len(v) )
                axis_diff_shapes[0] = True # always write zero axis (batch)
                dyn[k] = np.where(axis_diff_shapes)[0].tolist()
            return dyn
        
        self.info.dynamic_shapes = {
            'in' : infer_dyn(self.state.shapes['in']),
            'out' : infer_dyn(self.state.shapes['out'])}
            
    def convert_train_batch (self, batch, to):
        inpt = utils.convert_sample(batch[0], to)
        tgt = utils.convert_sample(batch[-1], to)
        if len(batch) == 2:
            w = None
        else:
            w = utils.convert_sample(batch[1], to)
        return inpt, w, tgt
            
    def epoch_step (self, ep, loader, valid=False):
        total_loss, sample_counter = 0,0
        n_repeat = (self.cfg.repeat_valid
                    if valid
                    else self.cfg.repeat_train)
        if self.cfg.recurrent:
            hidden_state = self.net.init_hidden().to(self.device)

        for _ in range(n_repeat):
            t = utils.Progress(loader, self.cfg.verbose)
            for batch in t:
                self.collect_batch_shapes(ep, batch)
                inpt,w,y = self.convert_train_batch(batch, self.device)

                if not valid:
                    self.optimizer.zero_grad(set_to_none=True)

                if self.cfg.recurrent:
                    raise NotImplementedError()
                    self.set_sample_input(*inpt, hidden_state)
                    yp, hidden_state = self.net(*inpt, hidden_state)
                    hidden_state = hidden_state.detach()
                else:
                    self.set_sample_input(inpt)
                    yp = utils.forward(self.net, inpt)

                if not valid:
                    loss = self.criterion(yp, y, weights=w)
                    loss.backward()
                    self.optimizer.step()
                    
                report_loss = float(
                    self.criterion(yp, y,
                                   weights=w,
                                   unscaled=True).detach())
                
                t.update(round(np.sqrt(report_loss), 5))
                total_loss += report_loss*len(y)
                sample_counter += len(y)

            
        mean_loss = np.sqrt(total_loss / sample_counter)
        return float(mean_loss)
            
    def report_epoch (self, train_loss, valid_loss, status_msg):        
        elapsed_time = int(time.time()-self.state.start_moment)
        log_msg = (f"{self.name} / train: {train_loss:.4f} /"
                   f" valid: {valid_loss:.4f} /"
                   f" epoch {self.state.cur_epoch} /"
                   f"{self.device} / elapsed time {elapsed_time}s /"
                   f" lr: {self.get_rate():.5f} / {status_msg}")
        fprint(log_msg)
        self.state.log.append(log_msg)
        with open(os.path.join(self.workdir,
                               'train_log.txt'),
                  'a') as fp:
            fp.write(log_msg+'\n')

    def get_rate (self):
        return [p['lr'] for p
                in self.optimizer.param_groups][0]
    
    def init_workdir (self, root):
        os.makedirs(root, exist_ok=True)
        workdir = os.path.join(root, self.name)
        self.workdir = workdir
        self.checkpoint_file = os.path.join(
            workdir, 'checkpoint.bin')
        self.best_file = os.path.join(
            workdir, 'best.bin')
        self.stop_file = os.path.join(
            workdir, 'stop')

        if not os.path.exists(workdir):
            os.mkdir(workdir)
        elif os.path.exists(self.checkpoint_file):
            self.load_checkpoint()
            for log_msg in self.state.log:
                fprint(log_msg)
            if self.state.early_stopped or \
               self.state.cur_epoch+1 >= self.cfg.n_epochs:
                fprint('Checkpoint is already finalized')
                return
            else:
                log_msg = 'Resuming training from latest checkpoint'
                fprint(log_msg)
                self.state.log.append(log_msg)
                with open(os.path.join(self.workdir,
                                       'train_log.txt'),
                          'a') as fp:
                    fp.write(log_msg+'\n')

        else:
            fprint(f'Removing workdir {workdir} with no `checkpoint.json`')
            shutil.rmtree(workdir)
            os.mkdir(workdir)

    def check_progress (self, value, maximize=False):
        restart_ratio = 0.2
        progress_thr = 0.995
        early_ratio = 0.15
        
        tgt = progress_thr * self.state.best_value
        if self.cfg.maximize:
            # doing "not value <= tgt" instead of "value > tgt"
            # to support NaN as start point value
            # for both maximize=True and False
            has_progress = not (value <= tgt)
        else:
            has_progress = not (value >= tgt)
        
        if has_progress:
            self.state.best_value = value
            self.state.early_stop_counter = 0
            self.state.restart_done = False
            self.save_best()
            status_msg = "Got improvement"
        else:
            status_msg = ""

        if self.cfg.with_restarts and not self.state.restart_done:
            if ( self.state.early_stop_counter >
                 restart_ratio*early_ratio*self.cfg.n_epochs ):
                self.init_optimizer()
                self.state.restart_done = True
                status_msg = "Restarted"
                
        if ( self.state.early_stop_counter >
             early_ratio*self.cfg.n_epochs ): # early stop criterion
            self.state.early_stopped = True
            status_msg = "Early stopped"
        elif status_msg == "":
            epochs_left = int(early_ratio*self.cfg.n_epochs -
                              self.state.early_stop_counter)
            status_msg = f"{epochs_left} epochs till early stop"
            
        self.state.early_stop_counter += 1
        return status_msg
            
    def make_loaders (self, dataset):
        train_split = int(len(dataset)*self.cfg.train_prop)

        if self.cfg.shuffle:
            train_ds, valid_ds = torch.utils.data.random_split(
                dataset, (train_split, len(dataset)-train_split),
                generator=torch.Generator().manual_seed(42))
        else:
            ids = np.arange(len(dataset))
            train_ds = torch.utils.data.Subset(dataset, ids[ :train_split])
            valid_ds = torch.utils.data.Subset(dataset, ids[train_split: ])

        loader_shuffle = self.cfg.recurrent
        collate_fn = utils.dataset_attr(
            dataset, 'collate_fn', ignore_missing=True)
        
        train_loader = torch.utils.data.DataLoader(
            train_ds,
            batch_size = self.cfg.batch_size,
            shuffle = loader_shuffle,
            num_workers = self.cfg.n_workers,
            persistent_workers=True if self.cfg.n_workers > 0 else False,
            pin_memory = False,
            prefetch_factor=1 if self.cfg.n_workers > 0 else None,
            collate_fn = collate_fn or self.collator)
        valid_loader = torch.utils.data.DataLoader(
            valid_ds,
            batch_size = self.cfg.batch_size,
            shuffle = False,
            num_workers = self.cfg.n_workers,
            pin_memory = False,
            prefetch_factor=1 if self.cfg.n_workers > 0 else None,
            persistent_workers=True if self.cfg.n_workers > 0 else False,
            collate_fn = collate_fn or self.collator)
        return train_loader, valid_loader

    def init_info (self):
        self.info = SimpleNamespace()
        self.info.version = self.version
        self.info.dynamic_shapes = {}

    def init_state (self):
        self.state = SimpleNamespace()
        self.state.early_stop_counter = 0
        self.state.cur_epoch = 0
        self.state.best_value = np.nan
        self.state.early_stopped = False
        self.state.restart_done = False
        self.state.log = []
        self.state.shapes = dict()

    def init_optimizer (self):
        opt_cfg = deepcopy(self.cfg.optimizer.__dict__)
        opt_name = opt_cfg.pop('name')
        opt = torch.optim.__getattribute__(opt_name)
        self.optimizer = opt(params=self.net.parameters(),
                             **opt_cfg)

        sch_cfg = deepcopy(self.cfg.scheduler.__dict__)
        sch_name = sch_cfg.pop('name')
        sch = torch.optim.lr_scheduler.__getattribute__(sch_name)
        self.scheduler = sch(optimizer=self.optimizer,
                             **sch_cfg)

    def init_criterion (self, dataset, caller_globals):
        crit_cfg = deepcopy(self.cfg.criterion.__dict__)
        crit_name = crit_cfg.pop('name')

        crit_module = caller_globals.get(crit_name)
        if crit_module is None:
            crit_module = getattr(globals()['criterion'],
                                  crit_name)

        self.criterion = crit_module.Criterion(
            dataset, self.device, **crit_cfg)
        
    def init_collator (self, caller_globals):
        col_cfg = deepcopy(self.cfg.collator.__dict__)
        col_name = col_cfg.pop('name')
        
        col_module = caller_globals.get(col_name)
        if col_module is None:
            col_module = getattr(globals()['collator'],
                                  col_name)

        self.collator = col_module.Collator(**col_cfg)
        
    def set_sample_input (self, sample_input):
        # copy logic from utils.forward
        if not hasattr(self, 'sample_input'):
            self.sample_input = utils.convert_sample(
                sample_input, 'numpy')                
            
    def save_checkpoint (self):
        ckpt = {'format' : 'torch',
                'net' : self.net.state_dict(),
                'optimizer' : self.optimizer.state_dict(),
                'scheduler' : self.scheduler.state_dict(),
                'criterion' : self.criterion.state_dict(),
                'state' : self.state,
                'info' : self.info,
                'cfg' : self.cfg,
                'sample_input' : self.sample_input}
        serialize.pack(ckpt, self.checkpoint_file)
        
    def save_best (self):
        ckpt = {'format' : 'torch',
                'net' : self.net.state_dict(),
                'criterion' : self.criterion.state_dict(),
                'cfg' : self.cfg,
                'info' : self.info,
                'sample_input' : self.sample_input}
        serialize.pack(ckpt, self.best_file)

    def load_checkpoint (self):
        ckpt = serialize.unpack(self.checkpoint_file)
            
        # TODO: compare given cfg with loaded, print the diff            
        self.state = ckpt['state']
        self.net.load_state_dict(ckpt['net'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.scheduler.load_state_dict(ckpt['scheduler'])
        self.criterion.load_state_dict(ckpt['criterion'])
        del ckpt

    def load_net_from_checkpoint (self, checkpoint_file):
        ckpt = serialize.unpack(checkpoint_file)
        self.net.load_state_dict(ckpt['net'])
