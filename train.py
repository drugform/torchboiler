import numpy as np
import datetime
import json
import math
import time
import copy
import os
import shutil
import socket
from copy import deepcopy
from types import SimpleNamespace
from . import utils
from .utils import fprint
from . import criterion
from . import serialize
from . import collator
from .version import __version__
torch = utils.LazyImport("torch")
tensorboard = utils.LazyImport("torch.utils.tensorboard")        

def is_port_free (port, host='localhost'):
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind((host, port))
            return True
    except (socket.error, OSError):
        return False

def get_next_free_port (port):
    for try_port in range(port, 30000):
        if  is_port_free(try_port):
            return try_port
    raise Exception('No free ports for DDP in range: {port}-30000')
        


class TrainBoiler ():
    def __init__ (self, name, config,
                  dataset, net, device,
                  root='/tmp/torchboiler', hooks={},
                  caller_globals={}):
        self.root = root
        self.name = name
        self.workdir = os.path.join(self.root, self.name)
        if self.get_result() is not None:
            fprint(f'Training `{self.name}` already finished')
            return


        args = [name, config, dataset, net, device,
                root, hooks, caller_globals]
        if not config.ddp:
            _TrainBoiler(*args)
        else:
            if hooks != {}:
                print('WARNING: be careful with post-epoch hooks in DDP mode')
            # проверку на делимость батч-сайза перенести сюда
            if device != 'cuda':
                raise Exception(f'Cannot run DDP training at device: {device}')
            torch.multiprocessing.set_start_method('spawn', force=True)
            world_size = torch.cuda.device_count()
            os.environ['MASTER_ADDR'] = 'localhost'
            os.environ['MASTER_PORT'] = str(get_next_free_port(29500))
            # can launch as daemon if no subprocesses spawned
            daemon_flag = (config.n_workers==0)
            procs = []
            for rank in range(world_size):
                args[4] = f'cuda:{rank}' # переписать красиво
                p = torch.multiprocessing.Process(
                    target=_TrainBoiler, args=args,
                    daemon=daemon_flag)
                p.start()
                procs.append(p)
            
            for p in procs:
                p.join()
        
        # нужна ли доп проверка успешности завершения?
        # или перенести создание success-файла в rank=0?
        open(os.path.join(self.workdir, 'success'),
             'w').close()
        
        if self.get_result() is None:
            raise Exception(f'Training `{self.name}` failed')
        else:
            fprint(f'Training `{self.name}` finished')            

    def get_result (self):
        best_file = os.path.join(self.workdir, 'best.bin')
        success_file = os.path.join(self.workdir, 'success')
        if (os.path.exists(success_file)
            and os.path.exists(best_file)):
            return best_file
        else:
            return None

    @staticmethod
    def make_config (args):
        cfg = {'batch_size' : 64,
               'criterion' : {'name' : 'mixed'},
               'collator'  : {'name' : 'default'},
               'optimizer' : {'name' : 'RAdam',
                              'lr' : 3e-4},
               'scheduler' : {'name' : 'ReduceLROnPlateau',
                              'mode' : 'min',
                              'factor' : 2./(1+math.sqrt(5)),
                              'patience' : 3,
                              'min_lr' : 3e-5},
               'dataparallel' : False,
               'ddp' : False,
               'n_epochs' : 50,
               'save_each_epoch' : False,
               'tune_steps' : 0,
               'train_prop' : 0.9,
               'n_workers' : 0,
               'verbose' : True,
               'repeat_train' : 1,
               'repeat_valid' : 1,
               'with_restarts' : True,
               'shuffle' : True,
               'recurrent' : False,
               'maximize' : False,
               'clip_grad' : False,
               'start_from_checkpoint' : None,
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
            

class _TrainBoiler ():
    def is_main (self):
        return (self.rank == 0)
    
    def __init__ (self, name, config,
                  dataset, net, device,
                  root='/tmp/torchboiler', hooks={},
                  caller_globals={}):
        if config.ddp: 
            self.rank = int(device.split('cuda:')[1])
            self.world_size = torch.cuda.device_count()
            torch.distributed.init_process_group(
                # добавить gloo для cpu
                'nccl',
                rank=self.rank,
                world_size=self.world_size,
                timeout=datetime.timedelta(minutes=30))
        else:
            self.rank = 0
        
        self.device = device
        self.root = root
        self.name = name
        
        utils.set_seed(42)
        self.cfg = config
        
        self.net = net

        self.net.to(device)
        if self.cfg.ddp:
            self.net = torch.nn.parallel.DistributedDataParallel(
                self.net,
                device_ids=[self.rank],
                output_device=self.rank)

        if self.cfg.start_from_checkpoint is not None:
            self.load_net_from_checkpoint(
                self.cfg.start_from_checkpoint)
            
        self.init_info()
        self.init_state()
        self.init_optimizer()
        # Initializing criterion at the whole dataset.
        # No ddp sync required
        self.init_criterion(dataset, caller_globals)
        self.init_collator(caller_globals)
        self.init_workdir(root)
        if self.is_main():
            self.log_writer = tensorboard.SummaryWriter(
                log_dir=self.workdir)

        if self.cfg.ddp:
            # нужно ли ставить барьер после каждой эпохи?
            torch.distributed.barrier(device_ids=[self.rank])
        self.state.start_moment = time.time()
        
        can_continue = self.main_loop(dataset, hooks)
        if can_continue and self.cfg.tune_steps > 0:
            self.load_checkpoint()
            if config.ddp:
                torch.distributed.barrier(device_ids=[self.rank])
            if self.state.tune_steps_counter == 0:
                self.init_optimizer(tune=True)
            self.tune_loop(dataset, hooks)

        # завернуть в finally?
        if self.cfg.ddp:
            torch.distributed.destroy_process_group()

            
    def main_loop (self, dataset, hooks):
        train_loader, valid_loader = self.make_loaders(dataset)
        
        for ep in range(self.state.cur_epoch,
                        self.cfg.n_epochs):
            # как правильно организовать выход
            ## stop сейчас работает только в одном процессе
            if self.state.early_stopped:
                return True
            #!
            if os.path.exists(self.stop_file):
                os.remove(self.stop_file)
                fprint("Stopped on demand")
                return False

            train_loss, train_metrics = self.epoch_step(ep, train_loader)
            with torch.no_grad():
                valid_loss, valid_metrics = self.epoch_step(
                    ep, valid_loader, valid=True)            
                
            if np.isnan(train_loss) or np.isnan(valid_loss):
                fprint('Got nan and stopped')
                # проверить
                raise Exception(f'Training failed (got NaN)')

            if ep == 0 and self.is_main():
                self.format_dynamic_shapes()        
                fprint(f'Infering dynamic shapes:')
                fprint(f'Inputs:  {self.info.dynamic_shapes["in"]}')
                fprint(f'Outputs: {self.info.dynamic_shapes["out"]}')
            
            self.scheduler.step(valid_loss)
            status_msg = self.check_progress(valid_loss)
            
            if (hooks.get('post_epoch_hook') is not None
                and self.is_main()):
                hooks['post_epoch_hook'](self)
                torch.distributed.barrier(device_ids=[self.rank])


            self.state.cur_epoch = ep+1
            self.report_epoch(train_metrics, valid_metrics, status_msg)
                
            self.save_checkpoint(str(ep))
        return True

    def tune_loop (self, dataset, hooks):
        train_loader, valid_loader = self.make_loaders(
            dataset, force_shuffle=True)

        for ep in range(self.state.tune_steps_counter,
                        self.cfg.tune_steps):
            
            time.sleep(1) # зачем?
            #!
            if os.path.exists(self.stop_file):
                os.remove(self.stop_file)
                fprint("Stopped on demand")
                break

            train_loss, train_metrics = self.epoch_step(ep, train_loader)
            with torch.no_grad():
                valid_loss, valid_metrics = self.epoch_step(
                    ep, valid_loader, valid=True)            
                
            if np.isnan(train_loss) or np.isnan(valid_loss):
                fprint('Got nan and stopped')
                # проверить
                raise Exception(f'Training failed (got NaN)')
            
            status_msg = f"Tune step {ep+1}/{self.cfg.tune_steps}"

            if (hooks.get('post_epoch_hook') is not None
                and self.is_main()):
                hooks['post_epoch_hook'](self)
                torch.distributed.barrier(device_ids=[self.rank])

            self.state.cur_epoch += 1
            self.report_epoch(train_metrics, valid_metrics, status_msg)

            self.state.tune_steps_counter += 1
            self.save_best()
            self.save_checkpoint('t'+str(ep))
        

    def collect_sample_shapes (self, sample, prefix):
        # TODO: проверять полное соответствие ключей в collect_shapes
        shapes = utils.get_sample_shapes(sample, prefix)
        if self.state.shapes.get(prefix) is None:
            self.state.shapes[prefix] = {
                k:set() for k in shapes.keys()}

        for k,v in shapes.items():
            self.state.shapes[prefix][k].add(v)

    def collect_batch_shapes (self, ep, batch):
        if ep == 0 and self.is_main():
            inpt, tgt = batch[0], batch[-1]
            self.collect_sample_shapes(inpt, 'in')
            self.collect_sample_shapes(tgt, 'out')

    def format_dynamic_shapes (self):
        def infer_dyn (shapes):
            dyn = {}
            for name,shapes_set in shapes.items():
                v = np.array(list(shapes_set))
                axis_diff_shapes = ( (v==v[0]).sum(axis=0)!=len(v) )
                axis_diff_shapes[0] = True # always write zero axis (batch)
                dyn[name] = np.where(axis_diff_shapes)[0].tolist()
            return dyn
        
        self.info.dynamic_shapes = {
            'in'  : infer_dyn(self.state.shapes['in']),
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
        if valid:
            self.net.eval()
        else:
            self.net.train()
            
        total_loss, sample_counter = 0, 0
        epoch_metrics = {}

        n_repeat = (self.cfg.repeat_valid
                    if valid
                    else self.cfg.repeat_train)
        if self.cfg.recurrent:
            hidden_state = self.net.init_hidden().to(self.device)

        for _ in range(n_repeat):
            t = utils.Progress(loader,
                               (self.cfg.verbose and self.is_main()))
            for batch in t:
                self.collect_batch_shapes(ep, batch)
                inpt,w,y = self.convert_train_batch(batch, self.device)
                
                if self.cfg.recurrent:
                    raise NotImplementedError()
                    self.set_sample_input(*inpt, hidden_state)
                    yp, hidden_state = self.net(*inpt, hidden_state)
                    hidden_state = hidden_state.detach()
                else:
                    self.set_sample_input(inpt)
                    yp = utils.forward(self.net, inpt)

                loss, metrics = self.criterion(
                    yp, y, weights=w)

                if not valid:
                    if self.cfg.clip_grad:
                        torch.nn.utils.clip_grad_norm_(
                            self.net.parameters(), max_norm=1.0)
                    loss.backward()
                    self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

                loss_value = float(loss.detach())
                total_loss += (loss_value*len(y))
                for name,value in metrics.items():
                    if epoch_metrics.get(name) is None:
                        epoch_metrics[name] = 0
                    epoch_metrics[name] += value*len(y)

                t.update(utils.round(loss_value, 4))
                sample_counter += len(y)

        total_loss /= sample_counter
        for name,value in epoch_metrics.items():
            epoch_metrics[name] = (epoch_metrics[name] / sample_counter)

        if self.cfg.ddp:
            # loss and metrics are syncronized here,
            # so early stopping and etc are same in all processes
            total_loss, epoch_metrics = self.ddp_sync_metrics(
                total_loss, epoch_metrics)
            
        return total_loss, epoch_metrics

    def ddp_sync_metrics (self, loss, metrics):
        # IDEA: create a general sync&average function
        loss_all = [None for _ in range(self.world_size)]
        metrics_all = [None for _ in range(self.world_size)]
        torch.cuda.set_device(self.device) # workaround for a bug/feature in pytorch
        torch.distributed.all_gather_object(loss_all, loss)
        torch.distributed.all_gather_object(metrics_all, metrics)
        avg_loss = np.mean(loss_all)
        for k in metrics:
            metrics[k] = np.mean([m[k] for m in metrics_all])
        return avg_loss, metrics
        
    def report_epoch (self, train_metrics, valid_metrics, status_msg):
        if not self.is_main():
            return
        
        elapsed_time = int(time.time() - self.state.start_moment)
        ep = self.state.cur_epoch
        train_msg, valid_msg = [],[]
        for name,value in train_metrics.items():
            suffix = 'train'
            self.log_writer.add_scalar(
                f'{name}/{suffix}',
                value, ep)
            round_val = utils.round(value, 3, significant=True, with_zeros=True)
            train_msg.append(f"{name}= {round_val}")
        for name,value in valid_metrics.items():
            suffix = 'valid'
            self.log_writer.add_scalar(
                f'{name}/{suffix}',
                value, ep)
            round_val = utils.round(value, 3, significant=True, with_zeros=True)
            valid_msg.append(f"{name}= {round_val}")

        train_msg = ", ".join(train_msg)
        valid_msg = ", ".join(valid_msg)
        
        log_msg = (f"{self.name} /"
                   f" Ep= {self.state.cur_epoch} /"
                   f" Trn: {train_msg} /"
                   f" Val: {valid_msg} /"
                   f" {self.device} / {elapsed_time}s /"
                   f" LR= {self.get_rate():.5f} / {status_msg}")
        fprint(log_msg)
        self.state.log.append(log_msg)
        with open(os.path.join(self.workdir,
                               'train_log.txt'),
                  'a') as fp:
            fp.write(log_msg+'\n')

    # убрать итерацию по всем param_groups
    def get_rate (self):
        return [p['lr'] for p
                in self.optimizer.param_groups][0]

        
    def init_workdir (self, root):
        # оптимизировать workdir под ddp
        if self.is_main():
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
            if self.is_main():
                os.mkdir(workdir)
        elif os.path.exists(self.checkpoint_file):
            self.load_checkpoint()
            for log_msg in self.state.log:
                if self.is_main():
                    fprint(log_msg)
            if self.state.early_stopped or \
               self.state.cur_epoch+1 >= self.cfg.n_epochs:
                if self.is_main():
                    fprint('Checkpoint is already finalized')
                return
            else:
                log_msg = 'Resuming training from latest checkpoint'
                if self.is_main():
                    fprint(log_msg)
                self.state.log.append(log_msg)
                if self.is_main():
                    with open(os.path.join(self.workdir,
                                           'train_log.txt'),
                              'a') as fp:
                        fp.write(log_msg+'\n')

        else:
            if self.is_main():
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
            
    def make_loaders (self, dataset, force_shuffle=False):
        train_split = int(len(dataset)*self.cfg.train_prop)

        if (not self.cfg.recurrent and
            (self.cfg.shuffle or force_shuffle)):
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

        ###
        if self.cfg.ddp:
            if self.cfg.batch_size % self.world_size != 0:
                raise Exception("Requested batch size ({self.cfg.batch_size}) is not dividable into equal parts by number of gpus ({world_size})")
            batch_size = self.cfg.batch_size//self.world_size
            self.train_sampler = torch.utils.data.distributed.DistributedSampler(
                train_ds,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=loader_shuffle)
            self.valid_sampler = torch.utils.data.distributed.DistributedSampler(
                valid_ds,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=False)
            train_sampler_args = {'shuffle' : False,
                                  'sampler' : self.train_sampler}
            valid_sampler_args = {'shuffle' : False,
                                  'sampler' : self.valid_sampler}
        else:
            batch_size = self.cfg.batch_size
            train_sampler_args = {'shuffle' : loader_shuffle}
            valid_sampler_args = {'shuffle' : False}

        # check pin_memory=True (maybe it conflicts with multiple workers)
        train_loader = torch.utils.data.DataLoader(
            train_ds,
            batch_size = batch_size,
            num_workers = self.cfg.n_workers,
            persistent_workers=True if self.cfg.n_workers > 0 else False,
            pin_memory = False,
            prefetch_factor=1 if self.cfg.n_workers > 0 else None,
            collate_fn = collate_fn or self.collator,
            **train_sampler_args)

        valid_loader = torch.utils.data.DataLoader(
            valid_ds,
            batch_size = batch_size,
            num_workers = self.cfg.n_workers,
            pin_memory = False,
            prefetch_factor=1 if self.cfg.n_workers > 0 else None,
            persistent_workers=True if self.cfg.n_workers > 0 else False,
            collate_fn = collate_fn or self.collator,
            **valid_sampler_args)

        return train_loader, valid_loader

    def init_info (self):
        self.info = SimpleNamespace()
        self.info.version = __version__
        self.info.dynamic_shapes = {}

    def init_state (self):
        self.state = SimpleNamespace()
        # self.state.success = False - не используется?
        self.state.early_stop_counter = 0
        self.state.cur_epoch = 0
        self.state.best_value = np.nan
        self.state.early_stopped = False
        self.state.restart_done = False
        self.state.log = []
        self.state.shapes = dict()
        self.state.tune_steps_counter = 0

    def init_optimizer (self, tune=False):
        # вставить обертку ZeroRedundancy
        opt_cfg = deepcopy(self.cfg.optimizer.__dict__)
        opt_name = opt_cfg.pop('name')
        opt = torch.optim.__getattribute__(opt_name)
        if tune:
            opt_cfg['lr'] /= 10
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
        if (self.is_main()
            and not hasattr(self, 'sample_input')):
            self.sample_input = utils.convert_sample(
                sample_input, 'numpy')                
            
    def save_checkpoint (self, ep):
        if not self.is_main():
            return
        
        ckpt = {'format' : 'torch',
                'net' : self.get_net_module().state_dict(),
                'optimizer' : self.optimizer.state_dict(),
                'scheduler' : self.scheduler.state_dict(),
                'criterion' : self.criterion.state_dict(),
                'state' : self.state,
                'info' : self.info,
                'cfg' : self.cfg,
                'sample_input' : self.sample_input}
        serialize.pack(ckpt, self.checkpoint_file)
        if self.cfg.save_each_epoch:
            epoch_savefile = (os.path.splitext(self.checkpoint_file)[0]
                              + str(ep) + '.bin')
            serialize.pack(ckpt, epoch_savefile)

        
    def save_best (self):
        if not self.is_main():
            return
        
        ckpt = {'format' : 'torch',
                'net' : self.get_net_module().state_dict(),
                'criterion' : self.criterion.state_dict(),
                'cfg' : self.cfg,
                'info' : self.info,
                'sample_input' : self.sample_input}
        serialize.pack(ckpt, self.best_file)

    def load_checkpoint (self):
        ckpt = serialize.unpack(self.checkpoint_file)            
        # TODO: compare given cfg with loaded, print the diff            
        self.state = ckpt['state']
        self.info = ckpt['info']
        self.get_net_module().load_state_dict(ckpt['net'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.scheduler.load_state_dict(ckpt['scheduler'])
        self.criterion.load_state_dict(ckpt['criterion'])
        del ckpt

    def load_net_from_checkpoint (self, checkpoint_file):
        ckpt = serialize.unpack(checkpoint_file)
        #self.get_net_module().load_state_dict(
        #        ckpt['net'],
        #        strict=False)
        load_partial_state_dict(
            self.get_net_module(),
            ckpt['net'])
                                

    def get_net_module (self):
        if (self.cfg.dataparallel
            or self.cfg.ddp):
            return self.net.module
        else:
            return self.net

def load_partial_state_dict (net, ckpt_state_dict):
    net_state_dict = net.state_dict()
    filtered_state_dict = {
        k: v for k, v in ckpt_state_dict.items() 
        if (k in net_state_dict
            and v.size() == net_state_dict[k].size())}

    net.load_state_dict(filtered_state_dict, strict=False)
