import os
import logging
import datetime
from pickle import PicklingError
from typing import Optional, Dict, Tuple, Union, List, Any, Sequence
from sklearn.cluster import KMeans
from sklearn.metrics import v_measure_score
from tqdm import tqdm
import numpy as np
import zipfile
import pprint
from collections import deque
from textwrap import dedent
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("agg")

import inspect
import IPython
import torch
import torch.nn
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
from torch.utils import collect_env
import torch_geometric
import elektronn3
from elektronn3.training.trainer import _change_log_file_to, NaNException
from elektronn3.training.train_utils import HistoryTracker, Timer, pretty_string_time

logger = logging.getLogger('elektronn3log')


class GNNTrainer:

    tb: SummaryWriter
    terminate: bool
    step: int
    epoch: int
    train_loader: torch.utils.data.DataLoader
    valid_loader: torch.utils.data.DataLoader
    exp_name: str
    save_path: str  # Full path to where training files are stored
    out_channels: Optional[int]  # Number of channels of the network outputs

    def __init__(
        self, 
        model: torch.nn.Module,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        save_root: str,
        data: Tuple[torch_geometric.data.Data],
        valid_metrics: Optional[Dict] = None,
        extra_save_steps: Sequence[int] = (),
        batch_size: int = 1,
        exp_name: Optional[str] = None,
        schedulers: Optional[Dict[Any, Any]] = None,
        early_stopping: Optional[Any] = None,
        num_workers: int = 0,
        enable_tensorboard: bool = True,
        tensorboard_root_path: Optional[str] = None,
        ignore_errors: bool = False,
        ipython_shell: bool = False,
        out_channels: Optional[int] = None,
        seed: Optional[int] = 0,
        mixed_precision: bool = False,
        tqdm_kwargs: Optional[Dict] = None,
    ):
        model.to(device)
        if isinstance(criterion, torch.nn.Module):
            criterion.to(device)

        self.device = device
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.save_root = os.path.expanduser(save_root)
        self.data = data
        self.extra_save_steps = extra_save_steps
        self.batch_size = batch_size
        self.schedulers = schedulers
        self.early_stopping = early_stopping
        self.num_workers = num_workers
        self.ignore_errors = ignore_errors
        self.ipython_shell = ipython_shell
        self.out_channels = out_channels
        self.seed = seed
        self.mixed_precision = mixed_precision
        self.tqdm_kwargs = {} if tqdm_kwargs is None else tqdm_kwargs
        
        self._shell_info = dedent("""
            Entering IPython training shell. To continue, hit Ctrl-D twice.
            To terminate, set self.terminate = True and then hit Ctrl-D twice.
        """).strip()

        lr = self.optimizer.state_dict()["param_groups"][0]["lr"]

        logger.info(f"Training configuration: lr={lr}, criterion={self.criterion.__class__.__name__}, optimizer={self.optimizer.__class__.__name__}, model={self.model.__class__.__name__}, early_stopping={True if self.early_stopping else False}")

        self._tracker = HistoryTracker()
        self._timer = Timer()

        if exp_name is None:
            timestamp = datetime.datetime.now().strftime('%y-%m-%d_%H-%M-%S')
            exp_name = model.__class__.__name__ + '__' + timestamp

        self.exp_name = exp_name
        self.save_path = os.path.join(save_root, self.exp_name)
        if os.path.isdir(self.save_path):
            raise RuntimeError(f"Save path {self.save_path} already exists")

        os.makedirs(self.save_path)
        _change_log_file_to(f'{self.save_path}/elektronn3.log')
        logger.info(f"Writing files to save_path {self.save_path}/\n")

        # Terminate training by setting to true
        self.terminate = False
        self.epoch = 0
        self.patience = self.early_stopping.patience if self.early_stopping else 0
        self._lr_nhood = deque(maxlen=3)
        self.out_channels = out_channels
        self.validate_every_epoch = 1
        self.tb = None

        if enable_tensorboard:
            if tensorboard_root_path is None:
                tb_path = self.save_path
            else:
                tensorboard_root_path = os.path.expanduser(tensorboard_root_path)
                tb_path = os.path.join(tensorboard_root_path, self.exp_name)
                os.makedirs(tb_path, exist_ok=True)
            self.tb = SummaryWriter(tb_path, flush_secs=20)

        self.best_val_loss = np.inf
        self.valid_metrics = {} if valid_metrics is None else valid_metrics
        self.validate = False

    def run(self, max_epochs: int=1, max_runtime=3600*24*7) -> None:
        self.start_time = datetime.datetime.now()
        self.end_time = self.start_time + datetime.timedelta(seconds=max_runtime)
        # Save initial model
        self._save_model(suffix="_initial", verbose=False)
        self._lr_nhood.clear()
        self._lr_nhood.append(self.optimizer.param_groups[0]['lr'])  # LR of the first training step

        stats: Dict[str, Union[float, List[float]]] = {}
        misc: Dict[str, Union[float, List[float]]] = {}
        
        while not self.terminate:
            try:
                timer = Timer()
                
                # Training loss
                dloss = self._train()
                loss = float(dloss.item())

                stats['tr_loss'] = loss
                misc["tr_speed"] = self.data.num_nodes / timer.t_passed
                misc['learning_rate'] = self.optimizer.param_groups[0]['lr']

                self._tracker.update_timeline([self._timer.t_passed, loss])
                
                # Increment epochs
                self._incr_epoch(max_epochs, max_runtime)

                valid_stats = self._validate()
                stats.update(valid_stats)

                self.schedulers['lr'].step(stats['val_loss'])

                # Log to stdout and text log file
                self._log_basic(stats, misc)
                # Render visualizations and log to tensorboard
                self._log_to_tensorboard(stats, misc)

                # Save current model
                self._save_model(val_loss=stats['val_loss'], verbose=False)

                # Save best model
                if stats["val_loss"] < self.best_val_loss:
                    self.best_val_loss = stats["val_loss"]
                    self._save_model(suffix="_best", val_loss=stats["val_loss"])

                if self.early_stopping is not None:
                    self.early_stopping(stats["val_loss"])
                    if self.early_stopping.stop:
                        logger.info(f"Validation loss did not improve for {self.patience} epochs. Terminating...")
                        self.terminate = True 

            except KeyboardInterrupt: 
                if self.ipython_shell:
                    IPython.embed(header=self._shell_info)
                else:
                    break
                if self.terminate:
                    break

            except Exception as e:
                logger.exception("Unhandled exception during training:")
                if self.ignore_errors:
                    pass
                elif self.ipython_shell:
                    print("\nEntering Command line such that Exception can be "
                          "further inspected by user.\n\n")
                    IPython.embed(header=self._shell_info)
                    if self.terminate:
                        break
                else:
                    raise e

        # Save final model
        self._save_model(suffix="_final")
        if self.tb is not None:
            self.tb.close()  # Ensure that everything is flushed

    def _train(self) -> None:
        self.model.train()
        self.optimizer.zero_grad()
        
        x, edge_index, edge_attr, y = self.data.x, self.data.edge_index, self.data.edge_attr, self.data.y
        mask = self.data.train_mask

        if self.model.__class__.__name__ == 'MLP':
            dout = self.model(x[mask])
            dloss = self.criterion(dout, y[mask])
        else:
            attr = edge_attr[:,2]+edge_attr[:,9]
            attr = attr - attr.min()
            min_value = attr.min()
            max_value = attr.max()
            attr.div_(max_value - min_value)
            if self.model.__class__.__name__ == 'GCN':
                dout = self.model(x, edge_index, attr)
            else:
                dout = self.model(x, edge_index, edge_attr[:,[2,9]])

            dloss = self.criterion(dout[mask], y[mask])
        
        if torch.isnan(dloss):
            logger.error(f"NaN loss encountered at step {self.epoch}. Aborting...")
            raise NaNException()
        
        dloss.backward()
        self.optimizer.step()
        
        return dloss

    @torch.no_grad()
    def _validate(self) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
        self.model.eval()
        stats = {name: [] for name in self.valid_metrics.keys()}

        # Validation loss
        x, edge_index, edge_attr, y = self.data.x, self.data.edge_index, self.data.edge_attr, self.data.y
        mask = self.data.val_mask

        if self.model.__class__.__name__ == 'MLP':
            dout = self.model(x[mask])
            dloss = self.criterion(dout, y[mask])
            out = dout.detach().cpu().numpy()
        else:
            attr = edge_attr[:,2]+edge_attr[:,9]
            attr = attr - attr.min()
            min_value = attr.min()
            max_value = attr.max()
            attr.div_(max_value - min_value)
            if self.model.__class__.__name__ == 'GCN':
                dout = self.model(x, edge_index, attr)
            else:
                dout = self.model(x, edge_index, edge_attr[:,[2,9]])

            dloss = self.criterion(dout[mask], y[mask])
            out = dout[mask].detach().cpu().numpy()
        
        stats['val_loss'] = float(dloss.item())
        target = y[mask].detach().cpu().numpy()
    
        if self.epoch % self.validate_every_epoch == 0:
            
            for name, evaluator in self.valid_metrics.items():
                stats[name] = evaluator(target, out)

        self.model.train()

        return stats

    def _incr_epoch(self, max_epochs, max_runtime):
        self.epoch += 1
        
        if self.epoch >= max_epochs:
            logger.info(f"max_epochs ({max_epochs}) exceeded. Terminating...")
            self.terminate = True

        if datetime.datetime.now() >= self.end_time:
            logger.info(f"max_runtime ({max_runtime} seconds) exceeded. Terminating...")
            self.terminate = True

    def _scheduler_step(self, loss):
        """Update schedules"""
        for sched in self.schedulers.values():
            # support ReduceLROnPlateau; doc. uses validation loss instead
            # http://pytorch.org/docs/master/optim.html#torch.optim.lr_scheduler.ReduceLROnPlateau
            if 'metrics' in inspect.signature(sched.step).parameters:
                sched.step(metrics=loss)
            else:
                sched.step()
        # Append LR of the next iteration (after sched.step()) for local LR minima detection
        self._lr_nhood.append(self.optimizer.param_groups[0]['lr'])

    def _save_model(
        self, 
        suffix: str="",
        verbose: bool = True,
        val_loss=np.nan
    ) -> None:
        log = logger.info if verbose else logger.debug

        model = self.model
        model_trainmode = model.training

        state_dict_path = os.path.join(self.save_path, f"state_dict{suffix}.pth")
        model_path = os.path.join(self.save_path, f"model{suffix}.pt")

        info = {
            'epoch': self.epoch,
            'best_val_loss': self.best_val_loss,
            'val_loss': val_loss,
            'elektronn3.__version__': elektronn3.__version__,
            'env_info': collect_env.get_pretty_env_info() 
        }

        info = {k: str(v) for k, v in info.items()}

        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'info': info
        }, state_dict_path)

        log(f'Saved state_dict as {state_dict_path}')
        pts_model_path = f"{model_path}"

        try:
            torch.save(model, model_path)
            log(f'Saved model as {model_path}')
        
        except (TypeError, PicklingError) as exc:
            raise exc

        finally:
            model.training = model_trainmode
        
        if os.path.isfile(pts_model_path):
            with zipfile.ZipFile(pts_model_path, 'a', compression=zipfile.ZIP_DEFLATED) as zfile:
                infostr = pprint.pformat(info, indent=2, width=120)
                zfile.writestr('info.txt', infostr)

    def _log_basic(self, stats, misc):
        tr_loss = stats['tr_loss']
        val_loss = stats['val_loss']
        lr = misc['learning_rate']
        tr_speed = misc['tr_speed']
        t = pretty_string_time(self._timer.t_passed)
        text = f'epoch={self.epoch:04d}, tr_loss={tr_loss:.3f}, val_loss={val_loss:.3f}, '
        text += f'lr={lr:.2e}, tr_speed={tr_speed:.2f} it/s, {t}'
        logger.info(text)

    def _log_to_tensorboard(
        self,
        stats: Dict,
        misc: Dict,
        file_stats: Optional[Dict] = None
    ) -> None:
        if self.tb:
            try:
                self._tb_log_scalars(stats, "stats")
                self._tb_log_scalars(misc, "misc")
                if file_stats is not None:
                    self._tb_log_scalars(file_stats, "file_stats")
                self._tb_log_histograms()
            except Exception as exc:
                logger.exception('Error occured while logging to tensorboard:')
                raise exc

    def _tb_log_scalars(
            self,
            scalars: Dict[str, float],
            tag: str = 'default'
    ) -> None:
        for key, value in scalars.items():
            if isinstance(value, (list, tuple, np.ndarray)):
                for i in range(len(value)):
                    if not np.isnan(value[i]):
                        self.tb.add_scalar(f'{tag}/{key}', value[i], self.epoch - len(value) + i)
            elif not np.isnan(value):
                self.tb.add_scalar(f'{tag}/{key}', value, self.epoch)

    def _tb_log_histograms(self) -> None:
        """Log histograms of model parameters and their current gradients.
        Make sure to run this between ``backward()`` and ``zero_grad()``,
        because otherwise gradient histograms will only consist of zeros.
        """
        for name, param in self.model.named_parameters():
            self.tb.add_histogram(f'param/{name}', param, self.epoch)
            grad = param.grad if param.grad is not None else torch.tensor(0)
            self.tb.add_histogram(f'grad/{name}', grad, self.epoch)