import os
import logging
import datetime
from pickle import PicklingError
from typing import Optional, Dict, Tuple, Union, List, Any, Sequence
from tqdm import tqdm
import numpy as np
import zipfile
import pprint
import matplotlib
import matplotlib.pyplot as plt
from collections import deque
from textwrap import dedent
matplotlib.use("agg")

import inspect
import IPython
import torch
import torch.nn
import torch.nn.functional as F
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
from torch.utils import collect_env
from torch.cuda import amp
import torch_geometric
from torch_geometric.loader import NeighborLoader, ImbalancedSampler
from torch_geometric.data import Data
from torch_geometric.utils import degree, dropout_adj
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.linear_model import SGDClassifier
from sklearn.metrics.cluster import homogeneity_completeness_v_measure
# import tensorboardX

import elektronn3
from elektronn3.training.trainer import _change_log_file_to, NaNException
from elektronn3.training.train_utils import HistoryTracker, Timer, pretty_string_time


logger = logging.getLogger('elektronn3log')

def visualize_embeddings(x, color, epoch):
    mapping = {0: "MSN", 1: "LMAN", 2: "HVC"}

    fig, ax = plt.subplots(figsize=(10, 10))
    scatter = ax.scatter(x[:, 0], x[:, 1], c=color, s=15)
    handles, labels = scatter.legend_elements()
    labels = [mapping[int(l.split("{")[1][0])] for l in labels]
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(handles, labels, loc="best", title="Cell types")
    ax.set_title(f"Node embeddings (t-SNE), Epoch {epoch}")

    # convert figure to numpy array
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    return data.reshape(3, data.shape[0], data.shape[1])  # (3, height, width)


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
        inductive: bool,
        data: torch_geometric.data.Data,
        num_neighbors: Optional[List[int]] = [25, 10],
        edge_attr_ix: Optional[Tuple[int]] = None,
        valid_metrics: Optional[Dict] = None,
        extra_save_steps: Sequence[int] = (),
        batch_size: int = 1,
        exp_name: Optional[str] = None,
        schedulers: Optional[Dict[Any, Any]] = None,
        early_stopping: Optional[Any] = None,
        self_supervised: Optional[bool] = False,
        embed_loss: Optional[bool] = False,
        negative_sampling: Optional[bool] = False,
        variational: Optional[bool] = False,
        num_workers: int = 0,
        enable_tensorboard: bool = True,
        tensorboard_root_path: Optional[str] = None,
        ignore_errors: bool = False,
        ipython_shell: bool = False,
        out_channels: Optional[int] = None,
        seed: Optional[int] = 0,
        debug_mode: bool = False,
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
        self.extra_save_steps = extra_save_steps
        self.save_root = os.path.expanduser(save_root)
        self.inductive = inductive
        self.data = data
        self.num_neighbors = num_neighbors
        self.edge_attr_ix = edge_attr_ix
        self.batch_size = batch_size
        self.schedulers = schedulers
        self.early_stopping = early_stopping
        self.self_supervised = self_supervised
        self.embed_loss = embed_loss
        self.negative_sampling = negative_sampling
        self.variational = variational
        self.num_workers = num_workers
        self.ignore_errors = ignore_errors
        self.ipython_shell = ipython_shell
        self.out_channels = out_channels
        self.seed = seed
        self.debug_mode = debug_mode
        self.mixed_precision = mixed_precision
        self.tqdm_kwargs = {} if tqdm_kwargs is None else tqdm_kwargs

        self._shell_info = dedent("""
            Entering IPython training shell. To continue, hit Ctrl-D twice.
            To terminate, set self.terminate = True and then hit Ctrl-D twice.
        """).strip()
        
        lr = self.optimizer.state_dict()["param_groups"][0]["lr"]
        logger.info(f"Training configuration: lr={lr}, batchsize={batch_size}, criterion={self.criterion.__class__.__name__}, optimizer={self.optimizer.__class__.__name__}, model={self.model.__class__.__name__}, num_neighbors={num_neighbors}, early_stopping={True if self.early_stopping else False}")

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
        self.step = 0
        self.epoch = 0
        self.patience = self.early_stopping.patience if self.early_stopping else 0
        self._lr_nhood = deque(maxlen=3)
        self.validate_every_epoch = 5

        if self.debug_mode:
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

        if self.debug_mode:
            # select a small subset of data for debugging purposes
            mask = torch.rand(data.num_nodes) > 0.9
            data = data.subgraph(mask)
            self.validate_every_epoch = 1  # validate in the first epoch

        if inductive:
            logger.info("Using inductive training")
            train_data = data.subgraph(data.train_mask)
            val_data = data.subgraph(data.val_mask)
            # test_data = data.subgraph(data.test_mask)

            self.train_loader = NeighborLoader(
                train_data,
                num_neighbors=self.num_neighbors,
                shuffle=True,
                batch_size=batch_size,
            )
            
            self.valid_loader = NeighborLoader(
                val_data,
                num_neighbors=self.num_neighbors,
                shuffle=False,
                batch_size=batch_size
            )
        
        else: 
            logger.info("Using transductive training")
            sampler = ImbalancedSampler(data, input_nodes=data.train_mask)
            self.train_loader = NeighborLoader(
                data,
                num_neighbors=self.num_neighbors,
                input_nodes=data.train_mask,
                # shuffle=True,
                batch_size=batch_size,
                sampler=sampler
            )

            self.valid_loader = NeighborLoader(
                data,
                num_neighbors=self.num_neighbors,
                input_nodes=data.val_mask,
                shuffle=False,
                batch_size=batch_size
            )

        self.best_val_loss = np.inf
        self.best_val_acc = -np.inf
        self.valid_metrics = {} if valid_metrics is None else valid_metrics
        self.validate = False
            
    def run(self, max_steps: int=1, max_runtime=3600*24*7) -> None:
        self.start_time = datetime.datetime.now()
        self.end_time = self.start_time + datetime.timedelta(seconds=max_runtime)
        # Save initial model
        self._save_model(suffix="_initial", verbose=False)
        self._lr_nhood.clear()
        self._lr_nhood.append(self.optimizer.param_groups[0]['lr'])  # LR of the first training step
        
        while not self.terminate:
            try:
                if self.self_supervised:
                    self.km = MiniBatchKMeans(n_clusters=self.out_channels)
                    self.lr = SGDClassifier(loss='log')

                stats, misc = self._train(max_steps, max_runtime)
                self.epoch += 1

                valid_stats = self._validate()
                stats.update(valid_stats)

                self.schedulers['lr'].step(np.mean(stats['val_loss']))

                # Log to stdout and text log file
                self._log_basic(stats, misc)
                # Render visualizations and log to tensorboard
                self._log_to_tensorboard(stats, misc, None, None)

                # Save trained model state
                self._save_model(val_loss=stats['val_loss'], verbose=False) 
                # self._save_model(val_loss=stats['val_accuracy_mean'], verbose=False)  # Not verbose because it can get spammy.

                if stats['val_loss'] < self.best_val_loss:
                    self.best_val_loss = stats['val_loss']
                    self._save_model(suffix='_best', val_loss=stats['val_loss'])

                # if stats['val_accuracy_mean'] > self.best_val_acc:
                #     self.best_val_acc = stats['val_accuracy_mean']
                #     self._save_model(suffix='_best', val_loss=stats['val_accuracy_mean'])

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

    def get_edge_attributes(self, edge_attr):
        # no edge attributes
        if self.edge_attr_ix is None:
            ea = None 
        # edge weight for GCN
        elif self.edge_attr_ix == -1: 
            ea = edge_attr[:,2] + edge_attr[:,9]
        # indexed pre-post edge attributes (e.g #incoming-#outgoing conns, incoming-outgoing area)
        else:
            ea = edge_attr[:,self.edge_attr_ix]
            assert ea.ndim == 2
            # min_value = ea.min(dim=1)[0]
            # max_value = ea.max(dim=1)[0]
            # ea -= min_value.view(-1,1)
            # diff = (max_value-min_value).view(-1, 1)
            # ea.div_(diff)
            # ea.masked_fill_(ea == float('inf'), 0)

        return ea

    def _train_step(self, batch: torch_geometric.data.Data) -> Tuple[torch.Tensor, torch.Tensor]:
        """Core training step on self.device

        Args:
            batch (torch_geometric.data.Data): batch of data

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 
        """
        dbatch = batch.to(self.device)
        x, y = dbatch.x[:,:30], dbatch.y
        edge_index, edge_attr = dbatch.edge_index, dbatch.edge_attr
        batch_size = dbatch.batch_size
        ea = self.get_edge_attributes(edge_attr)
        
        with amp.autocast(enabled=self.mixed_precision):
            self.optimizer.zero_grad()
            # self-supervised
            if self.self_supervised:
                dloss = 0
                if self.model.encoder.__class__.__name__ == 'Linear':
                    dz = self.model.encode(x[:batch_size], edge_index)
                else:
                    dz = self.model.encode(x, edge_index, ea)[:batch_size]

                if self.embed_loss:
                    edge_index_drop, ea_drop = dropout_adj(edge_index, ea, p=0.5, force_undirected=True)
                    pos_z = self.model.encode(x, edge_index_drop, ea_drop)[:batch_size]

                    neg_ix = torch.randint(0, batch_size, size=(x.size(0),), dtype=torch.long, device=self.device)
                    neg_z = self.model.encode(x[neg_ix], edge_index, ea)[:batch_size]

                    dloss += self.model.contrastive_loss(dz, pos_z, neg_z)

                dloss += self.model.recon_loss(dz, dbatch)
                        
                z = dz.detach().cpu().numpy()
                target = y[:batch_size].detach().cpu().numpy()

                self.lr.partial_fit(z, target, classes=np.arange(self.out_channels))
                self.km.partial_fit(z)

            # semi-supervised (transductive) / supervised (inductive)
            else:  
                if self.model.__class__.__name__ == 'MLP':
                    dout = self.model(x[:batch_size])
                else:
                    ea = self.get_edge_attributes(edge_attr)
                    dout = self.model(x, edge_index, ea)[:batch_size]

                dloss = self.criterion(dout, y[:batch_size])

            dloss.backward()
            self.optimizer.step()

        if torch.isnan(dloss):
            logger.error('NaN loss detected! Aborting training.')
            raise NaNException

        return dloss

    def _train(self, max_steps, max_runtime):
        """Train for one epoch or until max_steps or max_runtime is reached"""
        self.model.train()

        # Scalar training stats that should be logged and written to tensorboard later
        stats: Dict[str, Union[float, List[float]]] = {stat: [] for stat in ['tr_loss']}
        # Other scalars to be logged
        misc: Dict[str, Union[float, List[float]]] = {misc: [] for misc in ['mean_target']}

        running_vx_size = 0  # Counts input sizes (number of pixels/voxels) of training batches

        timer = Timer()
        batch_iter = tqdm(
            enumerate(self.train_loader),
            'Training',
            total=len(self.train_loader),
            dynamic_ncols=True,
            **self.tqdm_kwargs
        )
        
        for i, batch in batch_iter:
            if self.step in self.extra_save_steps:
                self._save_model(f'_step{self.step}', verbose=False)

            dloss = self._train_step(batch)
            batch_size = batch.batch_size

            with torch.no_grad():
                loss = float(dloss)
                target = batch.y[:batch_size] if not self.self_supervised else batch.x[:batch_size]
                mean_target = float(target.to(torch.float32).mean()) if target is not None else 0.
                # misc['mean_target'].append(mean_target)
                stats['tr_loss'].append(loss)
                batch_iter.set_description(f'Training (loss {loss:.4f})')
                self._tracker.update_timeline([self._timer.t_passed, loss, mean_target])

             # Not using .get_lr()[-1] because ReduceLROnPlateau does not implement get_lr()
            misc['learning_rate'] = self.optimizer.param_groups[0]['lr']  # LR for the this iteration

            running_vx_size += batch.x[:batch_size].numel()

            self._incr_step(max_runtime, max_steps)

            if self.terminate:
                break

        stats['tr_loss_std'] = np.std(stats['tr_loss'])
        misc['tr_speed'] = len(self.train_loader) / timer.t_passed
        misc['tr_speed_vx'] = running_vx_size / timer.t_passed / 1e6  # MVx

        return stats, misc

    @torch.no_grad()
    def _validate(self) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
        self.model.eval()  # Set dropout and batchnorm to eval mode

        val_loss = []
        lr_outs = []
        km_outs = []
        targets = []
        embeddings = []
        stats = {name: [] for name in self.valid_metrics.keys()}

        batch_iter = tqdm(
            enumerate(self.valid_loader),
            'Validating',
            total=len(self.valid_loader),
            dynamic_ncols=True,
            **self.tqdm_kwargs
        )

        for i, batch in batch_iter:
            # Everything with a "d" prefix refers to tensors on self.device (i.e. probably on GPU)
            dbatch = batch.to(self.device)
            x, y = dbatch.x[:,:30], dbatch.y
            edge_index, edge_attr = dbatch.edge_index, dbatch.edge_attr
            batch_size = dbatch.batch_size
            ea = self.get_edge_attributes(edge_attr)

            with amp.autocast(enabled=self.mixed_precision):
                # self-supervised
                if self.self_supervised:
                    if self.model.encoder.__class__.__name__ == 'Linear':
                        dz = self.model.encode(x[:batch_size], edge_index)
                    else:
                        dz = self.model.encode(x, edge_index, ea)[:batch_size]
                        
                    dloss = self.model.recon_loss(dz, dbatch)
                    z = dz.detach().cpu().numpy()
                    target = y[:batch_size].detach().cpu().numpy()

                    val_loss.append(dloss.item())

                    if self.epoch % self.validate_every_epoch == 0:
                        self.validate = True
                        # predictions
                        pred = self.lr.predict_proba(z)
                        out = self.km.predict(z)
                        lr_outs.append(pred)
                        km_outs.append(out)
                        targets.append(target)
                        embeddings.append(z)

                # semi-supervised (transductive) / supervised (inductive)
                else:
                    if self.model.__class__.__name__ == 'MLP':
                        dout = self.model(x[:batch_size])
                    else:
                        ea = self.get_edge_attributes(edge_attr)
                        dout = self.model(x, edge_index, ea)[:batch_size]

                    val_loss.append(self.criterion(dout, y[:batch_size]).item())

                    out = dout.detach().cpu().numpy()
                    target = y[:batch_size].detach().cpu().numpy()

                    if self.epoch % self.validate_every_epoch == 0:
                        self.validate = True
                        lr_outs.append(out)
                        targets.append(target)

        stats['val_loss'] = np.mean(val_loss)
        stats['val_loss_std'] = np.std(val_loss)
       
        if self.validate:
            for name, evaluator in self.valid_metrics.items():
                if evaluator.__class__.__name__ == 'SilhouetteScore':
                    mvals = [evaluator(emb, out) for emb, out in zip(embeddings, km_outs)]
                elif evaluator.__class__.__name__ in ['Accuracy', 'Precision', 'Recall', 'DSC']:  # classification evaluation
                    mvals = [evaluator(target, out) for target, out in zip(targets, lr_outs)]
                else:  # clustering evaluation
                    mvals = [evaluator(target, out) for target, out in zip(targets, km_outs)]

                if np.all(np.isnan(mvals)):
                    stats[name] = np.nan
                else:
                    stats[name] = np.nanmean(mvals)

            self.validate = False

        self.model.train()

        return stats

    def _incr_step(self, max_runtime, max_steps):
        self.step += 1

        if self.step >= max_steps:
            logger.info(f"max_steps ({max_steps}) exceeded. Terminating...")
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
            'global_step': self.step,
            'epoch': self.epoch,
            'best_val_loss': self.best_val_loss,
            'val_loss': val_loss,
            'num_neighbors': self.num_neighbors,
            'elektronn3.__version__': elektronn3.__version__,
            'env_info': collect_env.get_pretty_env_info() 
        }

        info = {k: str(v) for k, v in info.items()}

        save_dict = {
            'model_state_dict': model.state_dict(),
            'info': info,
        }

        if self.variational:
            save_dict['optimizer_state_dict'] = {
                'encoder': self.encoder_optimizer.state_dict(),
                'discriminator': self.discriminator_optimizer.state_dict(),
            }

        else:
            save_dict['optimizer_state_dict'] = self.optimizer.state_dict()

        torch.save(save_dict, state_dict_path)

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
        """Log to stdout and text log file"""
        tr_loss = np.mean(stats['tr_loss'])
        val_loss = np.mean(stats['val_loss'])
        # val_acc = stats['val_accuracy_mean']
        # print(val_acc, type(val_acc))
        lr = misc['learning_rate']
        tr_speed = misc['tr_speed']
        tr_speed_vx = misc['tr_speed_vx']
        t = pretty_string_time(self._timer.t_passed)
        text = f'step={self.step:07d}, tr_loss={tr_loss:.3f}, val_loss={val_loss:.3f}, '
        # text = f'step={self.step:07d}, tr_loss={tr_loss:.3f}, val_acc={val_acc:.3f}, '
        text += f'lr={lr:.2e}, {tr_speed:.2f} it/s, {tr_speed_vx:.2f} MVx/s, {t}'
        logger.info(text)

    def _log_to_tensorboard(
        self,
        stats: Dict,
        misc: Dict,
        tr_nodes: Dict,
        val_nodes: Optional[Dict] = None,
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
                        self.tb.add_scalar(f'{tag}/{key}', value[i], self.step - len(value) + i)
            elif not np.isnan(value):
                self.tb.add_scalar(f'{tag}/{key}', value, self.step)

    def _tb_log_histograms(self) -> None:
        """Log histograms of model parameters and their current gradients.

        Make sure to run this between ``backward()`` and ``zero_grad()``,
        because otherwise gradient histograms will only consist of zeros.
        """
        for name, param in self.model.named_parameters():
            self.tb.add_histogram(f'param/{name}', param, self.step)
            grad = param.grad if param.grad is not None else torch.tensor(0)
            self.tb.add_histogram(f'grad/{name}', grad, self.step)
