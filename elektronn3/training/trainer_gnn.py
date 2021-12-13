import os
import logging
import datetime
from pickle import PicklingError
from typing import Optional, Dict, Tuple, Union, List, Any
from tqdm import tqdm
import numpy as np
import zipfile
import pprint
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("agg")

import torch
import torch.nn
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
from torch.utils import collect_env
import torch_geometric
from sklearn.manifold import TSNE
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
        data: torch_geometric.data.Data,
        valid_metrics: Optional[Dict] = None,
        exp_name: Optional[str] = None,
        batch_size: int = 1,
        early_stopping: Optional[Any] = None,
        num_workers: int = 0,
        enable_tensorboard: bool = True,
        tensorboard_root_path: Optional[str] = None,
        out_channels: Optional[int] = None,
        tqdm_kwargs: Optional[Dict] = None,
    ):
        model.to(device)
        if isinstance(criterion, torch.nn.Module):
            criterion.to(device)

        self.device = device
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.data = data
        self.valid_metrics = valid_metrics
        self.save_root = os.path.expanduser(save_root)
        self.batch_size = batch_size
        self.early_stopping = early_stopping
        self.num_workers = num_workers
        self.tqdm_kwargs = {} if tqdm_kwargs is None else tqdm_kwargs
        
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

        self.out_channels = out_channels
        self.tb = None
        if enable_tensorboard:
            if tensorboard_root_path is None:
                tb_path = self.save_path
            else:
                tensorboard_root_path = os.path.expanduser(tensorboard_root_path)
                tb_path = os.path.join(tensorboard_root_path, self.exp_name)
                os.makedirs(tb_path, exist_ok=True)
            self.tb = SummaryWriter(tb_path, flush_secs=20)

        self.num_nodes = self.data.x.shape[0]
        # Train and val mask for semi-supervised training
        
        logger.info(f"Training nodes: {self.data.train_mask.sum()}")
        logger.info(f"Validation nodes: {self.data.val_mask.sum()}")
        logger.info(f"Training node label rate: {(self.data.train_mask.sum() / self.num_nodes) * 100:.2f}%\n")

        assert self.data.has_isolated_nodes() == False, "Dataset has isolated nodes"
        assert self.data.is_undirected() == True, "Dataset is not undirected"
        assert self.data.has_self_loops() == False, "Dataset has self-loops"

        self.best_val_loss = np.inf
        self.best_tr_loss = np.inf

    def run(self, max_epochs: int=1, max_runtime=3600*24*7) -> None:
        self.start_time = datetime.datetime.now()
        self.end_time = self.start_time + datetime.timedelta(seconds=max_runtime)
        # Save initial model
        self._save_model(suffix="_initial", verbose=False)

        # logger.info("Saving initial node embeddings\n")
        # with torch.no_grad():
        #     out = self.model(self.data)

        # embeddings = TSNE(perplexity=20, n_iter=750).fit_transform(out.cpu().detach().numpy())
        # img_data = visualize_embeddings(embeddings, self.data.y.cpu().numpy(), self.epoch)
        # self.tb.add_image("embeddings_initial", img_data, self.epoch)

        stats: Dict[str, Union[float, List[float]]] = {stat: [] for stat in ['tr_loss', 'tr_acc']}
        misc: Dict[str, Union[float, List[float]]] = {}
        
        while not self.terminate:
            try:
                timer = Timer()
                correct = 0
                
                # Training loss
                dloss, dout = self._train()
                loss = float(dloss.item())

                # Training accuracy
                pred = torch.argmax(dout, dim=1)
                correct = torch.sum(pred[self.data.train_mask] == self.data.y[self.data.train_mask]).item()

                stats['tr_loss'].append(loss)
                stats["tr_loss_std"] = np.std(stats["tr_loss"])
                stats['tr_acc'].append(correct / self.data.train_mask.sum())
                misc["tr_speed"] = self.num_nodes / timer.t_passed

                self._tracker.update_timeline([self._timer.t_passed, loss])

                valid_stats = self._validate()
                stats.update(valid_stats)

                if self.early_stopping is not None:
                    self.early_stopping(stats["val_loss"][-1])
                    if self.early_stopping.stop:
                        logger.info(f"Validation loss did not improve for {self.patience} epochs. Terminating...")
                        self.terminate = True
    
                # Log on the console
                self._log_basic(stats, misc)

                # Log on tensorboard
                self.tb.add_scalar('stats/tr_loss', stats["tr_loss"][-1], self.epoch)
                self.tb.add_scalar('stats/tr_acc', stats['tr_acc'][-1], self.epoch)
                self.tb.add_scalar('stats/val_loss', stats['val_loss'][-1], self.epoch)
                self.tb.add_scalar('stats/val_acc', stats['val_acc'][-1], self.epoch)
                self._tb_log_histograms()

                # Save current model
                self._save_model(val_loss=stats['val_loss'][-1], verbose=False)
                # Save best model
                if stats["val_loss"][-1] < self.best_val_loss:
                    self.best_val_loss = stats["val_loss"][-1]
                    self._save_model(suffix="_best", val_loss=stats["val_loss"][-1])
                    print()

                # Increment epochs
                self._incr_epoch(max_epochs, max_runtime)

            except KeyboardInterrupt: 
                break

            except Exception as e:
                logger.exception("Unhandled exception during training:")
                raise e

        # Save final model
        self._save_model(suffix="_final")
        # logger.info("Saving final node embeddings\n")
        # with torch.no_grad():
        #     out = self.model(self.data)

        # embeddings = TSNE(perplexity=20, n_iter=750).fit_transform(out.cpu().detach().numpy())
        # img_data = visualize_embeddings(embeddings, self.data.y.cpu().numpy(), self.epoch)
        # self.tb.add_image("embeddings_final", img_data, self.epoch)

        if self.tb is not None:
            self.tb.close()  # Ensure that everything is flushed

    def _train(self) -> None:
        self.model.train()
        self.optimizer.zero_grad()
        
        dout = self.model(self.data)
        dloss = self.criterion(dout[self.data.train_mask], self.data.y[self.data.train_mask].long())
        
        if torch.isnan(dloss):
            logger.error(f"NaN loss encountered at step {self.epoch}. Aborting...")
            raise NaNException()
        
        dloss.backward()
        self.optimizer.step()
        
        return dloss, dout

    @torch.no_grad()
    def _validate(self) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
        self.model.eval()
        stats = {stat: [] for stat in ['val_loss', 'val_acc']}

        # Validation loss
        dout = self.model(self.data)
        dloss = self.criterion(dout[self.data.val_mask], self.data.y[self.data.val_mask].long())

        # Validation accuracy
        pred = torch.argmax(dout, dim=1)
        correct = torch.sum(pred[self.data.val_mask] == self.data.y[self.data.val_mask]).item()
        
        stats["val_loss"].append(dloss.item())
        stats["val_loss_std"] = np.std(stats["val_loss"])
        stats["val_acc"].append(correct / self.data.val_mask.sum())

        return stats

    def _incr_epoch(self, max_epochs, max_runtime):
        self.epoch += 1
        
        if self.epoch >= max_epochs:
            logger.info(f"max_steps ({max_epochs}) exceeded. Terminating...")
            self.terminate = True

        if datetime.datetime.now() >= self.end_time:
            logger.info(f"max_runtime ({max_runtime} seconds) exceeded. Terminating...")
            self.terminate = True

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
        tr_loss = stats["tr_loss"][-1]
        val_loss = stats["val_loss"][-1]
        val_acc = stats["val_acc"][-1]
        tr_speed = misc['tr_speed']
        t = pretty_string_time(self._timer.t_passed)
        text = f'epoch={self.epoch:02d}, tr_loss={tr_loss:.3f}, val_loss={val_loss:.3f}, val_acc={val_acc:.2f} '
        text += f'tr_speed={tr_speed:.2f} it/s, {t}'
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
