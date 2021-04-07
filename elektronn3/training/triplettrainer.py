from typing import Dict, Any, Tuple, Union, List

import numpy as np
import torch
from tqdm import tqdm

from elektronn3.training import Trainer, handlers
from elektronn3.training.train_utils import Timer
from elektronn3.training.trainer import logger, NaNException


class TripletTrainer(Trainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        def noop(*args, **kwargs): pass

        self.preview_plotting_handler = noop  # TODO
        self.sample_plotting_handler = handlers._tb_log_sample_images_all_img

    def _train_step_triplet(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Core training step for triplet loss on self.device"""
        # Everything with a "d" prefix refers to tensors on self.device (i.e. probably on GPU)
        danchor = batch['anchor'].to(self.device, non_blocking=True)
        dpos = batch['pos'].to(self.device, non_blocking=True)
        dneg = batch['neg'].to(self.device, non_blocking=True)
        # forward pass
        danc_out = self.model(danchor)
        dpos_out = self.model(dpos)
        dneg_out = self.model(dneg)
        dloss = self.criterion(danc_out, dpos_out, dneg_out)
        if torch.isnan(dloss):
            logger.error('NaN loss detected! Aborting training.')
            raise NaNException
        # update step
        self.optimizer.zero_grad()
        dloss.backward()
        self.optimizer.step()
        return dloss, {'anchor_out': danc_out, 'pos_out': dpos_out, 'neg_out': dneg_out}

    def _train(self, max_steps, max_runtime):
        """Train for one epoch or until max_steps or max_runtime is reached"""
        self.model.train()

        # Scalar training stats that should be logged and written to tensorboard later
        stats: Dict[str, Union[float, List[float]]] = {stat: [] for stat in ['tr_loss']}
        # Other scalars to be logged
        misc: Dict[str, Union[float, List[float]]] = {misc: [] for misc in ['mean_target']}
        # Hold image tensors for real-time training sample visualization in tensorboard
        images: Dict[str, np.ndarray] = {}

        running_vx_size = 0  # Counts input sizes (number of pixels/voxels) of training batches
        timer = Timer()
        batch_iter = tqdm(
            self.train_loader, 'Training', total=len(self.train_loader), dynamic_ncols=True, **self.tqdm_kwargs
        )
        for i, batch in enumerate(batch_iter):
            if self.step in self.extra_save_steps:
                self._save_model(f'_step{self.step}', verbose=True)

            dloss, dout_imgs = self._train_step_triplet(batch)

            with torch.no_grad():
                loss = float(dloss)
                mean_target = 0.  # Dummy value
                misc['mean_target'].append(mean_target)
                stats['tr_loss'].append(loss)
                batch_iter.set_description(f'Training (loss {loss:.4f})')
                self._tracker.update_timeline([self._timer.t_passed, loss, mean_target])

            # Not using .get_lr()[-1] because ReduceLROnPlateau does not implement get_lr()
            misc['learning_rate'] = self.optimizer.param_groups[0]['lr']  # LR for the this iteration
            self._scheduler_step(loss)

            running_vx_size += batch['anchor'].numel()

            self._incr_step(max_runtime, max_steps)
            if i == len(self.train_loader) - 1 or self.terminate:
                # Last step in this epoch or in the whole training
                # Preserve last training batch and network output for later visualization
                for key, img in batch.items():
                    if isinstance(img, torch.Tensor):
                        img = img.detach().cpu().numpy()
                    images[key] = img
                self._put_current_attention_maps_into(images)

                # TODO: The plotting handler abstraction is inadequate here. Figure out how
                #       we can handle plotting cleanly in one place.
                # Outputs are visualized here, while inputs are visualized in the plotting handler
                #  which is called in _run()...
                for name, img in dout_imgs.items():
                    img = img.detach()[0].cpu().numpy()  # select first item of batch
                    for c in range(img.shape[0]):
                        if img.ndim == 4:  # 3D data
                            img = img[:, img.shape[0] // 2]  # take center slice of depth dim -> 2D
                        self.tb.add_figure(
                            f'tr_samples/{name}_c{c}',
                            handlers.plot_image(img[c], cmap='gray'),
                            global_step=self.step
                        )

            if self.terminate:
                break

        stats['tr_loss_std'] = np.std(stats['tr_loss'])
        misc['tr_speed'] = len(self.train_loader) / timer.t_passed
        misc['tr_speed_vx'] = running_vx_size / timer.t_passed / 1e6  # MVx

        return stats, misc, images

    def _validate(self) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
        raise NotImplementedError
