import os

import torch
from torch import nn
import torch.utils.data
import numpy as np
from typing import Tuple, Dict, Callable

from tqdm import tqdm

import elektronn3
#from elektronn3.modules import DiceLoss, CombinedLoss
from elektronn3.modules import DiceLoss
from elektronn3.data import PatchCreator, transforms

from elektronn3.training import metrics

from unet_for_ax import train

from ax import RangeParameter, ParameterType
from ax import optimize
from ax import *
from ax.plot.contour import plot_contour
from ax.utils.notebook.plotting import render



if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


def validate(
        model: torch.nn.Module,
        valid_loader: torch.utils.data.DataLoader,
        criterion: torch.nn.Module,
        device: torch.device,
        valid_metrics: Dict[str, Callable] = {}
) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
    model.eval()  # Set dropout and batchnorm to eval mode

    val_loss = []
    stats = {name: [] for name in valid_metrics.keys()}
    # TODO: Avoid unnecessary cpu -> gpu -> cpu moves, just save cpu tensors for later
    print("reaching here!")
    max_steps = 100
    steps = 0
    for inp, target in tqdm(valid_loader, 'Validating'):
        #inp = torch.from_numpy(inp).float()
        inp = torch.as_tensor(inp)
        inp = inp[None]
        #target = torch.from_numpy(target).float()
        target= torch.as_tensor(target)
        target= target[None]
        # Everything with a "d" prefix refers to tensors on self.device (i.e. probably on GPU)
        dinp = inp.to(device, non_blocking=True)
        dtarget = target.to(device, non_blocking=True)
        with torch.no_grad():
            dout = model(dinp)
            val_loss.append(criterion(dout, dtarget).item())
            out = dout.detach().cpu()
            for name, evaluator in valid_metrics.items():
                stats[name].append(evaluator(target, out))
        print("printing steps:", steps)
        steps+=1
        if steps >= max_steps:
            break

    images = {
        'inp': inp.numpy(),
        'out': out.numpy(),
        'target': target.numpy()
    }

    stats['val_loss'] = np.mean(val_loss)
    stats['val_loss_std'] = np.std(val_loss)
    for name in valid_metrics.keys():
        stats[name] = np.nanmean(stats[name])

    model.train()  # Reset model to training mode
    print("printing:", stats['val_loss'])
    return stats, images


valid_metrics = {
    'val_accuracy': metrics.bin_accuracy,
    'val_precision': metrics.bin_precision,
    'val_recall': metrics.bin_recall,
    'val_DSC': metrics.bin_dice_coefficient,
    'val_IoU': metrics.bin_iou,
}

# These statistics are computed from the training dataset.
# Remember to re-compute and change them when switching the dataset.
dataset_mean = (0.6170815,)
dataset_std = (0.15687169,)
# Class weights for imbalanced dataset
class_weights = torch.tensor([0.2808, 0.7192]).to(device)

aniso_factor = 2  # Anisotropy in z dimension. E.g. 2 means half resolution in z dimension.
common_data_kwargs = {  # Common options for training and valid sets.
    'aniso_factor': aniso_factor,
    'patch_shape': (48, 96, 96),
    # 'offset': (8, 20, 20),
    'num_classes': 2,
}

crossentropy = nn.CrossEntropyLoss(weight=class_weights)
dice = DiceLoss(apply_softmax=True, weight=class_weights)
#criterion = CombinedLoss([crossentropy, dice], weight=[0.5, 0.5], device=device)
criterion = crossentropy
data_root = '/wholebrain/scratch/j0126/barrier_gt_phil/'

# Transformations to be applied to samples before feeding them to the network
common_transforms = [
    transforms.SqueezeTarget(dim=0),  # Workaround for neuro_data_cdhw
    transforms.Normalize(mean=dataset_mean, std=dataset_std)
]
train_transform = transforms.Compose(common_transforms + [
    # transforms.RandomGrayAugment(channels=[0], prob=0.3),
    # transforms.RandomGammaCorrection(gamma_std=0.25, gamma_min=0.25, prob=0.3),
    # transforms.AdditiveGaussianNoise(sigma=0.1, channels=[0], prob=0.3),
])
valid_transform = transforms.Compose(common_transforms + [])

valid_dataset = PatchCreator(
    #input_h5data=[input_h5data[i] for i in range(len(input_h5data)) if i in valid_indices],
    #target_h5data=[target_h5data[i] for i in range(len(input_h5data)) if i in valid_indices],
    #data_root = '/wholebrain/scratch/j0126/barrier_gt_phil/',
    #data_root = os.path.expanduser('~/neuro_data_cdhw/'),
    #fnames = sorted([f for f in os.listdir(data_root) if f.endswith('.h5')]),
    #input_h5data = [(os.path.join(data_root, f), 'raW') for f in fnames],
    #target_h5data = [(os.path.join(data_root, f), 'labels') for f in fnames],
    input_h5data= [('/u/mahsabh/neuro_data_cdhw/raw_0.h5', 'raw')],
    target_h5data = [('/u/mahsabh/neuro_data_cdhw/barrier_int16_0.h5', 'lab')],
    train=False,
    epoch_size=1,  # How many samples to use for each validation run
    warp_prob=0,
    warp_kwargs={'sample_aniso': aniso_factor != 1},
    transform=valid_transform,
    **common_data_kwargs
)

model_path="/u/mahsabh/e3training_june2019/withElasticRandomGray/UNet__19-06-19_14-55-09/state_dict_swa.pth"
model = torch.load(model_path)




#validate(model, valid_loader=valid_dataset, criterion=criterion, device=device, valid_metrics=valid_metrics)



#range_param = RangeParameter(name="elastic_prob", parameter_type=ParameterType.FLOAT, lower=0.0, upper=1.0)
#range_param = RangeParameter(name="gray_prob", parameter_type=ParameterType.FLOAT, lower=0.0, upper=1.0)


#def evaluation_function(
#        parameterization,
#        weight= None,
#):
#
#    prob = parameterization["elastic_prob"]
#    stats, _ = validate(model, valid_loader=valid_dataset, criterion=criterion, device=device, valid_metrics=valid_metrics)
#    objective_val= stats['val_loss']
#    return {"objective": objective_val}


def train_evaluate(parameterization):
    trained_model = train(parameterization, max_steps = 1000, resume = None)
    print("trained model returned ")
    stats, _ = validate(trained_model, valid_loader=valid_dataset, criterion=criterion, device=device,
                        valid_metrics=valid_metrics)
    print("validation over ")
    objective_val = stats['val_loss']
    return objective_val

print("starting the optimization loop")

#choice_param = ChoiceParameter(name="start_filts", values=list(range(2,33)), parameter_type=ParameterType.INT)

best_parameters, best_values, experiment, model = optimize(
    parameters=[
        {"name": "AGN_prob", "type": "range", "bounds": [0.0, 1.0]},
        {"name": "start_filts", "type": "choice", "values" : list(range(2,33)) },
        #"start_filts",

        #ChoiceParameter(name="start_filts", values=list(range(2,33)), parameter_type=ParameterType.INT),
    ],
    evaluation_function = train_evaluate,
    minimize = True,
    total_trials = 1,

)

print(best_parameters)
print(best_values)

render(plot_contour(model=model, param_x='AGN_prob', param_y='start_filts', metric_name='val_loss'))