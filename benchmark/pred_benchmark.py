import os
import time

import torch
import numpy as np

from elektronn3.models.unet import UNet


torch.backends.cudnn.benchmark = True

print(' == Setting up...')

jit = False

# Determine input sizes for optimal VRAM usage
cluster = os.getenv('CLUSTER', default='UNKNOWN')
# These sizes work for wholebrain (16125MiB VRAM @ RTX 5000) and larger GPUs
s2 = 512 + 128
s3 = 64 + 16

## Uncomment to get cluster-specific sizes
# if cluster == 'WHOLEBRAIN':
#     # wholebrain (16125MiB VRAM @ RTX 5000)
#     s2 = 512 + 128
#     s3 = 64 + 16
# elif cluster == 'COBRA':
#     # cobra (32510MiB VRAM @ V100)
#     s2 = 512 + 256
#     s3 = 64 + 32
# elif cluster == 'RAVEN':
#     # raven (40536MiB VRAM @ A100)
#     s2 = 512 + 256
#     s3 = 64 + 32
# else:
#     s2 = 512
#     s3 = 64

print(f'Running on {cluster}')


inp_shape2 = (8, 1, s2, s2)
inp_shape3 = (8, 1, s3, s3, s3)

inp_shapes = [
    inp_shape2,
    inp_shape3,
]

def benchmark(float16, inp_shape):
    torch.cuda.empty_cache()
    dim = len(inp_shape) - 2
    device = torch.device('cuda')
    n = 10  # Number of measured repetitions
    dtype = torch.float16 if float16 else torch.float32
    print(f'dtype: {dtype}')
    print(f'dim: {dim}')
    print(f'inp_shape: {inp_shape}')
    _float16_str = 'float16' if float16 else 'float32'
    experiment_name = f'{dim}d_{_float16_str}'
    print(f'Experiment name: {experiment_name}')

    model = UNet(
        dim=dim,
        out_channels=2,
        n_blocks=4,
        start_filts=32,
        activation='relu',
        normalization='batch',
        # conv_mode='valid',
    ).to(device, dtype)
    if jit:
        model = torch.jit.script(model)

    x_warmup = torch.randn(*inp_shape, dtype=dtype)


    print(' == Warming up...')
    r = model(x_warmup.to(device)).cpu()
    torch.cuda.synchronize()
    del r

    torch.cuda.empty_cache()


    # Generate random inputs of same shape for measurements
    xm = [torch.randn(*inp_shape, dtype=dtype) for _ in range(n)]

    print(' == Start timing inference speed...')
    start_total = time.time()

    for i in range(n):
        startm = time.time()
        model(xm[i].to(device)).cpu()
        torch.cuda.synchronize()
        dt = time.time() - startm
        print(f'Inference run time (sec): {dt:.2f}')

    dt_total = time.time() - start_total
    dt_total_per_run = dt_total / n
    throughput = np.prod(inp_shape) / dt_total_per_run
    mvoxs = throughput / 1e6
    print(f'Average inference time ({n} runs) (sec): {dt_total_per_run:.2f}')
    print(f'Average MVox/s: {mvoxs:.2f}')
    print('\n\n')

for _inp_shape in inp_shapes:
    for _float16 in [False, True]:
        benchmark(float16=_float16, inp_shape=_inp_shape)
