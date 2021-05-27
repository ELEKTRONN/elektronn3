import time

import torch
import numpy as np

from elektronn3.inference import Predictor
from elektronn3.models.unet import UNet


torch.backends.cudnn.benchmark = True

print(' == Setting up...')

# float16 = False
# jit = False
sample_shape = (1, 1, 512, 512, 512)
tile_shape = (128, 128, 128)
overlap_shape = (0, 0, 0)


def benchmark(float16, jit):

    device = torch.device('cuda')
    n = 3  # Number of repetitions
    dtype = torch.float16 if float16 else torch.float32
    print(f'dtype: {dtype}')
    print(f'jit: {jit}')
    print(f'sample_shape, tile_shape, overlap_shape: {sample_shape, tile_shape, overlap_shape}')
    _float16_str = 'float16' if float16 else 'float32'
    _jit_str = 'jit' if jit else 'nojit'
    experiment_name = f'{_float16_str}_{_jit_str}'
    print(f'Experiment name: {experiment_name}')

    model = UNet(
        out_channels=2,
        n_blocks=4,
        start_filts=32,
        planar_blocks=(0,),
        activation='relu',
        normalization='batch',
        # conv_mode='valid',
    ).to(device)
    if jit:
        model = torch.jit.script(model)

    x = torch.ones(*sample_shape, dtype=dtype, device=device)
    predictor = Predictor(
        model,
        # tile_shape=(64, 128, 128),
        # overlap_shape=(32, 64, 64),
        tile_shape=tile_shape,
        overlap_shape=overlap_shape,
        out_shape=(2, *x.shape[2:]),
        verbose=False,
        float16=float16
    )

    print(' == Warming up...')
    r = predictor.predict(x).cpu()
    del r

    print(' == Start timing inference speed...')
    start_total = time.time()

    for _ in range(n):
        startm = time.time()
        predictor.predict(x)
        torch.cuda.synchronize()
        dt = time.time() - startm
        print(f'Inference run time (sec): {dt:.2f}')

    dt_total = time.time() - start_total
    dt_total_per_run = dt_total / n
    print(f'Average time ({n} runs) (sec): {dt_total_per_run:.2f}')
    print('\n\n')


    # def trace_handler(prof):
    #     print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=-1))
    #     prof.export_chrome_trace("./profiler_trace_" + str(prof.step_num) + ".json")
    #     prof.tensorboard_trace_handler('./profiler_tb')


    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        use_cuda=True,
        # record_shapes=True,
        # with_flops=True,
        # schedule=torch.profiler.schedule(
        #     wait=1,
        #     warmup=1,
        #     active=2,
        # ),
        # on_trace_ready=torch.profiler.tensorboard_trace_handler(f'./profiler_tb_{experiment_name}'),
    ) as prof:
        with torch.profiler.record_function(f'model_inference'):
            predictor.predict(x)

    print('torch.profiler results:\n')

    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=-1))
    print()
    # prof.export_chrome_trace(f"./profiler_trace.json_{experiment_name}")


for _float16 in [False, True]:
    for _jit in [False, True]:
        benchmark(float16=_float16, jit=_jit)
