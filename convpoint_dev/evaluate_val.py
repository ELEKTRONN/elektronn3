import os
import glob
import pickle
import morphx.processing.visualize as visualize

data_path = os.path.expanduser('~/wholebrain/wholebrain/u/jklimesch/gt/simple_training/SegSmall_b16_r10000_s800/val_examples/')
files = glob.glob(data_path + '*.pkl')

for file in files:
    with open(file, 'rb') as f:
        results = pickle.load(f)
    for i in range(0, int(len(results)/2), 2):
        orig = results[i]
        pred = results[i+1]
        visualize.visualize_parallel([orig], [pred])
