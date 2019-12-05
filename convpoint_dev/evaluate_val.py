import os
import glob
import pickle
import random
import argparse
import morphx.processing.visualize as visualize
from getkey import keys

parser = argparse.ArgumentParser(description='Evaluate validation set.')
parser.add_argument('--bs', type=int, default=16, help='Set batch size.')
parser.add_argument('--np', type=int, default=1000, help='Set number of sample points.')
parser.add_argument('--rad', type=int, default=10000, help='Set radius of local BFS.')

args = parser.parse_args()

batch_size = args.bs
npoints = args.np
radius = args.rad
base_path = os.path.expanduser("~/wholebrain/wholebrain/u/jklimesch/gt/simple_training/"
                               "SegSmall_b{}_r{}_s{}/".format(batch_size, radius, npoints))
data_path = base_path + 'val_examples/'
save_path = base_path + 'im_examples/'
if not os.path.exists(save_path):
    os.makedirs(save_path)

print(data_path)
files = glob.glob(data_path + '*.pkl')
random.shuffle(files)

idx = 0
reverse = False
while idx < len(files):
    file = files[idx]
    slashs = [pos for pos, char in enumerate(file) if char == '/']
    filename = file[slashs[-1]:-4]
    print("Viewing: " + filename)

    with open(file, 'rb') as f:
        results = pickle.load(f)
    if reverse:
        i = int(len(results))-2
    else:
        i = 0
    while i < int(len(results)):
        orig = results[i]
        pred = results[i+1]
        key = visualize.visualize_parallel([orig], [pred],
                                           name1=filename+'_i{}_orig'.format(i),
                                           name2=filename+'_i{}_pred'.format(i+1),
                                           static=True)
        if key == keys.RIGHT:
            reverse = False
            i += 2
        if key == keys.UP:
            print("Saving to png...")
            path = save_path + '{}_i{}'.format(filename, i)
            visualize.visualize_single([orig], capture=True, path=path + '_1.png')
            visualize.visualize_single([pred], capture=True, path=path + '_2.png')
        if key == keys.DOWN:
            print("Displaying interactive view...")
            # display images for interaction
            visualize.visualize_parallel([orig], [pred])
        if key == keys.LEFT:
            reverse = True
            i -= 2
            if i < 0:
                break
        if key == keys.ENTER:
            quit()

    if reverse:
        idx -= 1
    else:
        idx += 1
