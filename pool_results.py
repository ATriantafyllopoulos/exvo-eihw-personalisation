import argparse
import numpy as np
import os
import yaml

from define import (
    EMOTIONS
)

parser = argparse.ArgumentParser('Pool results')
parser.add_argument('root')
args = parser.parse_args()

print('====================')
print(f'TASK1')
print('====================')
print('| Model | Emo-CCC | Cou-UAR | Age-MAE | Score |')
print('| - | - | - | - | - |')
for model in os.listdir(os.path.join(args.root, 'task1')):
    with open(os.path.join(args.root, 'task1', model, 'dev.yaml'), 'r') as fp:
        results = yaml.load(fp, Loader=yaml.Loader)
    uar = results['Country']['UAR']
    mae = results['Age']['MAE']
    ccc = np.mean([results[x]['CCC'] for x in EMOTIONS])
    score = 3 / ( (1 / uar) + mae + (1 / ccc))
    print(f'| {model} | {ccc:.4f} | {uar:.4f} | {mae:.4f} | {score:.4f} |')
print()
print('====================')
print(f'TASK3')
print('====================')
print('| Model | Emo-CCC |')
print('| - | - |')
for model in os.listdir(os.path.join(args.root, 'task3')):
    with open(os.path.join(args.root, 'task3', model, 'dev.yaml'), 'r') as fp:
        results = yaml.load(fp, Loader=yaml.Loader)
    ccc = np.mean([results[x]['CCC'] for x in EMOTIONS])
    print(f'| {model} | {ccc:.4f} |')
    