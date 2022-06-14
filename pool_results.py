import argparse
import audmetric
import pandas as pd
import numpy as np
import os
import yaml

from define import (
    EMOTIONS
)

def score_func(preds, labels):

    ccc = 0
    for x, y in zip(labels.T, preds.T):
        ccc += audmetric.concordance_cc(x, y)
    ccc = ccc / 10
    return ccc

def get_CI(preds, labels):

    scores = []
    for s in range(1000):
        np.random.seed(s)
        sample = np.random.choice(range(len(preds)), len(preds), replace=True) #boost with replacement
        sample_preds = preds[sample]
        sample_labels = labels[sample]
        scores.append(score_func(sample_labels, sample_preds))

    q_0 = pd.DataFrame(np.array(scores)).quantile(0.025)[0] #2.5% percentile
    q_1 = pd.DataFrame(np.array(scores)).quantile(0.975)[0] #97.5% percentile

    return q_0, q_1

parser = argparse.ArgumentParser('Pool results')
parser.add_argument('root')
parser.add_argument('data_root')
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
print('| Model | Emo-CCC | Country-mean | Country-std | Speaker-mean | Speaker-std |')
print('| - | - | - | - | - | - |')
for model in os.listdir(os.path.join(args.root, 'task3')):
    try:
        with open(os.path.join(args.root, 'task3', model, 'dev.yaml'), 'r') as fp:
            results = yaml.load(fp, Loader=yaml.Loader)
    except FileNotFoundError as e:
        print(e)
        continue
    ccc = np.mean([results[x]['CCC'] for x in EMOTIONS])
    
    df = pd.read_csv(os.path.join(args.data_root, 'data_info.csv'))
    df['file'] = df['File_ID'].apply(lambda x: x.strip('[').strip(']') + '.wav')
    df.set_index('file', inplace=True)
    df = df.loc[df['Split'] == 'Val']

    preds = np.load(os.path.join(args.root, 'task3', model, 'outputs.npy'))
    labels = np.load(os.path.join(args.root, 'task3', model, 'targets.npy'))
    
    c_scores = []
    for country in df['Country_string'].unique():
        indices = df['Country_string'] == country
        c_scores.append(score_func(preds=preds[indices], labels=labels[indices]))
    
    s_scores = []
    for speaker in df['Subject_ID'].unique():
        indices = df['Subject_ID'] == speaker
        s_scores.append(score_func(preds=preds[indices], labels=labels[indices]))

    # scores = ' |'.join([f'{results[x]["CCC"]:.3f}'[1:] for x in EMOTIONS])
    # print(f'| {model} | {scores}')
    # q_0, q_1 = get_CI(
    #     preds=np.load(os.path.join(args.root, 'task3', model, 'outputs.npy')),
    #     labels=np.load(os.path.join(args.root, 'task3', model, 'targets.npy'))
    # )
    # print(f'| {model} | {ccc:.4f} [{q_0:.4f}-{q_1:.4f}]|')
    # [{q_0:.4f}-{q_1:.4f}]
    print((
        f'| {model} | {ccc:.4f} | {np.mean(c_scores):.4f} | {np.std(c_scores):.4f}'
        f'| {np.mean(s_scores):.4f} | {np.std(s_scores):.4f} |'
    ))
    
    