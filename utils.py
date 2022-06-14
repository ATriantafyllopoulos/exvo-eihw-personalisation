import audmetric
import audobject
import numpy as np
import torch
import tqdm

from define import (
    EMOTIONS
)


def transfer_features(features, device):
    return features.to(device).float()


def evaluate_multitask(
    model,
    device, 
    loader,
    task_dict,
    transfer_func,
    output_dim: int = None,
    score: bool = True
):
    metrics = {
        'classification': {
            'UAR': audmetric.unweighted_average_recall,
            'ACC': audmetric.accuracy,
            'F1': audmetric.unweighted_average_fscore
        },
        'regression': {
            'CC': audmetric.pearson_cc,
            'CCC': audmetric.concordance_cc,
            'MSE': audmetric.mean_squared_error,
            'MAE': audmetric.mean_absolute_error
        }
    }

    model.to(device)
    model.eval()

    outputs = torch.zeros((len(loader.dataset), model.output_dim if output_dim is None else output_dim))
    if score:
        targets = torch.zeros((len(loader.dataset), len(task_dict)))
    with torch.no_grad():
        for index, (features, target) in tqdm.tqdm(
            enumerate(loader),
            desc='Batch',
            total=len(loader),
            disable=score
        ):
            start_index = index * loader.batch_size
            end_index = (index + 1) * loader.batch_size
            if end_index > len(loader.dataset):
                end_index = len(loader.dataset)
            outputs[start_index:end_index, :] = model(
                transfer_func(features, device))
            if score:
                targets[start_index:end_index] = target
            # break

    outputs = outputs.cpu().numpy()
    if not score:
        return outputs
    targets = targets.numpy()
    predictions = []
    results = {}
    for task in task_dict:
        results[task] = {}
        if task_dict[task]['type'] == 'regression':
            preds = outputs[:, task_dict[task]['unit']]
        else:
            preds = outputs[:, task_dict[task]['unit']].argmax(1)
        predictions.append(preds)
        for metric in metrics[task_dict[task]['type']]:
            results[task][metric] = metrics[task_dict[task]['type']][metric](
                targets[:, task_dict[task]['target']],
                preds
            )
    predictions = np.stack(predictions).T
    total_score = []
    for task in task_dict:
        score = results[task][task_dict[task]['score']]
        if task_dict[task]['score'] in ['MAE', 'MSE']:
            score = 1 / (score + 1e-9)
        total_score.append(score)
    emo_score = sum([x for x, y in zip(total_score, task_dict) if y in EMOTIONS]) / len(EMOTIONS)
    if len(task_dict) == len(EMOTIONS):
        total_score = emo_score
    elif len(task_dict) == 1:
        total_score = total_score[0]
    else:
        scores = [emo_score] + [x for x, y in zip(total_score, task_dict) if y not in EMOTIONS]
        total_score = len(scores) / sum([1 / (score + 1e-9) for score in scores])

    return total_score, results, targets, outputs, predictions


class CCCLoss(torch.nn.Module):
    def forward(self, output, target):
        out_mean = torch.mean(output)
        target_mean = torch.mean(target)

        covariance = torch.mean((output - out_mean) * (target - target_mean))
        target_var = torch.mean((target - target_mean)**2)
        out_var = torch.mean((output - out_mean)**2)

        ccc = 2.0 * covariance / \
            (target_var + out_var + (target_mean - out_mean)**2 + 1e-10)
        loss_ccc = 1.0 - ccc

        return loss_ccc


class LabelEncoder(audobject.Object):
    def __init__(self, labels):
        self.labels = sorted(labels)
        codes = range(len(labels))
        self.inverse_map = {code: label for code,
                    label in zip(codes, labels)}
        self.map = {label: code for code,
                            label in zip(codes, labels)}

    def encode(self, x):
        return self.map[x]

    def decode(self, x):
        return self.inverse_map[x]



def disaggregated_evaluation(df, groundtruth, task_dict, stratify):
    metrics = {
        'classification': {
            'UAR': audmetric.unweighted_average_recall,
            'ACC': audmetric.accuracy,
            'F1': audmetric.unweighted_average_fscore
        },
        'regression': {
            'CC': audmetric.pearson_cc,
            'CCC': audmetric.concordance_cc,
            'MSE': audmetric.mean_squared_error,
            'MAE': audmetric.mean_absolute_error
        }
    }
    df = df.reindex(groundtruth.index)
    results = {}
    for task in task_dict:
        results[task] = {}
        for stratifier in stratify:
            for variable in groundtruth[stratifier].unique():
                indices = groundtruth.loc[groundtruth[stratifier]
                                          == variable].index
                df_strat = df.reindex(indices)[f'{task}.pred']
                gt_strat = groundtruth.reindex(indices)[task]
                for metric in metrics[task_dict[task]['type']]:
                    if metric not in results[task]:
                        results[task][metric] = {}
                    results[task][metric][variable] = metrics[task_dict[task]['type']][metric](
                        gt_strat,
                        df_strat
                    )
    return results