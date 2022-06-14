import argparse
import audtorch
import numpy as np
import os
import pandas as pd
import random
import shutil
import torch
import tqdm
import yaml

from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter

from datasets import (
    CachedDataset,
    WavDataset
)
from define import (
    TASK_DICT
)
from models import (
    Cnn10,
    Cnn14
)

from utils import (
    disaggregated_evaluation,
    evaluate_multitask,
    transfer_features,
    LabelEncoder
)


def fix_index(df, root):
    df.reset_index(inplace=True)
    df['filename'] = df['filename'].apply(lambda x: os.path.join(args.data_root, x))
    df.set_index('filename', inplace=True)
    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser('EXVO Subject ID Training')
    parser.add_argument(
        '--data-root', 
        help='Path data has been extracted', 
        required=True
    )
    parser.add_argument(
        '--results-root', 
        help='Path where results are to be stored', 
        required=True
    )
    parser.add_argument(
        '--features',
        help='Path to features', 
        required=True
    )
    parser.add_argument(
        '--device', 
        help='CUDA-enabled device to use for training',
        required=True
    )
    parser.add_argument(
        '--state', 
        help='Optional initial state'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=8,
        help='Batch size'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=60
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=0.001
    )
    parser.add_argument(
        '--model',
        choices=[
            'CNN5',
            'cnn10',
            'cnn14'
        ],
        default='CNN5'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=0
    )
    parser.add_argument(
        '--mixup',
        default=False,
        action='store_true',
    )
    parser.add_argument(
        '--optimizer',
        default='SGD',
        choices=[
            'SGD',
            'Adam',
            'RMSprop'
        ]
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    experiment_folder = args.results_root
    os.makedirs(experiment_folder, exist_ok=True)

    df = pd.read_csv(os.path.join(args.data_root, 'data_info.csv'))
    df['file'] = df['File_ID'].apply(lambda x: x.strip('[').strip(']') + '.wav')
    df.set_index('file', inplace=True)
    df = df.loc[df['Split'] == 'Train']
    encoder = LabelEncoder(df['Subject_ID'].unique())
    encoder.to_yaml(os.path.join(experiment_folder, 'encoder.yaml'))
    df['Subject_ID'] = df['Subject_ID'].apply(encoder.encode)

    df_train, df_dev = train_test_split(df)
    df_test = df_dev
        
    target_column = ['Subject_ID']

    task_dict = {key: TASK_DICT[key] for key in target_column}
    unit_counter = 0
    for task_index, task in enumerate(target_column):
        task_dict[task]['target'] = task_index
        if task_dict[task]['type'] == 'regression':
            task_dict[task]['unit'] = unit_counter
            unit_counter += 1
        elif task_dict[task]['type'] == 'classification':
            task_dict[task]['unit'] = list(range(unit_counter, unit_counter + len(df_train[task].unique())))
            unit_counter += len(df_train[task].unique())
    output_dim = unit_counter

    criterion = torch.nn.CrossEntropyLoss()

    features = pd.read_csv(args.features).set_index('file')
    features['features'] = features['features'].apply(lambda x: os.path.join(os.path.dirname(args.features), os.path.basename(x)))
    
    db_args = {
        'features': features,
        'target_column': 'Subject_ID',
        'transform': audtorch.transforms.RandomCrop(250, axis=-2)
    }
    if args.model == 'CNN5':
        model = torch.nn.Sequential(
            torch.nn.Conv2d(1, 64, kernel_size=(3, 5), padding=(1, 2)),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.Dropout2d(0.2),
            torch.nn.Conv2d(64, 256, kernel_size=(3, 5), padding=(1, 2)),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            torch.nn.Dropout2d(0.2),
            torch.nn.Conv2d(256, 1024, kernel_size=(3, 5), padding=(1, 2)),
            torch.nn.BatchNorm2d(1024),
            torch.nn.ReLU(),
            torch.nn.Dropout2d(0.2),
            torch.nn.AdaptiveAvgPool2d((1, 1)),
            torch.nn.Flatten(1, 3),
            torch.nn.Linear(1024, 1024, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, output_dim, bias=True)
        )
    elif args.model == 'cnn10':
        model = Cnn10(
            output_dim=output_dim
        )
    elif args.model == 'cnn14':
        model = Cnn14(
            output_dim=output_dim
        )
    db_class = CachedDataset

    if args.state is not None:
        initial_state = torch.load(args.state)
        model.load_state_dict(
            initial_state,
            strict=False
        )

    train_dataset = db_class(
        df_train,
        **db_args
    )
    x, y = train_dataset[0]
    print(f'Input shape: {x.shape}')
    print(f'Output shape: {y.shape}')
    # exit()
    db_args.pop('transform')

    dev_dataset = db_class(
        df_dev,
        **db_args
    )

    test_dataset = db_class(
        df_test,
        **db_args
    )
    # create DataLoaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.batch_size,
        num_workers=4
    )
    # x, y = next(iter(train_loader))
    # model.to(args.device)
    # model.eval()
    # print(f'Input shape: {x.shape}')
    # print(f'Output shape: {y.shape}')
    # with torch.no_grad():
    #     print(f'Model shape: {model(x.to(args.device)).shape}')
    # exit()

    dev_loader = torch.utils.data.DataLoader(
        dev_dataset,
        shuffle=False,
        batch_size=1,
        num_workers=4
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=1,
        num_workers=4
    )
    device = args.device
    if not os.path.exists(os.path.join(experiment_folder, 'state.pth.tar')):

        with open(os.path.join(experiment_folder, 'hparams.yaml'), 'w') as fp:
            yaml.dump(vars(args), fp)

        writer = SummaryWriter(log_dir=os.path.join(experiment_folder, 'log'))

        torch.save(
            model.state_dict(), 
            os.path.join(
                experiment_folder, 
                'initial.pth.tar'
            )
        )

        if args.optimizer == 'SGD':
            optimizer = torch.optim.SGD(
                model.parameters(), 
                momentum=0.9, 
                lr=args.learning_rate
            )
        elif args.optimizer == 'Adam':
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=args.learning_rate
            )
        elif args.optimizer == 'RMSprop':
            optimizer = torch.optim.RMSprop(
                model.parameters(),
                lr=args.learning_rate,
                alpha=.95,
                eps=1e-7
            )
        epochs = args.epochs

        max_metric = -1
        best_epoch = 0
        best_state = None
        best_results = None

        plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='max', 
            factor=0.9,
            patience=5
        )

        for epoch in range(epochs):
            model.to(device)
            model.train()
            epoch_folder = os.path.join(
                experiment_folder, 
                f'Epoch_{epoch+1}'
            )
            os.makedirs(epoch_folder, exist_ok=True)
            for index, (features, targets) in tqdm.tqdm(
                    enumerate(train_loader), 
                    desc=f'Epoch {epoch}', 
                    total=len(train_loader)
                ):
                
                if (features != features).sum():
                    raise ValueError(features)
                
                output = model(transfer_features(features, device))
                targets = targets.to(device)
                loss = criterion(output, targets)
                if index % 50 == 0:
                    writer.add_scalar(
                        'train/loss', 
                        loss, 
                        global_step=epoch * len(train_loader) + index
                    )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # break

            # dev set evaluation
            score, results, targets, outputs, predictions = evaluate_multitask(
                model=model, 
                device=device, 
                loader=dev_loader, 
                transfer_func=transfer_features,
                task_dict=task_dict,
                output_dim=output_dim
            )
            results_df = pd.DataFrame(
                index=df_dev.index, 
                data=predictions, 
                columns=[f'{task}.pred' for task in target_column]
            )
            results_df.reset_index().to_csv(os.path.join(epoch_folder, 'dev.csv'), index=False)
            np.save(os.path.join(epoch_folder, 'outputs.npy'), outputs)
            logging_results = disaggregated_evaluation(
                df=results_df, 
                groundtruth=df_dev,
                stratify=['Country_string'],
                task_dict=task_dict
            )
            with open(os.path.join(epoch_folder, 'dev.yaml'), 'w') as fp:
                yaml.dump(logging_results, fp)
            writer.add_scalar(
                'dev/score',
                score,
                (epoch + 1) * len(train_loader)
            )
            
            for task in logging_results.keys():
                metric = task_dict[task]['score']
                logging_results[task][metric]['all'] = results[task][metric]
                writer.add_scalars(
                    f'dev/{task}/{metric}', 
                    logging_results[task][metric], 
                    (epoch + 1) * len(train_loader)
                )

            torch.save(model.cpu().state_dict(), os.path.join(
                epoch_folder, 'state.pth.tar'))

            print(f'Dev score at epoch {epoch+1}:{score}')
            # print(f'Dev results at epoch {epoch+1}:\n{yaml.dump(results)}')
            if score > max_metric:
                max_metric = score
                best_epoch = epoch
                best_state = model.cpu().state_dict()
                best_results = results.copy()

            plateau_scheduler.step(score)

        print(
            f'Best dev results found at epoch {best_epoch+1}:\n{yaml.dump(best_results)}')
        best_results['Epoch'] = best_epoch + 1
        with open(os.path.join(experiment_folder, 'dev.yaml'), 'w') as fp:
            yaml.dump(best_results, fp)
        writer.close()
    else:
        best_state = torch.load(os.path.join(
            experiment_folder, 'state.pth.tar'))
        print('Training already run')

    if not os.path.exists(os.path.join(experiment_folder, 'test_holistic.yaml')):
        model.load_state_dict(best_state)
        torch.save(best_state, os.path.join(
            experiment_folder, 'state.pth.tar'))
        score, results, targets, outputs, predictions = evaluate_multitask(
            model=model, 
            device=device, 
            loader=test_loader, 
            transfer_func=transfer_features,
            task_dict=task_dict,
            output_dim=output_dim
        )
        results_df = pd.DataFrame(
            index=df_dev.index, 
            data=predictions, 
            columns=[f'{task}.pred' for task in target_column]
        )
        print(f'Best test results:\n{yaml.dump(results)}')
        np.save(os.path.join(experiment_folder, 'targets.npy'), targets)
        np.save(os.path.join(experiment_folder, 'outputs.npy'), outputs)
        results_df.reset_index().to_csv(os.path.join(epoch_folder, 'test.csv'), index=False)
        with open(os.path.join(experiment_folder, 'test.yaml'), 'w') as fp:
            yaml.dump(results, fp)
        logging_results = disaggregated_evaluation(
            df=results_df, 
            groundtruth=df_test,
            stratify=['Country_string'],
            task_dict=task_dict
        )
        with open(os.path.join(experiment_folder, 'test_holistic.yaml'), 'w') as fp:
            yaml.dump(logging_results, fp)
    else:
        print('Evaluation already run')
