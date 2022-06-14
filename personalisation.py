import argparse
import audmetric
import audtorch
import numpy as np
import os
import pandas as pd
import random
import shutil
import torch
import tqdm
import yaml


from torch.utils.tensorboard import SummaryWriter


from define import (
    EMOTIONS,
    TASK_DICT
)

from datasets import (
    ConditioningCachedDataset
)

from exemplar_models import (
    ExemplarNetwork,
    FixedEmbeddingCnn14,
    AttentionFusionCnn14
)

from models import (
    Cnn10,
    Cnn14
)

from utils import (
    disaggregated_evaluation,
    evaluate_multitask,
    transfer_features,
    CCCLoss
)


def fix_index(df, root):
    df.reset_index(inplace=True)
    df['filename'] = df['filename'].apply(lambda x: os.path.join(args.data_root, x))
    df.set_index('filename', inplace=True)
    return df

def evaluate_multitask(
    model,
    device, 
    loader,
    task_dict,
    transfer_func,
    score: bool = False
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

    outputs = torch.zeros((len(loader.dataset), model.output_dim))
    if score:
        targets = torch.zeros((len(loader.dataset), len(task_dict)))
    with torch.no_grad():
        for index, (features, target, exemplars, _, _) in tqdm.tqdm(
            enumerate(loader),
            desc='Batch',
            total=len(loader),
            disable=not score
        ):
            start_index = index * loader.batch_size
            end_index = (index + 1) * loader.batch_size
            if end_index > len(loader.dataset):
                end_index = len(loader.dataset)
            outputs[start_index:end_index, :] = model(
                (transfer_func(features, device), transfer_func(exemplars, device)))
            if score:
                targets[start_index:end_index] = target
            # break

    outputs = outputs.cpu().numpy()
    if not score:
        return _, _, _, outputs, _

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
    else:
        scores = [emo_score] + [x for x, y in zip(total_score, task_dict) if y not in EMOTIONS]
        total_score = len(scores) / sum([1 / (score + 1e-9) for score in scores])
    
    return total_score, results, targets, outputs, predictions


if __name__ == '__main__':
    parser = argparse.ArgumentParser('EXVO Training')
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
        '--approach',
        default='cnn10',
        choices=[
            'cnn14',
            'cnn10'
        ]
    )
    parser.add_argument(
        '--auxil-net',
        default='CNN5',
        choices=[
            'CNN5',
            'cnn10'
        ]
    )
    parser.add_argument(
        '--embedding-norm',
        default=None,
        choices=[
            'LayerNorm',
            'BatchNorm'
        ]
    )
    parser.add_argument(
        '--fusion',
        default='style-transfer',
        choices=[
            'style-transfer',
            'attention'
        ]
    )
    parser.add_argument(
        '--ignore-auxil',
        default=False,
        action='store_true'
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
        '--seed',
        type=int,
        default=0
    )
    parser.add_argument(
        '--optimizer',
        default='SGD',
        choices=[
            'SGD',
            'Adam',
            'AdamW',
            'RMSprop'
        ]
    )
    parser.add_argument(
        '--emo-loss',
        default='CCC',
        choices=[
            'CCC',
            'MSE',
        ]
    )
    parser.add_argument(
        '--adversarial',
        action='store_true',
        default=False
    )
    parser.add_argument(
        '--skip-softmax',
        action='store_true',
        default=False
    )
    parser.add_argument(
        '--multitask',
        action='store_true',
        default=False
    )
    parser.add_argument(
        '--auxil-learn-condition',
        action='store_true',
        default=False,
        help='Train auxiliary model to learn condition'
    )
    parser.add_argument(
        '--main-forget-condition',
        action='store_true',
        default=False,
        help='Train main model to forget condition (i.e. adversarial)'
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
    df_train = df.loc[df['Split'] == 'Train']
    df_dev = df.loc[df['Split'] == 'Val']
    df_test = df.loc[df['Split'] == 'Val']
        
    target_column = EMOTIONS

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

    if args.emo_loss == 'MSE':
        criterion = torch.nn.MSELoss()
    elif args.emo_loss == 'CCC':
        criterion = CCCLoss()

    features = pd.read_csv(args.features).set_index('file')
    features['features'] = features['features'].apply(
        lambda x: os.path.join(os.path.dirname(args.features), os.path.basename(x)))
    
    db_args = {
        'features': features,
        'target_column': target_column
    }
    if args.approach == 'cnn14':
        
        if args.auxil_net == 'CNN5':
            subnet = torch.nn.Sequential(
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
                torch.nn.AdaptiveAvgPool2d((1, 1))
            )
            embedding_dim = 1024
        elif args.auxil_net == 'cnn10':
            class CNN10Auxil(Cnn10):
                def forward(self, x):
                    return self.get_embedding(x)
            subnet = CNN10Auxil(output_dim=output_dim)
            embedding_dim = 512
        else:
            raise NotImplementedError(args.auxil_net)
        if args.fusion == 'style-transfer':
            main_net = FixedEmbeddingCnn14(
                output_dim=output_dim,
                embedding_dim=embedding_dim
            )
        elif args.fusion == 'attention':
            main_net = AttentionFusionCnn14(
                output_dim=output_dim,
                embedding_dim=embedding_dim,
                norm=args.embedding_norm,
                softmax=not args.skip_softmax
            )
        else:
            raise NotImplementedError(args.fusion)
        model = ExemplarNetwork(
            main_net=Cnn14(output_dim=output_dim, return_embedding=True) if args.ignore_auxil else main_net,
            subnet=subnet,
            adversarial_exemplars=args.adversarial,
            auxil_conditions=len(df_train['Subject_ID'].unique()) if args.auxil_learn_condition else None,
            main_conditions=len(df_train['Subject_ID'].unique()) if args.main_forget_condition else None,
            ignore_auxil=args.ignore_auxil
        )
        db_class = ConditioningCachedDataset
        db_args['transform'] = audtorch.transforms.RandomCrop(250, axis=-2)
        # model.to_yaml(os.path.join(experiment_folder, 'model.yaml'))
    elif args.approach == 'cnn10':
        raise NotImplementedError
        model = FixedEmbeddingCnn14(
            output_dim=output_dim
        )
        db_class = ConditioningCachedDataset
        db_args['transform'] = audtorch.transforms.RandomCrop(250, axis=-2)
        # model.to_yaml(os.path.join(experiment_folder, 'model.yaml'))
    
    if args.state is not None:
        print('Loading old state to continue training...')
        initial_state = torch.load(args.state)
        model.load_state_dict(
            initial_state,
            strict=False
        )

    print(model)
    print("Starting data loading...")
    # exit()
    train_dataset = db_class(
        df_train,
        **db_args
    )
    print('Testing dataset...')
    x, y, e, et, c = train_dataset[0]
    print(f'Input shape: {x.shape}')
    print(f'Output shape: {y.shape}')
    print(f'Exemplar shape: {e.shape}')
    print(f'Exemplar target shape: {et.shape}')
    print(f'Condition: {c.shape}')
    # exit()
    db_args.pop('transform')

    dev_dataset = db_class(
        df_dev,
        mode='evaluation',
        **db_args
    )

    test_dataset = db_class(
        df_test,
        mode='evaluation',
        **db_args
    )
    # create DataLoaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.batch_size,
        num_workers=2
    )

    print('Testing data loader...')
    x, y, e, et, c = next(iter(train_loader))
    print(f'Input shape: {x.shape}')
    print(f'Output shape: {y.shape}')
    print(f'Exemplar shape: {e.shape}')
    print(f'Exemplar target shape: {et.shape}')
    print(f'Condition shape: {c.shape}')

    # this ensures that whatever reshaping happens
    # (to run exemplars through encoder)
    # does not mess up the order of exemplars
    assert torch.equal(
        e,
        e.view(-1, e.shape[2], e.shape[3]).view(e.shape)
    )

    dev_loader = torch.utils.data.DataLoader(
        dev_dataset,
        shuffle=False,
        batch_size=1,
        num_workers=2
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=1,
        num_workers=2
    )
    # print('Finished')
    # exit()
    device = args.device
    if not os.path.exists(os.path.join(experiment_folder, 'state.pth.tar')):

        with open(os.path.join(experiment_folder, 'hparams.yaml'), 'w') as fp:
            yaml.dump(vars(args), fp)

        writer = SummaryWriter(log_dir=os.path.join(experiment_folder, 'log'))

        torch.save(
            model.state_dict(), 
            os.path.join(
            experiment_folder, 
            'initial.pth.tar')
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
        elif args.optimizer == 'AdamW':
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=args.learning_rate,
                weight_decay=0.0001
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
            for index, (features, targets, exemplars, exemplar_targets, conditions) in tqdm.tqdm(
                    enumerate(train_loader), 
                    desc=f'Epoch {epoch}', 
                    total=len(train_loader),
                    disable=True
                ):
                
                if (features != features).sum():
                    raise ValueError(features)
                
                output, exemplar_output = model(
                    (
                        transfer_features(features, device),
                        transfer_features(exemplars, device)
                    )
                )
                if args.main_forget_condition:
                    output, condition_output = output
                    main_auxil_loss = torch.nn.CrossEntropyLoss()(
                        condition_output, 
                        conditions.to(device)
                    )
                    writer.add_scalar(
                        'train/main_auxil_loss', 
                        main_auxil_loss, 
                        global_step=epoch * len(train_loader) + index
                    )

                loss = criterion(output, targets.to(device))
                if index % 50 == 0:
                    writer.add_scalar(
                        'train/loss', 
                        loss, 
                        global_step=epoch * len(train_loader) + index
                    )
                if args.main_forget_condition:
                    loss = (loss + main_auxil_loss) / 2

                if args.multitask:
                    if args.auxil_learn_condition:
                        # assumes that network returns a tuple
                        # with first output being task (i.e. emotion)
                        # and second output being condition (i.e. speaker ID)
                        exemplar_condition = exemplar_output[1]
                        exemplar_output = exemplar_output[0]
                    exemplar_targets = exemplar_targets.view(-1, targets.shape[1])
                    exemplar_loss = criterion(exemplar_output, exemplar_targets.to(device))
                    if index % 50 == 0:
                        writer.add_scalar(
                            'train/exemplar_loss', 
                            exemplar_loss, 
                            global_step=epoch * len(train_loader) + index
                        )
                    if args.auxil_learn_condition:
                        conditions = torch.repeat_interleave(conditions, 2)
                        auxil_loss = torch.nn.CrossEntropyLoss()(
                            exemplar_condition, 
                            conditions.to(device)
                        )
                        writer.add_scalar(
                            'train/auxil_loss', 
                            auxil_loss, 
                            global_step=epoch * len(train_loader) + index
                        )
                        # auxil loss starts with values near 6
                        # exemplar loss is < 1 (as it's a CCC loss)
                        # auxil_loss was previously multiplied by 0.1
                        exemplar_loss = (exemplar_loss + auxil_loss) / 2  # TODO: might need balancing
                    loss = (loss + exemplar_loss) / 2  # TODO: this might require balancing
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            # TODO: adversarial loss requires scheduling
            if args.adversarial or args.main_forget_condition:
                if epoch < 10:
                    pass
                elif epoch < 60:
                    model.lambd = (epoch - 10) / (50)
                else:
                    model.lambd = 1
                print(f'Lambda scheduling: {model.lambd}')
            # break

            # dev set evaluation
            score, results, targets, outputs, predictions = evaluate_multitask(
                model=model, 
                device=device, 
                loader=dev_loader, 
                transfer_func=transfer_features,
                task_dict=task_dict
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
            task_dict=task_dict
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
