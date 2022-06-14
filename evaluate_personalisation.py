import argparse
import os
import pandas as pd
import torch
import yaml

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

from personalisation import (
    evaluate_multitask
)

from utils import (
    transfer_features
)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('EXVO Training')
    parser.add_argument(
        '--labels', 
        help='Path to test set labels', 
        required=True
    )
    parser.add_argument(
        '--model-root', 
        help='Path where model is stored', 
        required=True
    )
    parser.add_argument(
        '--results', 
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
    args = parser.parse_args()

    experiment_folder = os.path.dirname(args.results)
    os.makedirs(experiment_folder, exist_ok=True)

    df = pd.read_csv(args.labels)
    df['file'] = df['File_ID'].apply(lambda x: x.strip('[').strip(']') + '.wav')
    df.set_index('file', inplace=True)
    df_test = df.loc[df['Split'] == 'Test']
    print(df_test.info())

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

    features = pd.read_csv(args.features).set_index('file')
    features['features'] = features['features'].apply(
        lambda x: os.path.join(os.path.dirname(args.features), os.path.basename(x)))

    with open(os.path.join(args.model_root, 'hparams.yaml'), 'r') as fp:
        hparams = yaml.load(fp, Loader=yaml.Loader)
    
    if 'skip_softmax' not in hparams:
        hparams['skip_softmax'] = False

    if hparams['approach'] == 'cnn14':
        
        if hparams['auxil_net'] == 'CNN5':
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
        elif hparams['auxil_net'] == 'cnn10':
            class CNN10Auxil(Cnn10):
                def forward(self, x):
                    return self.get_embedding(x)
            subnet = CNN10Auxil(output_dim=output_dim)
            embedding_dim = 512
        else:
            raise NotImplementedError(hparams['auxil_net'])
        if hparams['fusion'] == 'style-transfer':
            main_net = FixedEmbeddingCnn14(
                output_dim=output_dim,
                embedding_dim=embedding_dim
            )
        elif hparams['fusion'] == 'attention':
            main_net = AttentionFusionCnn14(
                output_dim=output_dim,
                embedding_dim=embedding_dim,
                norm=hparams['embedding_norm'],
                softmax=not hparams['skip_softmax']
            )
        else:
            raise NotImplementedError(hparams['fusion'])
        model = ExemplarNetwork(
            main_net=main_net,
            subnet=subnet
        )
    elif args.approach == 'cnn10':
        raise NotImplementedError

    print(model)
    test_dataset = ConditioningCachedDataset(
        df_test,
        mode='evaluation',
        features=features,
        target_column=target_column
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=1,
        num_workers=0
    )
    device = args.device
    best_state = torch.load(
        os.path.join(
            args.model_root, 'state.pth.tar'
    ))
    model.load_state_dict(
        best_state,
        strict=False
    )

    _, _, _, outputs, _ = evaluate_multitask(
        model=model, 
        device=device, 
        loader=test_loader, 
        transfer_func=transfer_features,
        task_dict=task_dict,
        score=False
    )
    results_df = pd.DataFrame(
        index=df_test.index, 
        data=outputs, 
        columns=[task for task in target_column]
    )
    results_df['File_ID'] = df_test['File_ID']
    results_df.to_csv(args.results)