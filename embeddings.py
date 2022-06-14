import argparse
import audmetric
import audtorch
import glob
import numpy as np
import os
import pandas as pd
import random
import shutil
import torch
import tqdm
import yaml

from define import (
    EMOTIONS,
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser('EXVO Subnet Embedding Extraction')
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
        '--approach',
        default='cnn14',
        choices=[
            'cnn14'
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
        '--fusion',
        default='style-transfer',
        choices=[
            'style-transfer',
            'attention'
        ]
    )
    parser.add_argument(
        '--source',
        default='exemplar',
        choices=[
            'main',
            'exemplar'
        ]
    )
    parser.add_argument(
        '--auxil-learn-condition',
        action='store_true',
        default=False,
        help='Train auxiliary model to learn condition'
    )
    parser.add_argument(
        '--adversarial',
        action='store_true',
        default=False
    )
    parser.add_argument(
        '--all-checkpoints',
        default=False,
        action='store_true'
    )
    parser.add_argument(
        '--multitask',
        action='store_true',
        default=False
    )
    args = parser.parse_args()

    output_dim = 10
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
            embedding_dim=embedding_dim
        )
    else:
        raise NotImplementedError(args.fusion)
    
    df = pd.read_csv(os.path.join(args.data_root, 'data_info.csv'))
    df['file'] = df['File_ID'].apply(lambda x: x.strip('[').strip(']') + '.wav')
    df.set_index('file', inplace=True)
    df_train = df.loc[df['Split'] == 'Train']
    df_dev = df.loc[df['Split'] == 'Val']
    df_test = df.loc[df['Split'] == 'Val']

    features = pd.read_csv(args.features).set_index('file')
    features['features'] = features['features'].apply(
        lambda x: os.path.join(os.path.dirname(args.features), os.path.basename(x)))


    print(f"Using {args.fusion} fusion with {args.auxil_net} auxiliary subnet")
    model = ExemplarNetwork(
        main_net=main_net,
        subnet=subnet,
        adversarial_exemplars=args.adversarial,
        auxil_conditions=len(df_train['Subject_ID'].unique()) if args.auxil_learn_condition else None
    )
    print(model)

    test_dataset = ConditioningCachedDataset(
        df_test,
        mode='evaluation',
        features=features,
        target_column=EMOTIONS
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=1,
        num_workers=2
    )
    device = args.device

    states = [os.path.join(args.results_root, 'state.pth.tar')]

    if args.all_checkpoints:
        states += [
            os.path.join(args.results_root, 'initial.pth.tar')
        ] + glob.glob(os.path.join(args.results_root, '**/state.pth.tar'))
    
    for state in states:
        codename = os.path.basename(state).split('.')[0]
        if not os.path.exists(os.path.join(os.path.dirname(state), f'{codename}_{args.source}_embeddings.npy')):
            model.load_state_dict(torch.load(state), strict=False)
            model.eval()
            model.to(device)
            embeddings = np.zeros((len(test_loader), 2048 if args.source == 'main' else model.main_net.embedding_dim))
            if args.multitask:
                outputs = np.zeros((len(test_loader), model.main_net.output_dim))
                targets = np.zeros((len(test_loader), model.main_net.output_dim))
            with torch.no_grad():
                for counter, (x, y, _, _, _) in tqdm.tqdm(
                    enumerate(test_loader),
                    total=len(test_loader),
                    desc=state
                ):
                    x = x.to(device)
                    if args.source == 'exemplar':
                        emb = model.subnet(x).squeeze(-1).squeeze(-1)
                        embeddings[counter, :] = emb.cpu()
                        if args.multitask:
                            outputs[counter, :] = model.exemplar_out(emb).cpu()
                            targets[counter, :] = y
                    elif args.source == 'main':
                        assert args.fusion == 'attention'
                        emb = model.main_net.get_embedding(x)
                        embeddings[counter, :] = emb.cpu()
            np.save(os.path.join(os.path.dirname(state), f'{codename}_{args.source}_embeddings.npy'), embeddings)
        else:
            outputs = np.load(os.path.join(os.path.dirname(state), f'{codename}_{args.source}_outputs.npy'))
            if args.source == 'exemplar':
                targets = np.load(os.path.join(os.path.dirname(state), f'targets.npy'))
        if args.multitask and args.source == 'exemplar':
            np.save(os.path.join(os.path.dirname(state), f'{codename}_{args.source}_outputs.npy'), outputs)
            ccc = 0
            for e_p, e_t in zip(outputs.T, targets.T):
                ccc += audmetric.concordance_cc(e_t, e_p)
            ccc /= outputs.shape[1]
            with open(os.path.join(os.path.dirname(state), f'{codename}_{args.source}_outputs.txt'), 'w') as fp:
                fp.write(f'Score: {ccc:.4f}')