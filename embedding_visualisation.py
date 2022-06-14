import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns

from sklearn.preprocessing import (
    StandardScaler
)
from sklearn.manifold import (
    TSNE
)

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
        '--all-checkpoints',
        default=False,
        action='store_true'
    )
    args = parser.parse_args()

    df = pd.read_csv(os.path.join(args.data_root, 'data_info.csv'))
    df['file'] = df['File_ID'].apply(lambda x: x.strip('[').strip(']') + '.wav')
    df.set_index('file', inplace=True)
    df_train = df.loc[df['Split'] == 'Train']
    df_dev = df.loc[df['Split'] == 'Val']
    df_test = df.loc[df['Split'] == 'Val']

    embeddings = [os.path.join(args.results_root, 'state_exemplar_embeddings.npy')]
    if args.all_checkpoints:
        embeddings += [
            os.path.join(args.results_root, '_exemplar_embeddings.npy')
        ] + glob.glob(os.path.join(args.results_root, '**/state_exemplar_embeddings.npy'))
    
    for emb in embeddings:
        codename = os.path.basename(emb).split('.')[0]
        if not os.path.exists(os.path.join(os.path.dirname(emb), f'{codename}_tsne.csv')):
            mapped_emb = TSNE(2).fit_transform(StandardScaler().fit_transform(np.load(emb)))
            data = pd.DataFrame(
                data=mapped_emb,
                index=df_test.index,
                columns=['TSNE_1', 'TSNE_2']
            )
            data['Subject_ID'] = df_test['Subject_ID']
            data['Country'] = df_test['Country_string']
            data.to_csv(os.path.join(os.path.dirname(emb), f'{codename}_tsne.csv'))
        else:
            data = pd.read_csv(os.path.join(os.path.dirname(emb), f'{codename}_tsne.csv'))
            data.set_index('file', inplace=True)

        data['Country'] = df_test['Country_string']
        plt.figure()
        sns.scatterplot(
            data=data,
            x='TSNE_1',
            y='TSNE_2',
            hue='Country',
            s=10,
            palette="tab10"
        )
        plt.title('Country')
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(emb), f'{codename}_tsne_country.png'))
        plt.close()

        plt.figure()
        g = sns.scatterplot(
            data=data,
            x='TSNE_1',
            y='TSNE_2',
            hue='Subject_ID',
            s=10
        )
        g.legend_.remove()
        plt.title('Subject_ID')
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(emb), f'{codename}_tsne_speaker.png'))
        plt.close()