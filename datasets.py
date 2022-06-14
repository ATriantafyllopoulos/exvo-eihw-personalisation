import audiofile
import audtorch
import numpy as np
import pandas as pd
import torch

from utils import LabelEncoder


class CachedDataset(torch.utils.data.Dataset):
    r"""Dataset of cached features.

    Args:
        df: partition dataframe containing labels
        features: dataframe with paths to features
        target_column: column to find labels in (in df)
        transform: function used to process features
        target_transform: function used to process labels
    """

    def __init__(
        self, 
        df: pd.DataFrame,
        features: pd.DataFrame,
        target_column: str, 
        transform=None, 
        target_transform=None,
        personalisation: bool = False
    ):
        self.df = df
        self.features = features
        self.target_column = target_column
        self.transform = transform
        self.target_transform = target_transform
        self.indices = list(self.df.index)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):
        index = self.indices[item]
        signal = np.load(self.features.loc[index, 'features'] + '.npy')
        target = self.df[self.target_column].loc[index]
        if signal.shape[0] == 2:
            signal = signal.mean(0, keepdims=True)
        if isinstance(self.target_column, list) and len(self.target_column) > 1:
            target = np.array(target.values)

        if self.transform is not None:
            signal = self.transform(signal)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return signal, target



class ConditioningCachedDataset(torch.utils.data.Dataset):
    r"""Dataset of cached features.

    Args:
        df: partition dataframe containing labels
        features: dataframe with paths to features
        target_column: column to find labels in (in df)
        transform: function used to process features
        target_transform: function used to process labels
        condition_column: column to find conditions in
        num_exemplars: number of examples to return
        mode: denoting if data is used for ``training`` or ``validation``.
            This results in the exemplars being picked at random.
            Else, the first exemplars are picked.
    """

    def __init__(
        self, 
        df: pd.DataFrame,
        features: pd.DataFrame,
        target_column: str, 
        transform=None, 
        target_transform=None,
        condition_column: str = 'Subject_ID',
        num_exemplars: int = 2,
        mode: str = 'training'
    ):
        self.df = df
        self.features = features
        self.target_column = target_column
        self.transform = transform
        self.target_transform = target_transform
        self.indices = list(self.df.index)
        self.condition_column = condition_column
        self.num_exemplars = num_exemplars
        self.mode = mode
        if mode == 'training':
            # when in training
            # transform conditions to integers
            # so that they can be used for auxiliary losses
            encoder = LabelEncoder(self.df[condition_column].unique())
            self.df[condition_column] = self.df[condition_column].apply(encoder.encode)
        # this builds a 'hash' table with all the indices 
        # corresponding to each different manifestation
        # of the condition variable
        # excluding rows where the target variable is missing
        # (to account for test predictions)
        self.hash = self.df.groupby(self.condition_column).apply(
            # lambda df: df.loc[~df[target_column].isna()].index -> does not work for multitasking
            # lambda df: df.dropna().index
            lambda df: df.loc[df[target_column].isna().sum(axis=1).apply(lambda x: x==0)].index
        )

    def __len__(self):
        return len(self.df)

    def _get_index(self, index, exemplars: bool = False):
        signal = np.load(self.features.loc[index, 'features'] + '.npy')
        target = self.df[self.target_column].loc[index]
        if signal.shape[0] == 2:
            signal = signal.mean(0, keepdims=True)
        if isinstance(self.target_column, list) and len(self.target_column) > 1:
            target = np.array(target.values)
        
        if self.transform is not None:
            signal = self.transform(signal)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if exemplars:
            condition = self.df[self.condition_column].loc[index]
            if self.mode == 'training':
                exemplar_indices = np.random.choice(
                    self.hash[condition], 
                    size=self.num_exemplars, 
                    replace=False
                )
            elif self.mode == 'evaluation':
                exemplar_indices = self.hash[condition][:self.num_exemplars]
            else:
                raise NotImplementedError(self.mode)
            return signal, target, exemplar_indices, condition
        else:
            return signal, target

    def __getitem__(self, item):
        index = self.indices[item]

        signal, target, exemplar_indices, condition = self._get_index(index, True)

        exemplars = []
        exemplar_targets = []
        for index in exemplar_indices:
            x, y = self._get_index(index)
            exemplars.append(x)
            exemplar_targets.append(y)
        if self.mode == 'evaluation':
            max_exemplar = max([x.shape[-2] for x in exemplars])
            transform = audtorch.transforms.Expand(max_exemplar, axis=-2)
            exemplars = [transform(x) for x in exemplars]
        exemplars = np.concatenate(exemplars)
        exemplar_targets = np.stack(exemplar_targets)

        return signal, target, exemplars, exemplar_targets, condition
        


class WavDataset(torch.utils.data.Dataset):
    r"""Dataset of raw audio data.

    Args:
        df: partition dataframe containing labels
        features: dataframe with paths to features
        target_column: column to find labels in (in df)
        transform: function used to process features
        target_transform: function used to process labels
    """

    def __init__(
        self, 
        df: pd.DataFrame,
        features: pd.DataFrame,
        target_column: str, 
        transform=None, 
        target_transform=None,
    ):
        self.df = df
        self.features = features
        self.target_column = target_column
        self.transform = transform
        self.target_transform = target_transform
        self.indices = list(self.df.index)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):
        index = self.indices[item]
        signal = audiofile.read(index, always_2d=False)[0]
        target = self.df[self.target_column].loc[index]
        if isinstance(self.target_column, list) and len(self.target_column) > 1:
            target = np.array(target.values)

        if self.transform is not None:
            signal = self.transform(signal)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return signal, target