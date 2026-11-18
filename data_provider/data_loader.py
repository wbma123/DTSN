from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

from utils.timefeatures import time_features

warnings.filterwarnings('ignore')


DEFAULT_WINDOW = (24 * 4 * 4, 24 * 4, 24 * 4)
FLAG_INDEX = {'train': 0, 'val': 1, 'test': 2}


def _resolve_window_size(size):
    if size is None:
        return DEFAULT_WINDOW
    return size[0], size[1], size[2]


def _resolve_split_index(flag):
    assert flag in ['train', 'test', 'val']
    return FLAG_INDEX[flag]


def _select_feature_frame(df_raw, features, target):
    if features in ('M', 'MS'):
        cols_data = df_raw.columns[1:]
        return df_raw[cols_data]
    if features == 'S':
        return df_raw[[target]]
    raise ValueError(f'Unsupported features mode: {features}')


def _build_time_stamp(df_stamp, timeenc, freq, include_minute=False):
    stamp = df_stamp.copy()
    stamp['date'] = pd.to_datetime(stamp['date'])
    if timeenc == 0:
        stamp['month'] = stamp['date'].dt.month
        stamp['day'] = stamp['date'].dt.day
        stamp['weekday'] = stamp['date'].dt.weekday
        stamp['hour'] = stamp['date'].dt.hour
        if include_minute:
            stamp['minute'] = stamp['date'].dt.minute
            stamp['minute'] = stamp['minute'].map(lambda x: x // 15)
        return stamp.drop(columns=['date']).values
    if timeenc == 1:
        data_stamp = time_features(pd.to_datetime(stamp['date'].values), freq=freq)
        return data_stamp.transpose(1, 0)
    raise ValueError(f'Unsupported time encoding mode: {timeenc}')


class _BaseForecastDataset(Dataset):
    def _set_calendar_window(self, df_raw, df_data, starts, ends, include_minute=False):
        start, end = starts[self.set_type], ends[self.set_type]
        if self.scale:
            # Fit scaling statistics on the training split only.
            train_data = df_data[starts[0]:ends[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][start:end]
        data_stamp = _build_time_stamp(df_stamp, self.timeenc, self.freq, include_minute=include_minute)
        self.data_x = data[start:end]
        self.data_y = data[start:end]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_ETT_hour(_BaseForecastDataset):
    def __init__(
        self,
        root_path,
        flag='train',
        size=None,
        features='S',
        data_path='ETTh1.csv',
        target='OT',
        scale=True,
        timeenc=0,
        freq='h',
        train_only=None,
    ):
        del train_only
        self.seq_len, self.label_len, self.pred_len = _resolve_window_size(size)
        self.set_type = _resolve_split_index(flag)

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]

        df_data = _select_feature_frame(df_raw, self.features, self.target)
        self._set_calendar_window(
            df_raw, df_data, border1s, border2s, include_minute=False,
        )


class Dataset_ETT_minute(_BaseForecastDataset):
    def __init__(
        self,
        root_path,
        flag='train',
        size=None,
        features='S',
        data_path='ETTm1.csv',
        target='OT',
        scale=True,
        timeenc=0,
        freq='t',
        train_only=False,
    ):
        del train_only
        self.seq_len, self.label_len, self.pred_len = _resolve_window_size(size)
        self.set_type = _resolve_split_index(flag)

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]

        df_data = _select_feature_frame(df_raw, self.features, self.target)
        self._set_calendar_window(
            df_raw, df_data, border1s, border2s, include_minute=True,
        )


class Dataset_Custom(_BaseForecastDataset):
    def __init__(
        self,
        root_path,
        flag='train',
        size=None,
        features='S',
        data_path='ETTh1.csv',
        target='OT',
        scale=True,
        timeenc=0,
        freq='h',
        train_only=False,
        train_ratio=0.7,
        val_ratio=0.1,
        test_ratio=0.2,
    ):
        self.seq_len, self.label_len, self.pred_len = _resolve_window_size(size)
        self.set_type = _resolve_split_index(flag)

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.train_only = train_only
        self.train_ratio = float(train_ratio)
        self.val_ratio = float(val_ratio)
        self.test_ratio = float(test_ratio)
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        cols = list(df_raw.columns)
        if self.features == 'S':
            cols.remove(self.target)
        cols.remove('date')

        if self.train_only:
            num_train = len(df_raw)
            num_test = 0
            num_vali = 0
        else:
            ratio_sum = self.train_ratio + self.val_ratio + self.test_ratio
            if abs(ratio_sum - 1.0) > 1e-6:
                raise ValueError(f'Custom split ratios must sum to 1.0, got {ratio_sum:.6f}')
            num_train = int(len(df_raw) * self.train_ratio)
            num_test = int(len(df_raw) * self.test_ratio)
            num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]

        if self.features in ('M', 'MS'):
            df_raw = df_raw[['date'] + cols]
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_raw = df_raw[['date'] + cols + [self.target]]
            df_data = df_raw[[self.target]]
        else:
            raise ValueError(f'Unsupported features mode: {self.features}')

        self._set_calendar_window(
            df_raw, df_data, border1s, border2s, include_minute=False,
        )


class Dataset_Solar(Dataset):
    def __init__(
        self,
        root_path,
        flag='train',
        size=None,
        features='S',
        data_path='ETTh1.csv',
        target='OT',
        scale=True,
        timeenc=0,
        freq='h',
        train_only=False,
    ):
        del train_only
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        self.set_type = _resolve_split_index(flag)

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        rows = []
        with open(os.path.join(self.root_path, self.data_path), 'r', encoding='utf-8') as f:
            for line in f.readlines():
                line = line.strip('\n').split(',')
                data_line = np.stack([float(item) for item in line])
                rows.append(data_line)

        df_raw = np.stack(rows, 0)
        df_raw = pd.DataFrame(df_raw)

        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_valid = int(len(df_raw) * 0.1)
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_valid, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        df_data = df_raw.values
        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data)
            data = self.scaler.transform(df_data)
        else:
            data = df_data

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = torch.zeros((seq_x.shape[0], 1))
        seq_y_mark = torch.zeros((seq_x.shape[0], 1))
        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Pred(Dataset):
    def __init__(
        self,
        root_path,
        flag='pred',
        size=None,
        features='S',
        data_path='ETTh1.csv',
        target='OT',
        scale=True,
        inverse=False,
        timeenc=0,
        freq='15min',
        cols=None,
        train_only=False,
    ):
        del train_only
        self.seq_len, self.label_len, self.pred_len = _resolve_window_size(size)
        assert flag in ['pred']

        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols = cols
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        if self.cols:
            cols = self.cols.copy()
        else:
            cols = list(df_raw.columns)
            self.cols = cols.copy()
            cols.remove('date')
        if self.features == 'S':
            cols.remove(self.target)

        border1 = len(df_raw) - self.seq_len
        border2 = len(df_raw)

        if self.features in ('M', 'MS'):
            df_raw = df_raw[['date'] + cols]
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_raw = df_raw[['date'] + cols + [self.target]]
            df_data = df_raw[[self.target]]
        else:
            raise ValueError(f'Unsupported features mode: {self.features}')

        if self.scale:
            self.scaler.fit(df_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        tmp_stamp = df_raw[['date']][border1:border2].copy()
        tmp_stamp['date'] = pd.to_datetime(tmp_stamp['date'])
        pred_dates = pd.date_range(tmp_stamp['date'].values[-1], periods=self.pred_len + 1, freq=self.freq)

        df_stamp = pd.DataFrame(columns=['date'])
        df_stamp['date'] = list(tmp_stamp['date'].values) + list(pred_dates[1:])
        self.future_dates = list(pred_dates[1:])
        data_stamp = _build_time_stamp(df_stamp, self.timeenc, self.freq, include_minute=True)

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = self.data_x[r_begin:r_begin + self.label_len]
        else:
            seq_y = self.data_y[r_begin:r_begin + self.label_len]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
