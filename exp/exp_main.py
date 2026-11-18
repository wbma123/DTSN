from __future__ import annotations

import os
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
from torch import optim

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.losses import DLoss, ArctanMAELoss
from utils.metrics import metric
from utils.tools import EarlyStopping, adjust_learning_rate
from utils.tools import visual

warnings.filterwarnings('ignore')


class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super().__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        return data_provider(self.args, flag)

    def _select_optimizer(self):
        return optim.AdamW(self.model.parameters(), lr=self.args.learning_rate)

    def _select_criterion(self):
        mode = str(self.args.loss_mode).lower()
        if mode == 'd_loss':
            train_criterion = DLoss(alpha=self.args.dloss_alpha, beta=self.args.dloss_beta)
        elif mode == 'arctan_mae':
            train_criterion = ArctanMAELoss()
        elif mode == 'mse':
            train_criterion = nn.MSELoss()
        elif mode == 'mae':
            train_criterion = nn.L1Loss()
        else:
            raise ValueError(f'Unsupported loss_mode: {self.args.loss_mode}')
        return train_criterion.to(self.device), nn.MSELoss().to(self.device)

    def _prepare_batch(self, batch_x, batch_y):
        return batch_x.float().to(self.device), batch_y.float().to(self.device)

    def _forecast(self, batch_x: torch.Tensor, batch_y: torch.Tensor):
        outputs = self.model(batch_x)
        f_dim = -1 if self.args.features == 'MS' else 0
        outputs = outputs[:, -self.args.pred_len:, f_dim:]
        target = batch_y[:, -self.args.pred_len:, f_dim:]
        return outputs, target

    def vali(self, vali_data, vali_loader, criterion, is_test=True):
        del vali_data
        losses = []
        self.model.eval()

        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, batch_y_mark in vali_loader:
                del batch_x_mark, batch_y_mark
                batch_x, batch_y = self._prepare_batch(batch_x, batch_y)

                pred, true = self._forecast(batch_x, batch_y)
                if is_test:
                    loss = nn.functional.mse_loss(pred, true)
                else:
                    loss = criterion(pred, true)

                losses.append(loss.item())

        self.model.train()
        return float(np.average(losses))

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')
        del train_data, test_data

        path = os.path.join(self.args.checkpoints, setting)
        os.makedirs(path, exist_ok=True)

        time_now = time.time()
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        model_optim = self._select_optimizer()
        train_criterion, test_criterion = self._select_criterion()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                del batch_x_mark, batch_y_mark
                iter_count += 1
                model_optim.zero_grad()

                batch_x, batch_y = self._prepare_batch(batch_x, batch_y)

                pred, true = self._forecast(batch_x, batch_y)
                loss = train_criterion(pred, true)
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = float(np.average(train_loss))
            vali_loss = self.vali(vali_data, vali_loader, train_criterion, is_test=False)
            test_loss = self.vali(None, test_loader, test_criterion, is_test=True)

            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test MSE: {4:.7f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss, test_loss
                )
            )

            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = os.path.join(path, 'checkpoint.pth')
        self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))
        if os.path.exists(best_model_path):
            os.remove(best_model_path)

        return self.model

    def test(self, setting, test=0):
        _, test_loader = self._get_data(flag='test')

        if test:
            print('loading model')
            checkpoint = os.path.join('./checkpoints/' + setting, 'checkpoint.pth')
            self.model.load_state_dict(torch.load(checkpoint, map_location=self.device))

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        os.makedirs(folder_path, exist_ok=True)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                del batch_x_mark, batch_y_mark
                batch_x, batch_y = self._prepare_batch(batch_x, batch_y)

                outputs, target = self._forecast(batch_x, batch_y)
                pred = outputs.detach().cpu().numpy()
                true = target.detach().cpu().numpy()

                preds.append(pred)
                trues.append(true)

                if i % 20 == 0:
                    input_series = batch_x.detach().cpu().numpy()
                    gt = np.concatenate((input_series[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input_series[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.array(preds)
        trues = np.array(trues)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])

        mae, mse = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))

        with open('DTSN-result.txt', 'a') as f:
            f.write(setting + '  \n')
            f.write('mse:{}, mae:{}'.format(mse, mae))
            f.write('\n\n')

        return
