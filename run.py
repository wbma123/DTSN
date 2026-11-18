from __future__ import annotations

import argparse
import random

import numpy as np
import torch

from exp.exp_main import Exp_Main
from utils.print_args import print_args


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)



def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='DTSN')

    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--train_only', type=bool, required=False, default=False, help='perform training on full input dataset without validation and testing')
    parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        default='DTSN',
        choices=['DTSN'],
        help='model name, options: [DTSN]',
    )
    parser.add_argument('--seed', type=int, default=2021, help='random seed')

    parser.add_argument('--data', type=str, required=True, default='ETTh1', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./dataset', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--features', type=str, default='M', help='forecasting task, options:[M, S, MS]')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h', help='freq for time features encoding')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
    parser.add_argument('--embed', type=str, default='timeF', help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--train_ratio', type=float, default=0.7, help='train split ratio for custom dataset')
    parser.add_argument('--val_ratio', type=float, default=0.1, help='validation split ratio for custom dataset')
    parser.add_argument('--test_ratio', type=float, default=0.2, help='test split ratio for custom dataset')

    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')

    parser.add_argument('--patch_len', type=int, default=16, help='patch length')
    parser.add_argument('--stride', type=int, default=8, help='stride')
    parser.add_argument('--padding_patch', default='end', help='None: None; end: padding on the end')

    parser.add_argument('--ma_type', type=str, default='ema', help='reg, ema, dema')
    parser.add_argument('--alpha', type=float, default=0.3, help='model EMA alpha')
    parser.add_argument('--beta', type=float, default=0.3, help='beta for dema in model')

    parser.add_argument('--use_gated_fusion', type=int, default=1, help='1 to enable gated residual fusion head')
    parser.add_argument('--fusion_hidden', type=int, default=64, help='hidden dim of fusion gate')
    parser.add_argument('--fusion_bias', type=float, default=0.0, help='initial bias of fusion gate')

    parser.add_argument(
        '--loss_mode',
        type=str,
        default='d_loss',
        choices=['d_loss', 'arctan_mae', 'mse', 'mae'],
        help='training objective',
    )
    parser.add_argument(
        '--dloss_alpha',
        type=float,
        default=0.2,
        help='EMA smoothing factor used inside DLoss decomposition',
    )
    parser.add_argument(
        '--dloss_beta',
        type=float,
        default=0.5,
        help='seasonal loss weight in DLoss',
    )

    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=100, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=10, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='mse', help='legacy arg, kept for compatibility')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
    parser.add_argument('--revin', type=int, default=1, help='RevIN; True 1 False 0')

    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')
    parser.add_argument('--test_flop', action='store_true', default=False, help='See utils/tools for usage')
    return parser



def experiment_name(args, iteration=0):
    return '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_{}_{}'.format(
        args.model_id, args.model, args.data, args.features,
        args.seq_len, args.label_len, args.pred_len, args.des, iteration,
    )



def main():
    parser = build_parser()
    args = parser.parse_args()
    set_seed(args.seed)

    if any(r <= 0 for r in [args.train_ratio, args.val_ratio, args.test_ratio]):
        raise ValueError('train_ratio, val_ratio and test_ratio must be positive.')
    split_sum = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(split_sum - 1.0) > 1e-6:
        raise ValueError(f'train/val/test ratios must sum to 1.0, got {split_sum:.6f}.')

    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(device_id) for device_id in device_ids]
        args.gpu = args.device_ids[0]

    print_args(args)

    exp = Exp_Main(args)
    if args.is_training:
        for ii in range(args.itr):
            setting = experiment_name(args, ii)
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)
            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)
            torch.cuda.empty_cache()
    else:
        setting = experiment_name(args)
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        torch.cuda.empty_cache()



if __name__ == '__main__':
    main()
