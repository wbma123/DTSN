from data_provider.data_loader import Dataset_Custom, Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Pred, Dataset_Solar
from torch.utils.data import DataLoader

DATASET_REGISTRY = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'Solar': Dataset_Solar,
    'custom': Dataset_Custom,
}


def _resolve_loader_settings(args, flag):
    dataset_cls = DATASET_REGISTRY[args.data]
    if flag == 'pred':
        return Dataset_Pred, False, False, 1, args.freq
    if flag == 'test':
        return dataset_cls, False, True, args.batch_size, args.freq
    return dataset_cls, True, True, args.batch_size, args.freq


def data_provider(args, flag):
    dataset_cls, shuffle_flag, drop_last, batch_size, freq = _resolve_loader_settings(args, flag)
    timeenc = 0 if args.embed != 'timeF' else 1

    dataset_kwargs = dict(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq=freq,
        train_only=args.train_only,
    )
    if args.data == 'custom':
        dataset_kwargs.update(
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
        )

    dataset = dataset_cls(**dataset_kwargs)
    print(flag, len(dataset))

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last,
    )
    return dataset, data_loader
