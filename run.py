import os
import argparse
from pickletools import optimize
import random

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data import random_split
from torchvision.datasets import MNIST
from torchvision import transforms
import pytorch_lightning as pl
from torch import optim
import numpy as np
from pytorch_lightning.callbacks import RichProgressBar
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBarTheme
import mlflow.pytorch


# from model import Trainer
from batch_gen import BatchGenerator, VAS_Dataset, read_data, get_vas_dataloader, get_train_dataloader
from model import MyTransformer
from pl_model import PL_ASFormer



# TODO: Docstring for this repo.
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = 1538574472
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

    parser = argparse.ArgumentParser()
    parser.add_argument('--action', default='pred')
    parser.add_argument('--dataset', default="coffee_room")
    parser.add_argument('--split', default='4')
    parser.add_argument('--checkpoint', default='my/path/epoch=62-val_loss=1.05.ckpt')

    parser.add_argument('--features_dim', default='2048', type=int)
    parser.add_argument('--bz', default=1, type=int)
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--lr', default='0.0005', type=float)


    parser.add_argument('--num_f_maps', default='64', type=int)

    # Need input
    parser.add_argument('--num_epochs', type=int, default=150)
    # parser.add_argument('--num_layers_PG', type=int, default=11)
    # parser.add_argument('--num_layers_R', type=int, default=10)
    # parser.add_argument('--num_R', type=int, default=4)

    parser.add_argument('--patience', type=int, default=40)

    args = parser.parse_args()

    num_epochs = args.num_epochs
    features_dim = args.features_dim
    bz = args.bz
    lr = args.lr
    num_workers = args.num_workers

    num_f_maps = args.num_f_maps

    # use the full temporal resolution @ 15fps
    sample_rate = 1
    # sample input features @ 15fps instead of 30 fps
    # for 50salads, and up-sample the output to 30 fps
    if args.dataset == "50salads":
        sample_rate = 2

    vid_list_file = "../MS_TCN2/data/"+args.dataset+"/splits/train.split"+args.split+".bundle"
    vid_list_file_tst = "../MS_TCN2/data/"+args.dataset+"/splits/test.split"+args.split+".bundle"
    features_path = r"C:\Users\test\Desktop\Leon\Datasets\coffee_room\events_door_feature\x3d_m/"
    # features_path = "../MS_TCN2/data/"+args.dataset+"/features/"
    gt_path = "../MS_TCN2/data/"+args.dataset+"/groundTruth/"

    mapping_file = "../MS_TCN2/data/"+args.dataset+"/mapping.txt"

    model_dir = "./models/"+args.dataset+"/split_"+args.split
    results_dir = "./results/"+args.dataset+"/split_"+args.split

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    file_ptr = open(mapping_file, 'r')
    actions = file_ptr.read().split('\n')[:-1]
    file_ptr.close()
    actions_dict = dict()
    for a in actions:
        actions_dict[a.split()[1]] = int(a.split()[0])

    num_classes = len(actions_dict)

    patience = args.patience

    # data
    train_files = read_data(vid_list_file)
    train_loader = get_vas_dataloader(
        train_files, num_classes, actions_dict, gt_path, features_path, 
        sample_rate, bz, num_workers
    )
    test_files = read_data(vid_list_file_tst)
    valid_loader = get_vas_dataloader(
        test_files, num_classes, actions_dict, gt_path, features_path, 
        sample_rate, 1, num_workers
    )
    
    # model
    num_layers = 10
    num_f_maps = 64
    features_dim = 2048
    # bz = 1
    r1 = 2
    r2 = 2
    
    channel_mask_rate = 0.3


    # use the full temporal resolution @ 15fps
    sample_rate = 1
    # sample input features @ 15fps instead of 30 fps
    # for 50salads, and up-sample the output to 30 fps
    if args.dataset == "50salads":
        sample_rate = 2

    # To prevent over-fitting for GTEA. Early stopping & large dropout rate
    if args.dataset == "gtea":
        channel_mask_rate = 0.5

    if args.dataset == 'breakfast':
        lr = 0.0001

    model = MyTransformer(
        3, num_layers, r1, r2, num_f_maps, features_dim, num_classes, channel_mask_rate
    )
    # from ..UVAST import model as uvast_model


    # XXX: model generating factory
    optimizer_config = {
        'name': 'Adam'
    }
    lr_scheduler_config = {
        'name': 'StepLR',
        'lr': lr,
        'gamma': 0.8,
        'step_size': 10
    }

    from utils.train_utils import ASFormerLoss
    loss_config = {
        'name': ASFormerLoss.__name__,
        'custom_loss': ASFormerLoss,
        'num_class': num_classes
    }

    model = PL_ASFormer(
        model=model,
        optimizer_config=optimizer_config,
        lr_scheduler_config=lr_scheduler_config,
        loss_config=loss_config,
        # num_class=num_classes,
        action_dict=actions_dict,
        sample_rate=sample_rate,
        results_dir=''
    )
    
    # training
    # XXX: pl.Trainer arguments
    # TODO: checkpoint
    # TODO: pre-trained
    # TODO: restore
    # TODO: predict
    
    mlf_logger = MLFlowLogger(experiment_name="lightning_logs", tracking_uri="file:./ml-runs")
    mlflow.pytorch.autolog()
    

    if args.action == 'train':
        trainer = pl.Trainer(
            accelerator='gpu', 
            devices=[0], 
            precision=32,
            callbacks=[
                RichProgressBar(theme=RichProgressBarTheme(progress_bar="green")),
                EarlyStopping(monitor="val_loss", mode="min", patience=patience),
                ModelCheckpoint(
                    dirpath="my/path/", 
                    filename='{epoch}-{val_loss:.2f}', 
                    save_top_k=1, 
                    monitor="val_loss",
                    save_weights_only=True,
                ),
            ],
            max_epochs=num_epochs,
            logger=mlf_logger,
        )

        trainer.fit(
            model=model, 
            train_dataloaders=train_loader, 
            val_dataloaders=valid_loader
        )
    elif args.action == 'predict':
        # pred
        trainer = pl.Trainer(
            accelerator='gpu', 
            devices=[0], 
            precision=32,
            callbacks=[
                RichProgressBar(theme=RichProgressBarTheme(progress_bar="green")),
            ],
            logger=mlf_logger,
        )

        files = read_data(vid_list_file_tst)
        test_loader = get_vas_dataloader(
            files, num_classes, actions_dict, gt_path, features_path, 
            sample_rate, 1, num_workers
        )
        model = PL_ASFormer.load_from_checkpoint(args.checkpoint)
        model.eval()
        trainer.predict(
            model=model,
            dataloaders=test_loader
        )
    

    
if __name__ == '__main__':
    main()
    