from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import argparse
import pprint
import shutil

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter

import _init_paths
from config import cfg
from config import update_config
from core.loss import JointsMSELoss, JointsWeightedMSELoss
from core.function import train
from core.function import validate
from utils.utils import get_optimizer
from utils.utils import save_checkpoint
from utils.utils import create_logger
from utils.utils import get_model_summary

import dataset
import models
from dataset.dataloader import get_train_dataloader, get_val_dataloader


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train keypoints network')
    
    parser.add_argument('--cfg',
                        help='Experiment configuration file path',
                        required=True,
                        type=str)
    parser.add_argument('opts',
                        help="Modify config options",
                        default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument('--modelDir',
                        help='Output directory for model checkpoints',
                        type=str,
                        default='')
    parser.add_argument('--logDir',
                        help='Log directory for training logs',
                        type=str,
                        default='')
    parser.add_argument('--dataDir',
                        help='Data directory root',
                        type=str,
                        default='')
    parser.add_argument('--prevModelDir',
                        help='Previous model directory (deprecated)',
                        type=str,
                        default='')
    parser.add_argument('--gpus',
                        help='GPU IDs to use',
                        type=str,
                        default='0')

    args = parser.parse_args()
    return args


def main():
    """Main training function."""
    args = parse_args()
    update_config(cfg, args)

    logger, final_output_dir, tb_log_dir = create_logger(
        cfg, args.cfg, 'train')

    logger.info(pprint.pformat(args))
    logger.info(cfg)

    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    model = eval('models.'+cfg.MODEL.NAME+'.get_pose_net')(
        cfg, is_train=True
    )

    this_dir = os.path.dirname(__file__)
    shutil.copy2(
        os.path.join(this_dir, '../lib/models', cfg.MODEL.NAME + '.py'),
        final_output_dir)

    writer_dict = {
        'writer': SummaryWriter(log_dir=tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    dump_input = torch.rand(
        (1, 3, cfg.MODEL.IMAGE_SIZE[1], cfg.MODEL.IMAGE_SIZE[0])
    )
    logger.info(get_model_summary(model, dump_input))

    if len(cfg.GPUS) > 0:
        print(f"Using GPU for training: {cfg.GPUS}")
        print(f"Number of GPU devices: {torch.cuda.device_count()}")
        print(f"Current GPU: {torch.cuda.current_device()}")
        print(f"GPU name: {torch.cuda.get_device_name(0)}")
        model = torch.nn.DataParallel(model).cuda()
    else:
        print("Using CPU for training")

    if getattr(cfg.TRAIN, 'JOINT_WEIGHTS', False):
        num_joints = cfg.MODEL.NUM_JOINTS
        joint_weights = [1.0] * num_joints
        
        hip_weight = getattr(cfg.TRAIN, 'HIP_WEIGHT', 1.5)
        femur_weight = getattr(cfg.TRAIN, 'FEMUR_WEIGHT', 2.5)
        knee_weight = getattr(cfg.TRAIN, 'KNEE_WEIGHT', 1.0)
        tibia_weight = getattr(cfg.TRAIN, 'TIBIA_WEIGHT', 2.0)
        ankle_weight = getattr(cfg.TRAIN, 'ANKLE_WEIGHT', 1.3)
        
        if hasattr(cfg.TRAIN, 'FEMUR_TIBIA_WEIGHT'):
            femur_weight = cfg.TRAIN.FEMUR_TIBIA_WEIGHT
            tibia_weight = cfg.TRAIN.FEMUR_TIBIA_WEIGHT
        
        for i in range(0, 10):
            joint_weights[i] = hip_weight
        for i in range(10, 18):
            joint_weights[i] = femur_weight
        for i in range(18, 26):
            joint_weights[i] = knee_weight
        for i in range(26, 34):
            joint_weights[i] = tibia_weight
        for i in range(34, num_joints):
            joint_weights[i] = ankle_weight
            
        criterion = JointsWeightedMSELoss(use_target_weight=False, joint_weights=joint_weights)
        logger.info(f"Using region-weighted loss function - Hip:{hip_weight}, Femur:{femur_weight}, Knee:{knee_weight}, Tibia:{tibia_weight}, Ankle:{ankle_weight}")
    else:
        criterion = JointsMSELoss(use_target_weight=False)
    
    if len(cfg.GPUS) > 0:
        criterion = criterion.cuda()

    train_loader = get_train_dataloader(cfg)
    valid_loader = get_val_dataloader(cfg)
    valid_dataset = valid_loader.dataset

    best_perf = 0
    best_model = False
    last_epoch = -1
    optimizer = get_optimizer(cfg, model)
    begin_epoch = cfg.TRAIN.BEGIN_EPOCH

    if getattr(cfg.TRAIN, 'LR_SCHEDULER', 'MultiStepLR') == 'ReduceLROnPlateau':
        patience = getattr(cfg.TRAIN, 'PATIENCE', 10)
        min_lr = getattr(cfg.TRAIN, 'MIN_LR', 1e-7)
        factor = getattr(cfg.TRAIN, 'LR_FACTOR', 0.5)
        
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='max',
            factor=factor,
            patience=patience,
            verbose=True,
            min_lr=min_lr
        )
    else:
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, cfg.TRAIN.LR_STEP, cfg.TRAIN.LR_FACTOR,
            last_epoch=last_epoch
        ) if cfg.TRAIN.LR_SCHEDULER == 'MultiStepLR' else torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, cfg.TRAIN.END_EPOCH, eta_min=cfg.TRAIN.LR_END, last_epoch=last_epoch)


    for epoch in range(begin_epoch, cfg.TRAIN.END_EPOCH):
        if hasattr(lr_scheduler, 'get_last_lr'):
            current_lr = lr_scheduler.get_last_lr()[0]
        else:
            current_lr = optimizer.param_groups[0]['lr']
        logger.info("=> current learning rate is {:.6f}".format(current_lr))

        train(cfg, train_loader, model, criterion, optimizer, epoch,
              final_output_dir, tb_log_dir, writer_dict)

        name_value = validate(
            cfg, valid_loader, valid_dataset, model, criterion,
            final_output_dir, tb_log_dir, writer_dict
        )
        if isinstance(name_value, dict):
            if 'PCK@0.02' in name_value:
                perf_indicator = name_value['PCK@0.02']
                logger.info(f"Using PCK@0.02 as learning rate scheduler monitoring metric: {perf_indicator:.4f}")
            elif 'AUC' in name_value:
                perf_indicator = name_value['AUC']
                logger.info(f"Using AUC as learning rate scheduler monitoring metric: {perf_indicator:.4f}")
            else:
                perf_indicator = list(name_value.values())[0]
                logger.info(f"Using default metric as learning rate scheduler monitoring metric: {perf_indicator:.4f}")
        else:
            perf_indicator = float(name_value)
            logger.info(f"Using numeric metric as learning rate scheduler monitoring metric: {perf_indicator:.4f}")
        
        print(f"Epoch {epoch} validation metrics:")
        if isinstance(name_value, dict):
            for k, v in name_value.items():
                print(f"  {k}: {v:.4f}")
        else:
            print(f"  value: {name_value:.4f}")
        
        if isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            lr_scheduler.step(perf_indicator)
        else:
            lr_scheduler.step()


        if perf_indicator >= best_perf:
            best_perf = perf_indicator
            best_model = True
        else:
            best_model = False

        logger.info('=> saving checkpoint to {}'.format(final_output_dir))
        save_checkpoint({
            'epoch': epoch + 1,
            'model': cfg.MODEL.NAME,
            'state_dict': model.state_dict(),
            'best_state_dict': model.state_dict(),
            'perf': perf_indicator,
            'optimizer': optimizer.state_dict(),
        }, best_model, final_output_dir)

    final_model_state_file = os.path.join(
        final_output_dir, 'final_state.pth'
    )
    logger.info('=> saving final model state to {}'.format(
        final_model_state_file)
    )
    torch.save(model.state_dict(), final_model_state_file)
    writer_dict['writer'].close()


if __name__ == '__main__':
    main()
