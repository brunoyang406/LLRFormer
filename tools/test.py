from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms

import _init_paths
from config import cfg
from config import update_config
from core.loss import JointsMSELoss
from core.function import validate
from utils.utils import create_logger

import dataset
import models
from dataset.dataloader import get_test_dataloader


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Test keypoints network')
    
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
                        help='Log directory for test logs',
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
    """Main testing/evaluation function."""
    args = parse_args()
    update_config(cfg, args)

    logger, final_output_dir, tb_log_dir = create_logger(
        cfg, args.cfg, 'valid')

    logger.info(pprint.pformat(args))
    logger.info(cfg)

    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    model = eval('models.'+cfg.MODEL.NAME+'.get_pose_net')(
        cfg, is_train=False
    )

    if cfg.TEST.MODEL_FILE:
        logger.info('=> loading model from {}'.format(cfg.TEST.MODEL_FILE))
        model.load_state_dict(torch.load(cfg.TEST.MODEL_FILE), strict=False)
    else:
        model_state_file = os.path.join(
            final_output_dir, 'model_best.pth'
        )
        logger.info('=> loading model from {}'.format(model_state_file))
        model.load_state_dict(torch.load(model_state_file), strict=False)

    if len(cfg.GPUS) > 0:
        print(f"Using GPU for testing: {cfg.GPUS}")
        model = torch.nn.DataParallel(model, device_ids=cfg.GPUS).cuda()
    else:
        print("Using CPU for testing")

    criterion = JointsMSELoss(
        use_target_weight=getattr(cfg.LOSS, 'USE_TARGET_WEIGHT', False)
    )
    if len(cfg.GPUS) > 0:
        criterion = criterion.cuda()

    valid_loader = get_test_dataloader(cfg)
    valid_dataset = valid_loader.dataset

    results = validate(cfg, valid_loader, valid_dataset, model, criterion,
             final_output_dir, tb_log_dir)

    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(log_dir=tb_log_dir)
    for k, v in results.items():
        if isinstance(v, (int, float)):
            writer.add_scalar(f'eval/{k}', v, 0)
    writer.close()


if __name__ == '__main__':
    main()
