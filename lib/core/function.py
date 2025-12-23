from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
 
import time
import logging
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import torch
import torch.nn as nn

from core.evaluate import accuracy
from core.inference import get_final_preds
from utils.transforms import flip_back
from utils.vis import save_debug_images, save_debug_images_with_circles
import matplotlib.pyplot as plt
from PIL import Image
import io


logger = logging.getLogger(__name__)


def train(config, train_loader, model, criterion, optimizer, epoch,
          output_dir, tb_log_dir, writer_dict, medical_training=None):
    """Training function for one epoch."""
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    model.train()

    end = time.time()
    for i, (input, target, meta) in enumerate(train_loader):
        data_time.update(time.time() - end)

        outputs = model(input)
        
        if isinstance(outputs, tuple):
            output, attention_losses = outputs
        else:
            output = outputs
            attention_losses = []

        if torch.cuda.is_available() and len(config.GPUS) > 0:
            target = target.cuda(non_blocking=True)
        else:
            target = target

        if isinstance(outputs, list):
            loss = criterion(outputs[0], target)
            for output in outputs[1:]:
                loss += criterion(output, target)
        else:
            loss = criterion(output, target)

        if attention_losses:
            attn_loss = sum(attention_losses)
            loss = loss + 0.1 * attn_loss

        if medical_training is not None:
            loss, _ = medical_training.train_step(input, target, meta, epoch)

        optimizer.zero_grad()
        loss.backward()
        
        if hasattr(config.TRAIN, 'GRADIENT_CLIP') and config.TRAIN.GRADIENT_CLIP > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.TRAIN.GRADIENT_CLIP)
            
        optimizer.step()

        losses.update(loss.item(), input.size(0))

        _, avg_acc, cnt, pred = accuracy(output.detach().cpu().numpy(),
                                         target.detach().cpu().numpy())
        acc.update(avg_acc, cnt)

        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
                  'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      speed=input.size(0)/batch_time.val,
                      data_time=data_time, loss=losses, acc=acc)
            logger.info(msg)

            writer = writer_dict['writer']
            global_steps = writer_dict['train_global_steps']
            writer.add_scalar('train_loss', losses.val, global_steps)
            writer.add_scalar('train_acc', acc.val, global_steps)
            writer_dict['train_global_steps'] = global_steps + 1

            prefix = '{}_{}'.format(os.path.join(output_dir, 'train'), i)
            save_debug_images_with_circles(config, input, meta, target, pred*4, output,
                              prefix)


def validate(config, val_loader, val_dataset, model, criterion, output_dir,
             tb_log_dir, writer_dict=None):
    """Validation function."""
    batch_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    model.eval()

    num_samples = len(val_dataset)
    all_preds = np.zeros(
        (num_samples, config.MODEL.NUM_JOINTS, 3),
        dtype=np.float32
    )
    all_boxes = np.zeros((num_samples, 6))
    image_path = []
    filenames = []
    imgnums = []
    idx = 0
    with torch.no_grad():
        end = time.time()
        for i, (input, target, meta) in enumerate(val_loader):
            outputs = model(input)
            if isinstance(outputs, list):
                output = outputs[-1]
            else:
                output = outputs

            if torch.cuda.is_available() and len(config.GPUS) > 0:
                target = target.cuda(non_blocking=True)
            else:
                target = target
            loss = criterion(output, target)

            num_images = input.size(0)
            losses.update(loss.item(), num_images)
            _, avg_acc, cnt, pred = accuracy(output.cpu().numpy(),
                                             target.cpu().numpy())

            acc.update(avg_acc, cnt)

            batch_time.update(time.time() - end)
            end = time.time()

            def to_numpy2d(meta_list):
                arr = []
                for x in meta_list:
                    if hasattr(x, 'cpu'):
                        x = x.cpu().numpy()
                    x = np.array(x)
                    arr.append(x.reshape(-1).tolist())
                return np.array(arr, dtype=np.float32)

            c = to_numpy2d(meta['center'])
            s = to_numpy2d(meta['scale'])
            c = np.array(c)
            if c.shape != (num_images, 2):
                c = c.reshape(num_images, 2)
            s = np.array(s)
            if s.shape != (num_images, 2):
                s = s.reshape(num_images, 2)
            score = np.ones((num_images,), dtype=np.float32)
            preds, maxvals = get_final_preds(
                config, output.clone().cpu().numpy(), c, s)

            all_preds[idx:idx + num_images, :, 0:2] = preds[:, :, 0:2]
            all_preds[idx:idx + num_images, :, 2:3] = maxvals
            all_boxes[idx:idx + num_images, 0:2] = c[:, 0:2]
            all_boxes[idx:idx + num_images, 2:4] = s[:, 0:2]
            all_boxes[idx:idx + num_images, 4] = np.prod(s*200, 1)
            all_boxes[idx:idx + num_images, 5] = score
            image_path.extend(meta['image'])

            idx += num_images

            if i % config.PRINT_FREQ == 0:
                msg = 'Test: [{0}/{1}]\t' \
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t' \
                      'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                          i, len(val_loader), batch_time=batch_time,
                          loss=losses, acc=acc)
                logger.info(msg)

                prefix = '{}_{}'.format(
                    os.path.join(output_dir, 'val'), i
                )
                save_debug_images_with_circles(config, input, meta, target, pred*4, output,
                                  prefix)

        name_values = val_dataset.evaluate(
            all_preds, output_dir, all_boxes, image_path,
            filenames, imgnums
        )
        perf_indicator = name_values['AUC'] if 'AUC' in name_values else list(name_values.values())[0]

        model_name = config.MODEL.NAME
        if isinstance(name_values, list):
            for name_value in name_values:
                _print_name_value(name_value, model_name)
        else:
            _print_name_value(name_values, model_name)

        if writer_dict:
            writer = writer_dict['writer']
            global_steps = writer_dict['valid_global_steps']
            scalar_name_values = {k: v for k, v in dict(name_values).items() if isinstance(v, (int, float, np.floating))}
            summary_keys = ['AUC', 'NME', 'MED', 'PCK@0.05']
            for k in summary_keys:
                if k in scalar_name_values:
                    tag = f'valid/summary/{k}'.replace('@', '_')
                    writer.add_scalar(tag, scalar_name_values[k], global_steps)
            for k in scalar_name_values:
                if k.startswith('PCK@0.05_'):
                    tag = f'valid/region/{k}'.replace('@', '_')
                    writer.add_scalar(tag, scalar_name_values[k], global_steps)
            for k in scalar_name_values:
                if k.startswith('CCC_'):
                    tag = f'valid/ccc/{k}'.replace('@', '_')
                    writer.add_scalar(tag, scalar_name_values[k], global_steps)
            for k in scalar_name_values:
                if not (k in summary_keys or k.startswith('PCK@0.05_') or k.startswith('CCC_')):
                    tag = f'valid/other/{k}'.replace('@', '_')
                    writer.add_scalar(tag, scalar_name_values[k], global_steps)
            if 'CCC_per_point' in name_values:
                for idx, d in enumerate(name_values['CCC_per_point']):
                    if isinstance(d, dict):
                        if 'x' in d and isinstance(d['x'], (int, float)):
                            tag = f'valid/per_point/CCC_x_{idx}'.replace('@', '_')
                            writer.add_scalar(tag, d['x'], global_steps)
                        if 'y' in d and isinstance(d['y'], (int, float)):
                            tag = f'valid/per_point/CCC_y_{idx}'.replace('@', '_')
                            writer.add_scalar(tag, d['y'], global_steps)
            writer_dict['valid_global_steps'] = global_steps + 1

        try:
            add_image_to_tensorboard(writer, 'ccc_boxplot', f'{output_dir}/ccc_boxplot.png', global_steps)
            add_image_to_tensorboard(writer, 'ccc_barplot', f'{output_dir}/ccc_barplot.png', global_steps)
        except Exception as e:
            print(f"[TensorBoard Image] Failed to write: {e}")

    return perf_indicator


def _print_name_value(name_value, full_arch_name):
    """Print evaluation metrics in markdown table format."""
    names = name_value.keys()
    values = name_value.values()
    num_values = len(name_value)
    logger.info(
        '| Arch ' +
        ' '.join(['| {}'.format(name) for name in names]) +
        ' |'
    )
    logger.info('|---' * (num_values+1) + '|')

    if len(full_arch_name) > 15:
        full_arch_name = full_arch_name[:8] + '...'
    logger.info(
        '| ' + full_arch_name + ' ' +
        ' '.join([
            '| {:.3f}'.format(value) if isinstance(value, (int, float, np.floating)) else f'| {str(value)}'
            for value in values
        ]) +
         ' |'
    )

    if 'CCC_per_point' in name_value:
        ccc_list = name_value['CCC_per_point']
        print("Per-point CCC:")
        print("Idx\tCCC_x\tCCC_y")
        for idx, d in enumerate(ccc_list):
            print(f"{idx}\t{d['x']:.3f}\t{d['y']:.3f}")


class AverageMeter(object):
    """Computes and stores the average and current value."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0


def add_image_to_tensorboard(writer, tag, image_path, global_step):
    """Add image to TensorBoard."""
    img = Image.open(image_path)
    img_np = np.array(img)
    writer.add_image(tag, img_np.transpose(2, 0, 1), global_step)
