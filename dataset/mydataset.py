import os
import json
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

class MyKeypointDataset(Dataset):
    def __init__(self, cfg, root, set_name, is_train, transform=None):
        self.image_dir = os.path.join(root, set_name, 'images')
        self.ann_dir = os.path.join(root, set_name, 'annotations')
        self.transform = transform
        self.is_train = is_train
        self.cfg = cfg
        
        # Dataset configuration
        self.color_rgb = getattr(cfg.DATASET, 'COLOR_RGB', True)
        self.flip = getattr(cfg.DATASET, 'FLIP', False)
        self.scale_factor = getattr(cfg.DATASET, 'SCALE_FACTOR', 0.0)
        self.rot_factor = getattr(cfg.DATASET, 'ROT_FACTOR', 0.0)
        self.prob_half_body = getattr(cfg.DATASET, 'PROB_HALF_BODY', 0.0)
        self.num_joints_half_body = getattr(cfg.DATASET, 'NUM_JOINTS_HALF_BODY', 8)
        self.medical_augmentation = getattr(cfg.DATASET, 'MEDICAL_AUGMENTATION', False)
        
        # Supported image formats
        data_format = getattr(cfg.DATASET, 'DATA_FORMAT', 'jpeg').lower()
        if data_format == 'jpeg':
            self.image_extensions = ['.jpeg', '.jpg', '.JPG', '.JPEG']
        elif data_format == 'png':
            self.image_extensions = ['.png', '.PNG']
        else:
            # Support multiple formats if not specified
            self.image_extensions = ['.jpeg', '.jpg', '.png', '.JPG', '.JPEG', '.PNG']
        
        # Load all image and annotation file pairs
        self.samples = []
        for fname in os.listdir(self.image_dir):
            # Check if file has supported image extension
            if any(fname.endswith(ext) for ext in self.image_extensions):
                # Find corresponding annotation file
                base_name = os.path.splitext(fname)[0]
                ann_path = os.path.join(self.ann_dir, base_name + '.json')
                if os.path.exists(ann_path):
                    self.samples.append((os.path.join(self.image_dir, fname), ann_path))
        
        # Initialize data augmentation transforms
        if self.medical_augmentation and is_train:
            from lib.utils.medical_transforms import MedicalImageTransforms
            self.medical_transform = MedicalImageTransforms(
                is_train=True, 
                image_size=(384, 1152)
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        import torch
        img_path, ann_path = self.samples[idx]
        
        # Load image (BGR format by default)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Failed to read image: {img_path}")
            img = np.zeros((576, 192, 3), dtype=np.uint8)
        
        # Convert grayscale to 3-channel if needed
        if img.ndim == 2:
            img = np.stack([img]*3, axis=-1)
        
        # Convert BGR to RGB if configured
        if hasattr(self, 'color_rgb') and self.color_rgb:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Load annotation file
        with open(ann_path, 'r', encoding='utf-8') as f:
            ann = json.load(f)
        
        # Parse keypoints (point and circle shapes)
        keypoints = []
        for shape in ann['shapes']:
            if shape['shape_type'] == 'point':
                keypoints.append(shape['points'][0])
            elif shape['shape_type'] == 'circle':
                # Use circle center (first point)
                keypoints.append(shape['points'][0])
        keypoints = np.array(keypoints, dtype=np.float32)
        
        # Get number of joints from config
        num_joints = getattr(self.cfg.MODEL, 'NUM_JOINTS', 36)

        # Resize and pad image to target size (384x1152)
        target_w, target_h = 384, 1152
        orig_h, orig_w = img.shape[:2]
        scale_w = target_w / orig_w
        scale_h = target_h / orig_h
        scale = min(scale_w, scale_h)
        new_w, new_h = int(orig_w * scale), int(orig_h * scale)
        resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        input_img = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        pad_left = (target_w - new_w) // 2
        pad_top = (target_h - new_h) // 2
        input_img[pad_top:pad_top+new_h, pad_left:pad_left+new_w, :] = resized_img
        
        # Transform keypoints accordingly and clip to boundaries
        keypoints[:, 0] = np.clip(keypoints[:, 0] * scale + pad_left, 0, target_w - 1)
        keypoints[:, 1] = np.clip(keypoints[:, 1] * scale + pad_top, 0, target_h - 1)

        # Apply data augmentation
        if self.medical_augmentation and self.is_train:
            input_img, keypoints = self.medical_transform(input_img, keypoints)
        elif self.transform:
            input_img = self.transform(input_img)  # [3, H, W]

        # Generate target heatmaps and visibility weights
        heatmap_size = [288, 96]  # [height, width], matching model output
        sigma = getattr(self.cfg.MODEL, 'SIGMA', 8)  # Get sigma from config
        # Target heatmap format: [num_joints, height, width]
        target = np.zeros((num_joints, heatmap_size[0], heatmap_size[1]), dtype=np.float32)
        target_weight = np.ones((num_joints, 1), dtype=np.float32)

        for joint_id in range(num_joints):
            # Skip zero-padded keypoints
            if np.all(keypoints[joint_id] == 0):
                continue
            # Map coordinates to heatmap space
            # heatmap_size = [288, 96] means [height, width]
            mu_x = int(keypoints[joint_id][0] * heatmap_size[1] / target_w)  # x maps to width
            mu_y = int(keypoints[joint_id][1] * heatmap_size[0] / target_h)  # y maps to height
            if mu_x < 0 or mu_y < 0 or mu_x >= heatmap_size[1] or mu_y >= heatmap_size[0]:
                continue
            tmp_size = sigma * 3
            ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
            br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]

            if ul[0] >= heatmap_size[1] or ul[1] >= heatmap_size[0] or br[0] < 0 or br[1] < 0:
                continue

            size = 2 * tmp_size + 1
            x = np.arange(0, size, 1, np.float32)
            y = x[:, np.newaxis]
            x0 = y0 = size // 2
            g = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

            g_x = max(0, -ul[0]), min(br[0], heatmap_size[1]) - ul[0]
            g_y = max(0, -ul[1]), min(br[1], heatmap_size[0]) - ul[1]
            img_x = max(0, ul[0]), min(br[0], heatmap_size[1])
            img_y = max(0, ul[1]), min(br[1], heatmap_size[0])

            target[joint_id, img_y[0]:img_y[1], img_x[0]:img_x[1]] = np.maximum(
                target[joint_id, img_y[0]:img_y[1], img_x[0]:img_x[1]],
                g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
            )

        joints_vis = np.ones((num_joints, 1), dtype=np.float32)  # All visible
        meta = {
            'image': img_path,
            'joints': keypoints,  # Transformed keypoints
            'center': [target_w / 2, target_h / 2],
            'scale': [target_w / 200, target_h / 200],
            'joints_vis': joints_vis,
        }

        target = torch.from_numpy(target)

        return input_img, target, meta

    def evaluate(self, preds, *args, **kwargs):
        """
        Evaluate model predictions.
        
        Args:
            preds: [N, num_joints, 2], predicted keypoint coordinates (after resize+padding)
        
        Returns:
            OrderedDict containing evaluation metrics: MED, NME, PCK@0.02, AUC, CCC, etc.
        """
        from collections import OrderedDict
        import numpy as np
        import json
        import time

        # Start timing
        start_time = time.time()
        
        # 1. Collect ground truth keypoints
        gts = []
        for img_path, ann_path in self.samples:
            with open(ann_path, 'r', encoding='utf-8') as f:
                ann = json.load(f)
            keypoints = []
            for shape in ann['shapes']:
                if shape['shape_type'] == 'point':
                    keypoints.append(shape['points'][0])
                elif shape['shape_type'] == 'circle':
                    # Use circle center, consistent with training
                    keypoints.append(shape['points'][0])
            
            # Ensure keypoint count matches model configuration
            num_joints = getattr(self.cfg.MODEL, 'NUM_JOINTS', 36)
            if len(keypoints) != num_joints:
                # Pad or truncate if count doesn't match
                if len(keypoints) < num_joints:
                    # Pad with zero keypoints
                    while len(keypoints) < num_joints:
                        keypoints.append([0.0, 0.0])
                else:
                    # Truncate excess keypoints
                    keypoints = keypoints[:num_joints]
            
            # Apply resize+padding transformation to GT keypoints
            img = cv2.imread(img_path)  # BGR or grayscale
            if img.ndim == 2:
                img = np.stack([img]*3, axis=-1)
            if hasattr(self, 'color_rgb') and self.color_rgb:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            orig_h, orig_w = img.shape[:2]
            target_w, target_h = 384, 1152
            scale = min(target_w / orig_w, target_h / orig_h)
            new_w, new_h = int(orig_w * scale), int(orig_h * scale)
            pad_w = target_w - new_w
            pad_h = target_h - new_h
            pad_left = pad_w // 2
            pad_top = pad_h // 2
            for i in range(len(keypoints)):
                keypoints[i][0] = keypoints[i][0] * scale + pad_left
                keypoints[i][1] = keypoints[i][1] * scale + pad_top
            
            gts.append(keypoints)
        gts = np.array(gts, dtype=np.float32)
        preds = np.array(preds, dtype=np.float32)

        dists = np.linalg.norm(preds[..., :2] - gts, axis=2)

        med = np.mean(dists)

        norm = np.sqrt(384**2 + 1152**2)
        nme = np.mean(dists / norm)

        # 5. PCK@0.02 (Percentage of Correct Keypoints at threshold 0.02)
        pck_thr = 0.01  # Threshold set to 1% (stricter)
        pck = np.mean((dists / norm) < pck_thr)

        # Region-wise PCK (for 36 keypoints)
        region_dict = {
            'hip': list(range(0, 8)),         # Hip (0-7)
            'femur': list(range(8, 12)),      # Femur (8-11)
            'knee': list(range(12, 28)),      # Knee (12-27)
            'tibia': list(range(28, 32)),    # Tibia (28-31)
            'ankle': list(range(32, 36)),     # Ankle (32-35)
        }
        for region, idxs in region_dict.items():
            region_dists = dists[:, idxs]
            region_pck = np.mean((region_dists / norm) < pck_thr)
            results_key = f'PCK@0.02_{region}'
            if 'region_pck_dict' not in locals():
                region_pck_dict = {}
            region_pck_dict[results_key] = region_pck

        # 6. AUC (Area Under PCK Curve, 0~0.5)
        pck_curve = []
        thresholds = np.linspace(0, 0.5, 51)
        for thr in thresholds:
            pck_curve.append(np.mean((dists / norm) < thr))
        auc = np.trapz(pck_curve, thresholds) / 0.5  # Normalized to [0,1]

        # === CCC (Concordance Correlation Coefficient) Calculation ===
        from scipy.stats import pearsonr
        def concordance_correlation_coefficient(y_true, y_pred):
            # Check if input is constant
            if np.var(y_true) == 0 or np.var(y_pred) == 0:
                return 0.0  # Return 0 if either input is constant
            
            mean_true = np.mean(y_true)
            mean_pred = np.mean(y_pred)
            var_true = np.var(y_true)
            var_pred = np.var(y_pred)
            
            try:
                cor = pearsonr(y_true, y_pred)[0]
                if np.isnan(cor):
                    return 0.0
            except:
                return 0.0
                
            ccc = (2 * cor * np.sqrt(var_true) * np.sqrt(var_pred)) / (var_true + var_pred + (mean_true - mean_pred) ** 2 + 1e-8)
            return ccc if not np.isnan(ccc) else 0.0

        # Calculate CCC per keypoint (x/y)
        ccc_per_point = []
        for i in range(gts.shape[1]):
            ccc_x = concordance_correlation_coefficient(gts[:, i, 0], preds[:, i, 0])
            ccc_y = concordance_correlation_coefficient(gts[:, i, 1], preds[:, i, 1])
            ccc_per_point.append({'x': ccc_x, 'y': ccc_y})

        # Calculate CCC for all keypoints combined (x/y)
        ccc_x_all = concordance_correlation_coefficient(gts[..., 0].flatten(), preds[..., 0].flatten())
        ccc_y_all = concordance_correlation_coefficient(gts[..., 1].flatten(), preds[..., 1].flatten())

        # === p-value Calculation ===
        from scipy.stats import ttest_1samp
        try:
            p_value = ttest_1samp(dists.flatten(), 0).pvalue
            if np.isnan(p_value):
                p_value = 0.0
        except:
            p_value = 0.0

        results = OrderedDict([
            ('MED', med),
            ('NME', nme),
            ('PCK@0.02', pck),
            ('AUC', auc),
            ('CCC_x_all', ccc_x_all),
            ('CCC_y_all', ccc_y_all),
            ('CCC_per_point', ccc_per_point),
            ('p-value', p_value),
        ])
        # Add region-wise PCK to results
        if 'region_pck_dict' in locals():
            for k, v in region_pck_dict.items():
                results[k] = v

        ccc_x_list = [item['x'] for item in ccc_per_point]
        ccc_y_list = [item['y'] for item in ccc_per_point]
        mean_x, std_x = np.mean(ccc_x_list), np.std(ccc_x_list)
        mean_y, std_y = np.mean(ccc_y_list), np.std(ccc_y_list)
        min_x, max_x = np.min(ccc_x_list), np.max(ccc_x_list)
        min_y, max_y = np.min(ccc_y_list), np.max(ccc_y_list)

        # Save visualization plots
        if 'output_dir' in kwargs and kwargs['output_dir']:
            outdir = kwargs['output_dir']
            # Box plot
            plt.figure(figsize=(6,4))
            plt.boxplot([ccc_x_list, ccc_y_list], labels=['CCC_x', 'CCC_y'])
            plt.title('Per-keypoint CCC Distribution')
            plt.ylabel('CCC')
            plt.savefig(f'{outdir}/ccc_boxplot.png', dpi=150)
            plt.close()
            # Bar plot
            plt.figure(figsize=(10,4))
            plt.bar(range(len(ccc_x_list)), ccc_x_list, label='CCC_x')
            plt.bar(range(len(ccc_y_list)), ccc_y_list, label='CCC_y', alpha=0.5)
            plt.xlabel('Keypoint Index')
            plt.ylabel('CCC')
            plt.legend()
            plt.title('CCC per Keypoint')
            plt.savefig(f'{outdir}/ccc_barplot.png', dpi=150)
            plt.close()

        results['CCC_x_mean'] = mean_x
        results['CCC_x_std'] = std_x
        results['CCC_x_min'] = min_x
        results['CCC_x_max'] = max_x
        results['CCC_y_mean'] = mean_y
        results['CCC_y_std'] = std_y
        results['CCC_y_min'] = min_y
        results['CCC_y_max'] = max_y

        # === End timing ===
        end_time = time.time()
        total_time = end_time - start_time
        num_images = len(self.samples)
        avg_time_per_img = total_time / num_images if num_images > 0 else 0
        fps = num_images / total_time if total_time > 0 else 0
        results['inference_total_time_sec'] = total_time
        results['inference_avg_time_per_img_ms'] = avg_time_per_img * 1000
        results['inference_fps'] = fps

        return results
