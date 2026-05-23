#!/usr/bin/env python3
"""
ST-SLAM 4.0 SuperPoint Feature Extractor
Generates .superpoint feature files for ST-SLAM to read.
Usage: python3 superpoint_infer.py <rgb_dir> <output_dir>

Output format (binary .sp file per image):
  num_keypoints (int32)
  for each keypoint: x(float32) y(float32) score(float32) descriptor(256*bool packed as 32 bytes)
"""

import numpy as np
import cv2
import os
import sys
import struct
from pathlib import Path

try:
    import torch
except ImportError:
    print("ERROR: PyTorch not installed. Run: pip install torch")
    sys.exit(1)

class SuperPoint:
    """Minimal SuperPoint implementation in PyTorch."""
    def __init__(self, device='cpu'):
        self.device = device
        self.nms_dist = 4
        self.conf_thresh = 0.015
        self.cell_size = 8
        self.border = 0

    def load_model(self, weights_path=None):
        """Load pretrained SuperPoint model or use default."""
        self.net = self._build_network().to(self.device)
        if weights_path and os.path.exists(weights_path):
            self.net.load_state_dict(torch.load(weights_path, map_location=self.device))
        else:
            print("INFO: No weights provided, using randomly initialized net (demo only)")
        self.net.eval()
        return self

    def _build_network(self):
        import torch.nn as nn
        import torch.nn.functional as F
        class SPNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.relu = nn.ReLU(inplace=True)
                self.pool = nn.MaxPool2d(2, 2)
                c1, c2, c3, c4, c5 = 64, 64, 128, 128, 256
                self.conv1a = nn.Conv2d(1, c1, 3, 1, 1)
                self.conv1b = nn.Conv2d(c1, c1, 3, 1, 1)
                self.conv2a = nn.Conv2d(c1, c2, 3, 1, 1)
                self.conv2b = nn.Conv2d(c2, c2, 3, 1, 1)
                self.conv3a = nn.Conv2d(c2, c3, 3, 1, 1)
                self.conv3b = nn.Conv2d(c3, c3, 3, 1, 1)
                self.conv4a = nn.Conv2d(c3, c4, 3, 1, 1)
                self.conv4b = nn.Conv2d(c4, c4, 3, 1, 1)
                self.convPa = nn.Conv2d(c4, c5, 3, 1, 1)
                self.convPb = nn.Conv2d(c5, 65, 1, 1, 0)
                self.convDa = nn.Conv2d(c4, c5, 3, 1, 1)
                self.convDb = nn.Conv2d(c5, 256, 1, 1, 0)
            def forward(self, x):
                x = self.relu(self.conv1a(x))
                x = self.relu(self.conv1b(x))
                x = self.pool(x)
                x = self.relu(self.conv2a(x))
                x = self.relu(self.conv2b(x))
                x = self.pool(x)
                x = self.relu(self.conv3a(x))
                x = self.relu(self.conv3b(x))
                x = self.pool(x)
                x = self.relu(self.conv4a(x))
                x = self.relu(self.conv4b(x))
                semi = self.convPb(self.relu(self.convPa(x)))
                desc = self.convDb(self.relu(self.convDa(x)))
                return semi, desc
        return SPNet()

    def extract(self, image_gray):
        """Extract SuperPoint features from grayscale image."""
        if isinstance(image_gray, str):
            image_gray = cv2.imread(image_gray, cv2.IMREAD_GRAYSCALE)
            if image_gray is None:
                raise ValueError(f"Cannot read image: {image_gray}")

        h, w = image_gray.shape
        inp = torch.from_numpy(image_gray.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0).to(self.device)

        with torch.no_grad():
            semi, desc = self.net(inp)

        # Convert semi-dense scores to keypoints
        semi_np = semi[0].cpu().numpy()
        dense = np.exp(semi_np) / np.sum(np.exp(semi_np), axis=0)[np.newaxis, :, :]
        prob = dense[0:-1, :, :]
        prob = np.max(prob, axis=0)

        # NMS
        kps = []
        scores = []
        for y in range(self.border, h - self.border, self.cell_size):
            for x in range(self.border, w - self.border, self.cell_size):
                patch = prob[max(0,y-self.nms_dist):min(h,y+self.nms_dist+1),
                             max(0,x-self.nms_dist):min(w,x+self.nms_dist+1)]
                center_val = prob[y, x]
                if center_val >= self.conf_thresh and center_val >= np.max(patch):
                    kps.append([float(x), float(y)])
                    scores.append(float(center_val))

        if not kps:
            return np.zeros((0, 2), dtype=np.float32), np.zeros((0, 256), dtype=np.uint8)

        kps = np.array(kps, dtype=np.float32)
        scores = np.array(scores, dtype=np.float32)

        # Extract descriptors
        desc_np = desc[0].cpu().numpy()
        num_kp = len(kps)
        desc_out = np.zeros((num_kp, 256), dtype=np.uint8)
        for i in range(num_kp):
            x, y = kps[i]
            xi, yi = int(round(x / self.cell_size)), int(round(y / self.cell_size))
            xi = min(max(xi, 0), desc_np.shape[2]-1)
            yi = min(max(yi, 0), desc_np.shape[1]-1)
            d = desc_np[:, yi, xi]
            desc_out[i] = (d > 0).astype(np.uint8)

        return kps, desc_out, scores


def write_sp_file(path, kps, descs, scores):
    """Write .sp file in ST-SLAM compatible format."""
    with open(path, 'wb') as f:
        n = len(kps)
        f.write(struct.pack('i', n))
        for i in range(n):
            f.write(struct.pack('ff', kps[i, 0], kps[i, 1]))
            f.write(struct.pack('f', scores[i] if scores is not None else 1.0))
            if i < len(descs):
                f.write(descs[i].tobytes())
            else:
                f.write(b'\x00' * 32)


def main():
    if len(sys.argv) < 3:
        print("Usage: python3 superpoint_infer.py <rgb_dir> <output_dir>")
        sys.exit(1)

    rgb_dir = sys.argv[1]
    out_dir = sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    extractor = SuperPoint(device=device)
    extractor.load_model()

    # Process all PNG/JPG files sorted by timestamp
    exts = ('.png', '.jpg', '.jpeg')
    files = sorted([f for f in os.listdir(rgb_dir) if f.lower().endswith(exts)])

    for i, fname in enumerate(files):
        img_path = os.path.join(rgb_dir, fname)
        out_path = os.path.join(out_dir, Path(fname).stem + '.sp')

        if os.path.exists(out_path):
            continue

        try:
            kps, descs, scores = extractor.extract(img_path)
            write_sp_file(out_path, kps, descs, scores)
            print(f"  [{i+1}/{len(files)}] {fname}: {len(kps)} features")
        except Exception as e:
            print(f"  [{i+1}/{len(files)}] {fname}: ERROR {e}")

    print(f"Done. {len(files)} images processed -> {out_dir}")


if __name__ == '__main__':
    main()
