import torch
from torch.backends import cudnn

from backbone import EfficientDetBackbone
import cv2
import matplotlib.pyplot as plt
import numpy as np

from efficientdet.utils import BBoxTransform, ClipBoxes
from utils.utils import preprocess, invert_affine, postprocess

compound_coef = 0
force_input_size = None  # set None to use default size
img_path = "datasets/numbers_comma_minus/val/8020.jpg"  # 8017
weight_file = "weights/numbers_comma_minus/efficientdet-d0_34_8700.pth"

threshold = 0.2
iou_threshold = 0.2

use_cuda = True
use_float16 = False
cudnn.fastest = True
cudnn.benchmark = True

obj_list = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", ".", ",", "-"]

# tf bilinear interpolation is different from any other's, just make do
input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536]
input_size = (
    input_sizes[compound_coef] if force_input_size is None else force_input_size
)
ori_imgs, framed_imgs, framed_metas = preprocess(img_path, max_size=input_size)

if use_cuda:
    x = torch.stack([torch.from_numpy(fi).cuda() for fi in framed_imgs], 0)
else:
    x = torch.stack([torch.from_numpy(fi) for fi in framed_imgs], 0)

x = x.to(torch.float32 if not use_float16 else torch.float16).permute(0, 3, 1, 2)

model = EfficientDetBackbone(
    compound_coef=compound_coef,
    num_classes=len(obj_list),
    # replace this part with your project's anchor config
    ratios=[(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)],
    scales=[2**0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)],
)

model.load_state_dict(torch.load("logs/" + weight_file))
model.requires_grad_(False)
model.eval()

if use_cuda:
    model = model.cuda()
if use_float16:
    model = model.half()

with torch.no_grad():
    features, regression, classification, anchors = model(x)

    regressBoxes = BBoxTransform()
    clipBoxes = ClipBoxes()

    out = postprocess(
        x,
        anchors,
        regression,
        classification,
        regressBoxes,
        clipBoxes,
        threshold,
        iou_threshold,
    )

out = invert_affine(framed_metas, out)

for i in range(len(ori_imgs)):
    if len(out[i]["rois"]) == 0:
        continue
    ori_imgs[i] = ori_imgs[i].copy()
    for j in range(len(out[i]["rois"])):
        (x1, y1, x2, y2) = out[i]["rois"][j].astype(int)
        cv2.rectangle(ori_imgs[i], (x1, y1), (x2, y2), (255, 255, 0), 2)
        obj = obj_list[out[i]["class_ids"][j]]
        score = float(out[i]["scores"][j])

        if score < 0.9:
            continue

        print(obj, x1, y1, x2, y2, score)

        cv2.putText(
            ori_imgs[i],
            "{}, {:.3f}".format(obj, score),
            (x1, y1 + 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 0),
            1,
        )

        plt.imshow(ori_imgs[i])
