import argparse
import datetime
import os
import traceback

import numpy as np
import torch
import yaml
from tensorboardX import SummaryWriter
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm.autonotebook import tqdm

from backbone import EfficientDetBackbone
from efficientdet.dataset import CocoDataset, Resizer, Normalizer, Augmenter, collater
from efficientdet.loss import FocalLoss

import logging
import math
from torch.nn.init import _calculate_fan_in_and_fan_out, _no_grad_normal_
from glob import glob

# adapted from https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch/tree/master by Dominik Lindorfer


def boolean_string(s):
    if s not in {"False", "True"}:
        raise ValueError("Not a valid boolean string")
    return s == "True"


def variance_scaling_(tensor, gain=1.0):
    # type: (Tensor, float) -> Tensor
    r"""
    initializer for SeparableConv in Regressor/Classifier
    reference: https://keras.io/zh/initializers/  VarianceScaling
    """
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = math.sqrt(gain / float(fan_in))

    return _no_grad_normal_(tensor, 0.0, std)


def get_last_weights(weights_path):
    weights_path = glob(weights_path + f"/*.pth")
    weights_path = sorted(
        weights_path, key=lambda x: int(x.rsplit("_")[-1].rsplit(".")[0]), reverse=True
    )[0]
    logging.info(f"using weights {weights_path}")
    return weights_path


def init_weights(model):
    for name, module in model.named_modules():
        is_conv_layer = isinstance(module, nn.Conv2d)

        if is_conv_layer:
            if "conv_list" or "header" in name:
                variance_scaling_(module.weight.data)
            else:
                nn.init.kaiming_uniform_(module.weight.data)

            if module.bias is not None:
                if "classifier.header" in name:
                    bias_value = -np.log((1 - 0.01) / 0.01)
                    torch.nn.init.constant_(module.bias, bias_value)
                else:
                    module.bias.data.zero_()


class Params:
    def __init__(self, project_file):
        self.params = yaml.safe_load(open(project_file).read())

    def __getattr__(self, item):
        return self.params.get(item, None)


class ModelWithLoss(nn.Module):
    def __init__(self, model, debug=False):
        super().__init__()
        self.criterion = FocalLoss()
        self.model = model
        self.debug = debug

    def forward(self, imgs, annotations, obj_list=None):
        _, regression, classification, anchors = self.model(imgs)
        if self.debug:
            cls_loss, reg_loss = self.criterion(
                classification,
                regression,
                anchors,
                annotations,
                imgs=imgs,
                obj_list=obj_list,
            )
        else:
            cls_loss, reg_loss = self.criterion(
                classification, regression, anchors, annotations
            )
        return cls_loss, reg_loss


def train(opt):
    params = Params(f"projects/{opt.project}.yml")

    if params.num_gpus == 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    if params.num_gpus > 0:
        params.num_gpus = 1

    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    else:
        torch.manual_seed(42)

    # Setup log-directories & training-parameters
    opt.saved_path = opt.saved_path + f"/{params.project_name}_D{opt.compound_coef}/"
    opt.log_path = (
        opt.log_path + f"/{params.project_name}_D{opt.compound_coef}/tensorboard/"
    )
    os.makedirs(opt.log_path, exist_ok=True)
    os.makedirs(opt.saved_path, exist_ok=True)

    training_params = {
        "batch_size": opt.batch_size,
        "shuffle": True,
        "drop_last": True,
        "collate_fn": collater,
        "num_workers": opt.num_workers,
    }

    val_params = {
        "batch_size": opt.batch_size,
        "shuffle": False,
        "drop_last": True,
        "collate_fn": collater,
        "num_workers": opt.num_workers,
    }

    # Setup Dataloaders for train / val
    input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]
    training_set = CocoDataset(
        root_dir=os.path.join(opt.data_path, params.project_name),
        set=params.train_set,
        transform=transforms.Compose(
            [
                Normalizer(mean=params.mean, std=params.std),
                Augmenter(),
                Resizer(input_sizes[opt.compound_coef]),
            ]
        ),
    )
    training_generator = DataLoader(training_set, **training_params)

    val_set = CocoDataset(
        root_dir=os.path.join(opt.data_path, params.project_name),
        set=params.val_set,
        transform=transforms.Compose(
            [
                Normalizer(mean=params.mean, std=params.std),
                Resizer(input_sizes[opt.compound_coef]),
            ]
        ),
    )
    val_generator = DataLoader(val_set, **val_params)

    model = EfficientDetBackbone(
        num_classes=len(params.obj_list),
        compound_coef=opt.compound_coef,
        ratios=eval(params.anchors_ratios),
        scales=eval(params.anchors_scales),
    )

    # Load specified or last-available weights
    if opt.load_weights is not None:
        if opt.load_weights.endswith(".pth"):
            weights_path = opt.load_weights
        else:
            weights_path = get_last_weights(opt.saved_path)
        try:
            last_step = int(os.path.basename(weights_path).split("_")[-1].split(".")[0])
        except:
            last_step = 0

        try:
            ret = model.load_state_dict(torch.load(weights_path), strict=False)
        except RuntimeError as e:
            logging.warning(f"Ignoring {e}")
            logging.warning(
                "Don't panic if you see this, this might be because you load a pretrained weights with different number of classes. The rest of the weights should be loaded already."
            )

        logging.info(
            f"Loaded weights: {os.path.basename(weights_path)}, resuming checkpoint from step: {last_step}"
        )
    else:
        last_step = 0
        logging.info("Initializing weights...")
        init_weights(model)

    # "Head-Only" training = frozen backbone of Efficient-Det architecture
    if opt.head_only:

        def freeze_backbone(m):
            classname = m.__class__.__name__
            for ntl in ["EfficientNet", "BiFPN"]:
                if ntl in classname:
                    for param in m.parameters():
                        param.requires_grad = False

        model.apply(freeze_backbone)
        logging.info(
            "Freezed EfficientDet backbone architecture and only train the head!"
        )

    writer = SummaryWriter(
        opt.log_path + f'/{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}/'
    )

    # warp the model with loss function, to reduce the memory usage on gpu0 and speedup
    model = ModelWithLoss(model, debug=opt.debug)

    # Incase we somehow would get more thatn just 1 GPU for this training
    if params.num_gpus > 0:
        model = model.cuda()

    # Set Optimizer
    if opt.optim == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), opt.lr)
    else:
        optimizer = torch.optim.SGD(
            model.parameters(), opt.lr, momentum=0.9, nesterov=True
        )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=3, verbose=True
    )

    epoch = 0
    best_loss = 1e5
    best_epoch = 0
    step = max(0, last_step)
    model.train()

    num_iter_per_epoch = len(training_generator)

    try:
        for epoch in range(opt.num_epochs):
            last_epoch = step // num_iter_per_epoch

            # skip epochs in case of 2-step head->full training
            if epoch < last_epoch:
                continue

            epoch_loss = []
            progress_bar = tqdm(training_generator)
            for iter, data in enumerate(progress_bar):
                if iter < step - last_epoch * num_iter_per_epoch:
                    progress_bar.update()
                    continue
                try:
                    imgs = data["img"]
                    annot = data["annot"]

                    if params.num_gpus == 1:
                        imgs = imgs.cuda()
                        annot = annot.cuda()

                    optimizer.zero_grad()
                    cls_loss, reg_loss = model(imgs, annot, obj_list=params.obj_list)
                    cls_loss = cls_loss.mean()
                    reg_loss = reg_loss.mean()

                    loss = cls_loss + reg_loss
                    if loss == 0 or not torch.isfinite(loss):
                        continue

                    loss.backward()
                    optimizer.step()

                    epoch_loss.append(float(loss))

                    progress_bar.set_description(
                        "Step: {}. Epoch: {}/{}. Iteration: {}/{}. Cls loss: {:.5f}. Reg loss: {:.5f}. Total loss: {:.5f}".format(
                            step,
                            epoch,
                            opt.num_epochs,
                            iter + 1,
                            num_iter_per_epoch,
                            cls_loss.item(),
                            reg_loss.item(),
                            loss.item(),
                        )
                    )
                    writer.add_scalars("Loss", {"train": loss}, step)
                    writer.add_scalars("Regression_loss", {"train": reg_loss}, step)
                    writer.add_scalars("Classfication_loss", {"train": cls_loss}, step)

                    # log learning_rate
                    current_lr = optimizer.param_groups[0]["lr"]
                    writer.add_scalar("learning_rate", current_lr, step)

                    step += 1

                    if step % opt.save_interval == 0 and step > 0:
                        torch.save(
                            model.model.state_dict(),
                            os.path.join(
                                opt_params.saved_path,
                                f"efficientdet-d{opt.compound_coef}_{epoch}_{step}.pth",
                            ),
                        )
                        logging.info("checkpoint...")

                except Exception as e:
                    logging.error(traceback.format_exc())
                    logging.error(e)
                    continue

            scheduler.step(np.mean(epoch_loss))

            if epoch % opt.val_interval == 0:
                model.eval()
                loss_regression_ls = []
                loss_classification_ls = []
                for iter, data in enumerate(val_generator):
                    with torch.no_grad():
                        imgs = data["img"]
                        annot = data["annot"]

                        if params.num_gpus == 1:
                            imgs = imgs.cuda()
                            annot = annot.cuda()

                        cls_loss, reg_loss = model(
                            imgs, annot, obj_list=params.obj_list
                        )
                        cls_loss = cls_loss.mean()
                        reg_loss = reg_loss.mean()

                        loss = cls_loss + reg_loss
                        if loss == 0 or not torch.isfinite(loss):
                            continue

                        loss_classification_ls.append(cls_loss.item())
                        loss_regression_ls.append(reg_loss.item())

                cls_loss = np.mean(loss_classification_ls)
                reg_loss = np.mean(loss_regression_ls)
                loss = cls_loss + reg_loss

                logging.info(
                    "Val. Epoch: {}/{}. Classification loss: {:1.5f}. Regression loss: {:1.5f}. Total loss: {:1.5f}".format(
                        epoch, opt.num_epochs, cls_loss, reg_loss, loss
                    )
                )
                writer.add_scalars("Loss", {"val": loss}, step)
                writer.add_scalars("Regression_loss", {"val": reg_loss}, step)
                writer.add_scalars("Classfication_loss", {"val": cls_loss}, step)

                if loss + opt.es_min_delta < best_loss:
                    best_loss = loss
                    best_epoch = epoch

                    torch.save(
                        model.model.state_dict(),
                        os.path.join(
                            opt_params.saved_path,
                            f"efficientdet-d{opt.compound_coef}_{epoch}_{step}.pth",
                        ),
                    )

                model.train()

                # Stop if early stopping criterion is specified
                if epoch - best_epoch > opt.es_patience > 0:
                    logging.info(
                        "Stop training at epoch {}. The lowest loss achieved is {}".format(
                            epoch, best_loss
                        )
                    )
                    break
    except KeyboardInterrupt:
        torch.save(
            model.model.state_dict(),
            os.path.join(
                opt_params.saved_path,
                f"efficientdet-d{opt.compound_coef}_{epoch}_{step}.pth",
            ),
        )

        writer.close()
    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Efficient-CharacterDet")
    parser.add_argument(
        "-p",
        "--project",
        type=str,
        default="numbers",
        help="Project-file that contains dataset-specific parameters",
    )
    parser.add_argument(
        "-c",
        "--compound_coef",
        type=int,
        default=0,
        help="Efficientdet Coefficient i.e. architecture",
    )
    parser.add_argument(
        "-n", "--num_workers", type=int, default=12, help="num_workers of dataloader"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=12,
        help="The number of images per batch",
    )
    parser.add_argument(
        "--head_only",
        type=boolean_string,
        default=True,
        help="Train only the EfficientDet-head or the full NN; useful in early stage convergence or for small/easy datasets",
    )
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--optim",
        type=str,
        default="adamw",
        help="Select the optimizer for training, AdamW or SGD",
    )
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument(
        "--val_interval",
        type=int,
        default=1,
        help="Number of epoches between validation phases",
    )
    parser.add_argument(
        "--save_interval", type=int, default=100, help="Number of steps between saving"
    )
    parser.add_argument(
        "--es_min_delta",
        type=float,
        default=0.0,
        help="Early stopping parameter: minimum change loss to qualify as an improvement",
    )
    parser.add_argument(
        "--es_patience",
        type=int,
        default=0,
        help="Early stopping parameter: number of epochs with no improvement after which training will be stopped. Set to 0 to disable this technique",
    )
    parser.add_argument(
        "--data_path", type=str, default="datasets/", help="The dataset folder"
    )
    parser.add_argument("--log_path", type=str, default="logs/")
    parser.add_argument(
        "-w",
        "--load_weights",
        type=str,
        default="weights/efficientdet-d0.pth",
        help="Loag weights from a checkpoint or set to None to initialize; set 'last' to load last available checkpoint in logs/project directory",
    )
    parser.add_argument("--saved_path", type=str, default="logs/")
    parser.add_argument(
        "--debug",
        type=boolean_string,
        default=False,
        help="whether visualize the predicted boxes of training; the output images will be in test/",
    )

    opt_params = parser.parse_args()

    train(opt_params)
