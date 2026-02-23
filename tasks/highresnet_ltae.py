import torch
import json
import torch.nn.functional as F
import torchvision.transforms as T
from models.misr_module import HighResLtaeNet
from utils.hparams import hparams
from trainer import Trainer
from losses.srdiff_loss import (
    pixel_wise_closest_sr_sits_aer_loss,
    grad_pixel_wise_closest_sr_sits_aer_loss,
)


class HighResNetLTAETrainer(Trainer):
    def build_model(self):
        with open("./tasks/config_hrnet.json", "r") as read_file:
            self.config = json.load(read_file)
        self.model = HighResLtaeNet(self.config)
        self.global_step = 0
        return self.model

    def build_optimizer(self, model):
        return torch.optim.Adam(model.parameters(), hparams["lr"])

    def build_scheduler(self, optimizer):
        return torch.optim.lr_scheduler.StepLR(optimizer, 50000, 0.7)

    def super_patch_crop(self, img_lr):
        # Cropping the center of the img_lr tensor which
        # represents the super-patten of the Sat LR image
        cropping_ratio = int(img_lr.shape[-1] / 4)
        transform = T.CenterCrop((cropping_ratio, cropping_ratio))
        return transform(img_lr)

    def sat_interpolate(self, img_lr, factor=100):
        # Interpolating the img_lr to match the size of the SR image

        B, _T, C, H, W = img_lr.size()
        img_lr_inter = F.interpolate(
            img_lr.reshape(B * _T, C, H, W),
            size=(factor, factor),
            mode="bilinear",
            align_corners=False,
        )
        return img_lr_inter.reshape(B, _T, C, factor, factor)

    def training_step(self, sample):
        # Training Step of HighResLTAE Model
        img_hr = sample["img_hr"][:, :4, :, :]
        img_lr = sample["img_lr"]
        dates = sample["dates_encoding"]
        closest_idx = sample["closest_idx"]
        img_sr = self.model(img_lr, dates, self.config)

        # Compute the loss at each time step
        # Only 4 channels are used for the loss computation
        # Because the 5th channel is the NDSM band not available in the
        # low-resolution image time series.

        loss = pixel_wise_closest_sr_sits_aer_loss(
            img_sr, img_hr, closest_idx
        ) + grad_pixel_wise_closest_sr_sits_aer_loss(img_sr, img_hr, closest_idx)

        return {"l": loss, "lr": self.scheduler.get_last_lr()[0]}, loss

    def sample_and_test(self, sample):
        ret = {k: [] for k in self.metric_keys}
        ret["n_samples"] = 0
        img_hr = sample["img_hr"][:, :4, :, :]
        img_lr = sample["img_lr"]
        dates = sample["dates_encoding"]
        closest_idx = sample["closest_idx"]
        img_sr = self.model(img_lr, dates, self.config)

        # Compute the loss at each time step
        # Expand along Time dimension

        loss = pixel_wise_closest_sr_sits_aer_loss(
            img_sr, img_hr, closest_idx
        ) + grad_pixel_wise_closest_sr_sits_aer_loss(img_sr, img_hr, closest_idx)

        img_lr = self.sat_interpolate(img_lr, img_sr.shape[-1])

        for b in range(img_sr.shape[0]):
            s = self.measure.measure(
                img_sr[b][int(closest_idx[b].item()), :, :, :],  # SR image at t
                img_hr[b],  # reference HR image
                img_lr[b][int(closest_idx[b].item()), :, :, :],  # LR input at t
            )
            ret["psnr"].append(s["psnr"])
            ret["ssim"].append(s["ssim"])
            ret["lpips"].append(s["lpips"])
            ret["mae"].append(s["mae"])
            ret["mse"].append(s["mse"])
            ret["shift_mae"].append(s["shift_mae"])
            ret["miou"].append(s["miou"])

            ret["n_samples"] += 1
        return img_sr, img_sr, img_sr, ret, loss
