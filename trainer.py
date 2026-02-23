import os
import sys
import random
import torch
import importlib
import subprocess
import warnings
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from pathlib import Path
import torch.nn as nn
from torch.nn import functional as F
import torch.distributed as dist
from torchvision import transforms
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer, seed_everything
from torch.utils.tensorboard import SummaryWriter
from utils.utils_prints import print_config, print_recap
from utils.utils_dataset import read_config
from data.load_data import load_data
from utils.utils_dataset import (
    read_config,
    pad_collate_train,
    pad_collate_predict,
    save_image_to_nested_folder,
    save_hr_image_to_nested_folder,
)

from utils.utils_prints import (
    print_config,
    print_recap,
    print_metrics,
    print_inference_time,
    print_iou_metrics,
    print_f1_metrics,
    print_overall_accuracy,
)
from utils.metrics import generate_miou, generate_mf1s

from pytorch_lightning.utilities.rank_zero import rank_zero_only

os.environ["KMP_DUPLICATE_LIB_OK"] = "1"
# sys.path.append('../')

from utils.utils import (
    move_to_cuda,
    load_checkpoint,
    save_checkpoint,
    tensors_to_scalars,
    Measure,
)
from utils.hparams import hparams, set_hparams
from data.dataset import FitDataset, PredictDataset
from torchvision import transforms as cT

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

warnings.filterwarnings("ignore", ".*does not have many workers.*")


class Trainer:
    def __init__(self, config_data):
        self.logger = self.build_tensorboard(
            save_dir=hparams["work_dir"], name="tb_logs"
        )
        d_train, d_val, d_test = load_data(
            config_data, val_percent=config_data["val_percent"]
        )
        self.d_train = d_train
        self.d_val = d_val
        self.d_test = d_test

        self.measure = Measure()
        self.dataset_cls = None
        self.metric_keys = [
            "psnr",
            "ssim",
            "lpips",
            "mae",
            "mse",
            "shift_mae",
            "miou",
        ]
        self.work_dir = hparams["work_dir"]
        self.first_val = True
        self.config_data = config_data
        self.device = device
        print_recap(config_data, d_train, d_val, d_test)

    def crop_sits_image(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        - cropped upsampled image to (5, 100, 100).
        Returns:
        - torch.Tensor: Upsampled tensor of shape (5, 256, 256).
        """
        cropping_ration = int(input_tensor.shape[-1] / 4)
        transform = cT.CenterCrop((cropping_ration, cropping_ration))
        cropped_tensor = transform(input_tensor)
        return cropped_tensor

    def seed_worker(self, worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    def build_tensorboard(self, save_dir, name, **kwargs):
        log_dir = os.path.join(save_dir, name)
        os.makedirs(log_dir, exist_ok=True)
        return SummaryWriter(log_dir=log_dir, **kwargs)

    def build_train_dataloader(self, subset=True):
        g = torch.Generator()
        g.manual_seed(0)
        dataset_train = FitDataset(
            self.d_train,
            config=self.config_data,
            use_augmentation=self.config_data["use_augmentation"],
        )

        dataloader = DataLoader(
            dataset_train,
            batch_size=hparams["batch_size"],
            shuffle=True,
            num_workers=hparams["num_workers"],
            drop_last=True,
            collate_fn=pad_collate_train,
        )
        return dataloader

    def build_val_dataloader(self, subset=True):
        dataset_val = FitDataset(
            self.d_val,
            config=self.config_data,
            use_augmentation=self.config_data["use_augmentation"],
        )
        dataloader = DataLoader(
            dataset_val,
            batch_size=hparams["eval_batch_size"],
            shuffle=False,
            num_workers=hparams["num_workers"],
            drop_last=True,
            collate_fn=pad_collate_train,
        )
        return dataloader

    def build_test_dataloader(self, subset=False):
        dataset_test = PredictDataset(
            self.d_test,
            config=self.config_data,
        )

        dataloader = DataLoader(
            dataset_test,
            batch_size=hparams["test_batch_size"],
            shuffle=False,
            num_workers=hparams["num_workers"],
            drop_last=False,
            collate_fn=pad_collate_predict,
        )
        return dataloader

    def build_model(self):
        raise NotImplementedError

    def sample_and_test(self, sample):
        raise NotImplementedError

    def build_optimizer(self, model):
        raise NotImplementedError

    def build_scheduler(self, optimizer):
        raise NotImplementedError

    def training_step(self, batch):
        raise NotImplementedError

    @rank_zero_only
    def print_model_parameters(self, model: nn.Module) -> None:
        """
        Print the total number of parameters in the model, broken down by component.
        Args:
            model (nn.Module): Full model containing gaussian, encoders, decoder, and fusion modules.
        """

        def count_params(module: nn.Module | None) -> int:
            return sum(p.numel() for p in module.parameters()) if module is not None else 0

        # Parameter counts
        gaussian_params_cond_net = count_params(model.gaussian.cond_net)
        gaussian_params_denoise_fn = count_params(model.gaussian.denoise_fn)
        sits_encoder_params = count_params(model.sits_net)
        aerial_net_params = count_params(model.aerial_net)
        decoder_params = count_params(model.decoder)
        fusion_params = count_params(model.fusion_module)

        total_params = (
            gaussian_params_cond_net
            + gaussian_params_denoise_fn
            + sits_encoder_params
            + aerial_net_params
            + decoder_params
            + fusion_params
        )

        # Build table
        table = " " + "-" * 113 + "\n"
        table += (
            "| {:<37} | {:<35} | {:<17} | {:<13} |\n"
            "| {} | {} | {} | {} |\n"
        ).format(
            "Model component",
            "Sub-module",
            "Type",
            "Parameters",
            "-" * 37,
            "-" * 35,
            "-" * 17,
            "-" * 13,
        )

        rows = [
            ("Gaussian", "cond_net", "conditioning net", gaussian_params_cond_net),
            ("Gaussian", "denoise_fn", "denoising net", gaussian_params_denoise_fn),
            ("SITS Encoder", "sits_net", "encoder", sits_encoder_params),
            ("Aerial Encoder", "aerial_net", "encoder", aerial_net_params),
            ("Decoder", "decoder", "task decoder", decoder_params),
            ("Fusion", "fusion_module", "fusion module", fusion_params),
        ]

        for component, submodule, module_type, params in rows:
            table += "| {:<37} | {:<35} | {:<17} | {:>13,} |\n".format(
                component, submodule, module_type, params
            )

        # Footer
        table += "|" + "-" * 113 + "|\n"
        table += "| {:<37}   {:<35}   {:<17}   {:>13,} |\n".format(
            "Total parameters", "", "", total_params
        )
        table += " " + "-" * 113

        print("")
        print(table)

    def train(self):
        model = self.build_model()
        self.print_model_parameters(model)

        optimizer = self.build_optimizer(model)
        self.global_step = training_step = load_checkpoint(
            model, optimizer, hparams["work_dir"]
        )
        self.scheduler = scheduler = self.build_scheduler(optimizer)
        dataloader = self.build_train_dataloader()

        train_pbar = tqdm(dataloader)

        list_loss = []
        val_list = []
        val_steps = []
        val_loss_list = []
        epoch = 0

        # Load existing logs if they exist
        train_loss_path = hparams["work_dir"] + "/train_loss.csv"
        val_res_path = hparams["work_dir"] + "/val_res.csv"

        try:
            previous_train_loss = pd.read_csv(train_loss_path, sep=";")
            list_loss = previous_train_loss["l"].tolist()
        except Exception:
            previous_train_loss = pd.DataFrame()

        try:
            previous_val_res = pd.read_csv(val_res_path, sep=";")
            val_list = previous_val_res["val_metrics"].tolist()
            val_steps = previous_val_res["train_step"].tolist()
            if not hparams["train_diffsr"] and "val_loss" in previous_val_res.columns:
                val_loss_list = previous_val_res["val_loss"].tolist()
        except Exception:
            previous_val_res = pd.DataFrame()

        while self.global_step < hparams["max_updates"] + 1:
            c = 0
            loss_ = 0

            for idx, batch in enumerate(train_pbar):
                if (training_step % hparams["val_check_interval"] == 0) and (
                    training_step != 0
                ):
                    if training_step not in val_steps:
                        with torch.no_grad():
                            model.eval()
                            val_res, val_step, val_loss = self.validate(training_step)
                            val_list.append(val_res)
                            val_steps.append(val_step)
                            if not hparams["train_diffsr"]:
                                val_loss_list.append(val_loss)

                # if training_step % hparams["save_ckpt_interval"] == 0:
                #     save_checkpoint(
                #         model,
                #         optimizer,
                #         self.work_dir,
                #         training_step,
                #         hparams["num_ckpt_keep"],
                #     )

                if training_step % hparams["save_ckpt_interval"] == 0:
                    val_miou = None
                    if (
                        val_list
                        and isinstance(val_list[-1], dict)
                        and "val/miou" in val_list[-1]
                    ):
                        val_miou = round(val_list[-1]["val/miou"], 2)

                    save_checkpoint(
                        model,
                        optimizer,
                        self.work_dir,
                        training_step,
                        hparams["num_ckpt_keep"],
                        val_miou=val_miou,
                    )

                model.train()
                batch = move_to_cuda(batch)
                losses, total_loss = self.training_step(batch)
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                training_step += 1

                loss_ += total_loss.detach().item()
                c += 1
                scheduler.step()
                self.global_step = training_step
                if training_step % 1000 == 0:
                    self.log_metrics(
                        {f"tr/{k}": v for k, v in losses.items()}, training_step
                    )
                train_pbar.set_postfix(**tensors_to_scalars(losses))

                list_loss.append(loss_ / c if c != 0 else 0.0)
                epoch += 1

            if epoch % 1 == 0:
                save_loss = pd.DataFrame(
                    {
                        "training_step": [i for i in range(len(list_loss))],
                        "l": list_loss,
                    }
                )
                save_loss.to_csv(train_loss_path, sep=";", index=False)

                save_val = pd.DataFrame(
                    {"train_step": val_steps, "val_metrics": val_list}
                )
                if not hparams["train_diffsr"]:
                    save_val["val_loss"] = val_loss_list
                save_val.to_csv(val_res_path, sep=";", index=False)

    def validate(self, training_step):
        val_dataloader = self.build_val_dataloader()
        pbar = tqdm(enumerate(val_dataloader), total=len(val_dataloader))
        val_loss = 0
        c = 0
        val_metrics = {k: [] for k in self.metric_keys}

        for _, batch in pbar:
            # for batch_idx, batch in pbar:
            batch = move_to_cuda(batch)
            _, _, _, ret, loss = self.sample_and_test(batch)
            # img, preds, cond_net_out, ret, loss = self.sample_and_test(batch)

            if not hparams["train_diffsr"]:
                val_loss += loss.detach().item()
                c += 1
            metrics = {}
            metrics.update({k: np.mean(ret[k]) for k in self.metric_keys})
            for k in self.metric_keys:
                val_metrics[k].append(np.mean(ret[k]))
            pbar.set_postfix(**tensors_to_scalars(metrics))
        if not hparams["train_diffsr"]:
            val_loss = val_loss / c
        if hparams["infer"]:
            print("Val results:", metrics)
        else:
            if not self.first_val:
                self.log_metrics(
                    {f"val/{k}": v for k, v in metrics.items()}, training_step
                )
                print("Val results:", metrics)
                return (
                    {f"val/{k}": np.mean(v) for k, v in val_metrics.items()},
                    training_step,
                    val_loss,
                )
            else:
                print("Sanity val results:", metrics)
                return (
                    {f"val/{k}": np.mean(v) for k, v in val_metrics.items()},
                    training_step,
                    val_loss,
                )
        self.first_val = False

    # Run Inference
    def test(self):
        model = self.build_model()
        # print(model)
        optimizer = self.build_optimizer(model)
        load_checkpoint(model, optimizer, hparams["work_dir"])
        optimizer = None
        self.results = {k: [] for k in self.metric_keys}
        self.results["key"] = []
        self.n_samples = 0
        self.gen_dir = f"{hparams['work_dir']}/results_{self.global_step}_{hparams['gen_dir_name']}"

        if hparams["test_save_png"]:
            os.makedirs(f"{self.gen_dir}/SR", exist_ok=True)
            os.makedirs(f"{self.gen_dir}/PR", exist_ok=True)

        self.model.sample_tqdm = False
        torch.backends.cudnn.benchmark = False
        if hparams["test_save_png"]:
            if hparams["test_diff"]:
                if hasattr(self.model.gaussian.denoise_fn, "make_generation_fast_"):
                    self.model.gaussian.denoise_fn.make_generation_fast_()
            # os.makedirs(f"{self.gen_dir}/RRDB", exist_ok=True)
            os.makedirs(f"{self.gen_dir}/HR", exist_ok=True)
            os.makedirs(f"{self.gen_dir}/LR", exist_ok=True)
            os.makedirs(f"{self.gen_dir}/UP", exist_ok=True)

        with torch.no_grad():
            model.eval()
            test_dataloader = self.build_test_dataloader()
            pbar = tqdm(enumerate(test_dataloader), total=len(test_dataloader))

            for _, batch in pbar:
                move_to_cuda(batch)
                gen_dir = self.gen_dir
                item_names = batch["item_name"]
                img_hr = batch["img_hr"]
                img_lr = batch["img_lr"]
                img_lr_up = batch["img_lr_up"]
                dates = batch["dates"]
                res = self.sample_and_test(batch)

                if len(res) == 5:
                    img_sr, preds, _, ret, _ = res
                else:
                    img_sr, preds, ret, ret = res

                if img_sr is not None:
                    metrics = list(self.metric_keys)
                    for k in metrics:
                        self.results[k] += ret[k]
                    self.n_samples += ret["n_samples"]
                    print(
                        {k: np.mean(self.results[k]) for k in metrics},
                        "total:",
                        self.n_samples,
                    )

                    if hparams["test_save_png"] and img_sr is not None:
                        img_hr = self.tensor2img(img_hr)

                        # For single image batch size, we can use the following code
                        if hparams["test_batch_size"] == 1:
                            img_lr = [
                                self.tensor2img(self.crop_sits_image(im[None, ...]))
                                for im in img_lr.squeeze()
                            ]

                            img_lr_up = [
                                self.tensor2img(im[None, ...])
                                for im in img_lr_up.squeeze()
                            ]

                            img_sr = [
                                self.tensor2img(im[None, ...])
                                for im in img_sr.squeeze()
                            ]

                        for item_name, _, hr_g, lr, lr_up, pred, _ in zip(
                            item_names, img_sr, img_hr, img_lr, img_lr_up, preds, dates
                        ):
                            # Save high-resolution ground truth image
                            # hr_g = Image.fromarray(hr_g[:, :, :4])
                            # save_hr_image_to_nested_folder(
                            #     hr_g, item_name, "HR", "img", None, base_dir=gen_dir
                            # )

                            # Save pixel-wise predictions
                            pred = pred.cpu().numpy().astype("uint8")
                            output_file = Path(
                                gen_dir,
                                "PR",
                                item_name.split("/")[-1].replace("IMG", "PRED"),
                            )
                            Image.fromarray(pred).save(
                                f"{output_file}", compression="tiff_lzw"
                            )

                            if hparams["test_batch_size"] == 1:
                                dates = [date for date in dates]

                                lr = [Image.fromarray(im[0]) for im in img_lr]
                                for e, (im, date) in enumerate(zip(lr, dates[0])):
                                    save_image_to_nested_folder(
                                        im,
                                        item_name,
                                        "LR",
                                        "sen",
                                        f"{e}_{date}",
                                        base_dir=gen_dir,
                                    )

                                lr_up = [Image.fromarray(im[0]) for im in img_lr_up]
                                for e, (im, date) in enumerate(zip(lr_up, dates[0])):
                                    save_image_to_nested_folder(
                                        im,
                                        item_name,
                                        "UP",
                                        "sen",
                                        f"{e}_{date}",
                                        base_dir=gen_dir,
                                    )

                                sr = [Image.fromarray(im[0]) for im in img_sr]
                                for e, (im, date) in enumerate(zip(sr, dates[0])):
                                    save_image_to_nested_folder(
                                        im,
                                        item_name,
                                        "SR",
                                        "sen",
                                        f"{e}_{date}",
                                        base_dir=gen_dir,
                                    )

            self.results = {
                k: self.results[k]
                for k in ["psnr", "ssim", "lpips", "mae", "mse", "shift_mae", "miou"]
            }
            res = pd.DataFrame(self.results)
            res.to_csv(hparams["work_dir"] + "/test_results.csv", sep=";")

    def generate_metrics(self):
        # Compute mIoU over the predictions - not done here as the test
        #  labels are not available, but if needed, you can use the
        #  generate_miou function from metrics.py
        truth_msk = self.config_data["data"]["path_labels_test"]
        pred_msk = os.path.join(self.gen_dir, "PR")

        mIou, ious = generate_miou(self.config_data, truth_msk, pred_msk)
        mf1, f1s, oa = generate_mf1s(truth_msk, pred_msk)

        print_iou_metrics(mIou, ious)
        print_f1_metrics(mf1, f1s)
        print_overall_accuracy(oa)

    # utils
    def log_metrics(self, metrics, step):
        metrics = self.metrics_to_scalars(metrics)
        logger = self.logger
        for k, v in metrics.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            logger.add_scalar(k, v, step)

    def metrics_to_scalars(self, metrics):
        new_metrics = {}
        for k, v in metrics.items():
            if isinstance(v, torch.Tensor):
                v = v.item()

            if type(v) is dict:
                v = self.metrics_to_scalars(v)

            new_metrics[k] = v

        return new_metrics

    def tensor2img(self, img):
        img = np.round((img.permute(0, 2, 3, 1).cpu().numpy() + 1) * 127.5)
        img = img.clip(min=0, max=255).astype(np.uint8)
        return img


if __name__ == "__main__":
    set_hparams()
    config = read_config(hparams["config_file"])
    pkg = ".".join(hparams["trainer_cls"].split(".")[:-1])
    cls_name = hparams["trainer_cls"].split(".")[-1]

    trainer = getattr(importlib.import_module(pkg), cls_name)(config)

    # printing model configuration
    print_config(config)

    if not hparams["infer"]:
        trainer.train()
    else:
        trainer.test()
        trainer.generate_metrics()


# Contributions:

# 1. Auxiliary Losses

# “We incorporate four auxiliary losses to guide the diffusion-based SR training:
# (i) a cross-sensor pixel alignment loss with high-resolution aerial references,
# (ii) a spatial gradient consistency loss to preserve edges,
# (iii) a temporal gradient magnitude loss to enforce smooth dynamics across the time series,
# (iv) a gray-value consistency loss to ensure spectral agreement with the low-resolution input.”


# Breakdown of each auxiliary loss

# 1. Pixel-wise closest SR–aerial consistency loss
# pixel_wise_closest_sr_sits_aer_loss(x0_pred, img_hr, closest_idx)
# Ensures that the super-resolved time series frame is consistent with its closest high-resolution aerial reference.
# Encourages cross-sensor alignment at the pixel level.

# 2. Gradient-based spatial detail preservation
# grad_pixel_wise_closest_sr_sits_aer_loss(x0_pred, img_hr, closest_idx)
# Weighted by hparams["grad_px_loss_weight"].
# Matches spatial gradients between SR output and HR aerial reference.
# Helps preserve edges and fine details (sharpness).

# 3. Temporal gradient magnitude consistency
# temp_gradient_magnitude_consistency_loss(x0_pred)
# Weighted by hparams["temp_grad_mag_loss_weight"].
# Encourages temporal smoothness by penalizing sudden changes in gradient magnitude across the time series.
# Acts as a temporal regularizer to prevent flickering artifacts.

# 4. Gray-value consistency with low-resolution inputs
# gray_value_consistency_loss(x0_pred, img_lr)
# Weighted by hparams["gray_value_px_loss_weight"].
# Ensures that when downsampled, the SR images remain consistent with the observed LR sequence.
# Preserves spectral fidelity and avoids hallucinations.


# python trainer.py --config configs/misr/pretrain.yaml --config_file flair-config.yml --exp_name misr/highresnet_ltae_ckpt --reset
# python trainer.py --config configs/diffsr_highresnet_ltae.yaml --config_file flair-config.yml --exp_name misr/srdiff_highresnet_ltae_ckpt --hparams="cond_net_ckpt=./results/checkpoints/misr/highresnet_ltae_ckpt" --reset


# python trainer.py --config configs/diffsr_highresnet_ltae.yaml --config_file flair-config.yml --exp_name misr/srdiff_highresnet_ltae_ckpt --hparams="cond_net_ckpt=D:\kanyamahanga\Datasets\pretrain_weights\LCC_SITS_MaxViT_UNet\checkpoints\ckpt-epoch=06-val_loss=1.36-val_miou=0.34_maxvit-unet-seg.ckpt" --reset
