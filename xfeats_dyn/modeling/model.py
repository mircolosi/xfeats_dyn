import copy
import torch

from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from lightning import LightningModule
from lightning.pytorch.loggers import TensorBoardLogger
from xfeats_dyn.thirdparty.xfeats.interpolator import InterpolateSparse2d
from xfeats_dyn.plots import produce_log_img as make_grid_log

ALLOWED_LOSSES = {
    "bce": nn.BCELoss(),
    "mse": nn.MSELoss(),
    "l1": nn.L1Loss(),
    "huber": nn.HuberLoss(),
    "smooth_l1": nn.SmoothL1Loss(),
    "bce_sum": nn.BCELoss(reduction="sum"),
    "mse_sum": nn.MSELoss(reduction="sum"),
    "l1_sum": nn.L1Loss(reduction="sum"),
    "huber_sum": nn.HuberLoss(reduction="sum"),
    "smooth_l1_sum": nn.SmoothL1Loss(reduction="sum"),
    "ppw_huber2": lambda pred, target, weights: torch.sum(F.huber_loss(pred, target, reduction="none") * weights)
    / torch.sum(weights),
    "ppw_huber_sum": lambda pred, target, weights: torch.sum(F.huber_loss(pred, target, reduction="none") * weights),
    "ppw_mse2": lambda pred, target, weights: torch.sum(F.mse_loss(pred, target, reduction="none") * weights)
    / torch.sum(weights),
    "ppw_mse_sum": lambda pred, target, weights: torch.sum(F.mse_loss(pred, target, reduction="none") * weights),
}


def resize_tensor_bilinear(tensor, size=None, scale_factor=None):
    """Resizes a tensor using bilinear interpolation.

    Args:
        tensor (torch.Tensor): Input tensor of shape (B, C, H, W).
        size (tuple or int, optional): Output size (H, W). If None, scale_factor must be provided.
        scale_factor (float or tuple, optional): Scaling factor for height and width. If None, size must be provided.

    Returns:
        torch.Tensor: Resized tensor.
    """
    if size is None and scale_factor is None:
        raise ValueError("Either size or scale_factor must be specified")

    if size is not None and scale_factor is not None:
        raise ValueError("Only size or scale_factor must be specified")

    if size is not None:
        return F.interpolate(tensor, size=size, mode="bilinear", align_corners=False)
    else:
        return F.interpolate(tensor, scale_factor=scale_factor, mode="bilinear", align_corners=False)


class XFeatsDynModule(LightningModule):
    def __init__(
        self,
        learning_rate=1e-3,
        top_k=4096,
        detection_threshold=0.05,
        per_pixel_weight=1e3,
        loss=None,
        lr_scheduler_name=None,
    ):
        super().__init__()
        self.top_k = top_k
        self.xfeats = torch.hub.load("verlab/accelerated_features", "XFeat", pretrained=True, top_k=self.top_k)
        self.detection_threshold = detection_threshold

        for param in self.xfeats.net.parameters():
            param.requires_grad = True
        # Freeze params from keypoint_head
        for param in self.xfeats.net.keypoint_head.parameters():
            param.requires_grad = False

        self.xfeats_freeze = copy.deepcopy(self.xfeats)

        for param in self.xfeats_freeze.net.parameters():
            param.requires_grad = False
        self.xfeats_freeze.eval()

        self.lr_scheduler_name = lr_scheduler_name

        self.learning_rate = learning_rate

        self.loss_function = loss if loss is not None else "bce"
        self.loss = ALLOWED_LOSSES[self.loss_function]
        self.tb_logger = TensorBoardLogger("tb_logs", name=f"xfeats_dyn/{self.loss_function}")
        self.per_pixel_weight = per_pixel_weight

        self.step = "init"
        self.interpolator = InterpolateSparse2d("bicubic")
        self.save_hyperparameters()

    def _stash_log(self, loss, on_step=True):
        if isinstance(loss, dict):
            self.log_dict(
                loss,
                on_step=on_step,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
        else:
            self.log(
                f"{self.step}_loss",
                loss,
                on_step=on_step,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )

    def produce_log_img(self, x, rel_map, target_rel_map):
        grid = make_grid_log(x, rel_map, target_rel_map)

        # import matplotlib.pyplot as plt
        # import numpy as np

        # fig, ax = plt.subplots(figsize=(10, 10))
        # ax.imshow(np.transpose(grid.cpu().numpy(), (1, 2, 0)))
        # plt.axis("off")
        # plt.show()

        self.tb_logger.experiment.add_image(f"{self.step}_imgs", grid, global_step=self.current_epoch)

    def _common_train_val_step(self, batch, batch_idx):
        images, labels = batch

        segmented_mask = labels["segmentation_mask"]
        x, _, _ = self.xfeats.preprocess_tensor(images)

        _, _, target_rel_map = self.xfeats_freeze.net(x)
        _, _, rel_map = self.xfeats.net(x)

        segmented_mask, _, _ = self.xfeats.preprocess_tensor(segmented_mask)
        segmented_mask = F.interpolate(
            segmented_mask,
            size=(target_rel_map.shape[2], target_rel_map.shape[3]),
            mode="bilinear",
            align_corners=False,
        )

        assert target_rel_map.shape == segmented_mask.shape
        target_rel_map = target_rel_map * segmented_mask
        # fig, ax = plt.subplots(1, 3, figsize=(10, 10))
        # ax[0].imshow(segmented_mask.permute(2, 3, 0, 1).squeeze().cpu().numpy())
        # ax[1].imshow(target_rel_map.permute(2, 3, 0, 1).squeeze().cpu().numpy())
        # thing = rel_map.permute(2, 3, 0, 1).squeeze()
        # if hasattr(thing, "detach"):
        #     ax[2].imshow(thing.detach().cpu().numpy())
        # else:
        #     ax[2].imshow(thing.cpu().numpy())
        # plt.axis("off")
        # plt.show()

        if batch_idx % 100 == 0:
            self.produce_log_img(x, rel_map=rel_map, target_rel_map=target_rel_map)

        loss_dict = {}
        loss = 0.0
        if self.loss_function.startswith("ppw"):
            weights = (1 - segmented_mask) * self.per_pixel_weight  # + 1
            loss1 = self.loss(rel_map, target_rel_map, weights)
            loss2 = nn.MSELoss()(rel_map, target_rel_map)
            self._stash_log(
                {
                    f"{self.step}_ppw_loss": loss1,
                    f"{self.step}_mse_loss": loss2,
                },
                False,
            )
            loss = loss1 + 10 * loss2
        else:
            loss = self.loss(rel_map, target_rel_map)
        loss_dict.update({f"{self.step}_loss": loss})
        self._stash_log(loss_dict)
        # print(loss.detach().cpu().numpy())
        return loss

    def training_step(self, batch, batch_idx):
        self.step = "train"
        return self._common_train_val_step(batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        self.step = "val"
        return self._common_train_val_step(batch, batch_idx)

    def test_step(self, batch):
        self.step = "test"
        images, labels = batch
        return self.forward(batch)

    def predict_step(self, batch, batch_idx=None):
        print(f"PREDICT STEP {batch_idx}")
        self.step = "inference"
        images, labels = batch
        return self.forward(images)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        res = {"optimizer": optimizer}

        if self.lr_scheduler_name == "cosine":
            res.update({"lr_scheduler": optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=0.001)})
        elif self.lr_scheduler_name == "plateau":
            res.update(
                {
                    "lr_scheduler": {
                        "scheduler": optim.lr_scheduler.ReduceLROnPlateau(optimizer),
                        "monitor": "val_loss",
                    }
                }
            )

        return res

    # We should segment the things, then if the heatmap is high on top of segment, we should lower it
    def forward(self, x):
        x, rh1, rw1 = self.xfeats.preprocess_tensor(x)
        batch_size, _, h_proc, w_proc = x.shape

        feats, kpts, hm = self.xfeats.net(x)
        feats = F.normalize(feats, dim=1)

        kpts_heatmap = self.xfeats.get_kpts_heatmap(kpts)
        mkpts = self.xfeats.NMS(kpts_heatmap, threshold=self.detection_threshold, kernel_size=5)

        _nearest = InterpolateSparse2d("nearest")
        _bilinear = InterpolateSparse2d("bilinear")
        scores = (_nearest(kpts_heatmap, mkpts, h_proc, w_proc) * _bilinear(hm, mkpts, h_proc, w_proc)).squeeze(-1)
        scores[torch.all(mkpts == 0, dim=-1)] = -1

        # Select top-k features
        idxs = torch.argsort(-scores)
        mkpts_x = torch.gather(mkpts[..., 0], -1, idxs)[:, : self.top_k]
        mkpts_y = torch.gather(mkpts[..., 1], -1, idxs)[:, : self.top_k]
        mkpts = torch.cat([mkpts_x[..., None], mkpts_y[..., None]], dim=-1)
        scores = torch.gather(scores, -1, idxs)[:, : self.top_k]

        # Interpolate descriptors at kpts positions
        feats = self.interpolator(feats, mkpts, H=h_proc, W=w_proc)

        # L2-Normalize
        feats = F.normalize(feats, dim=-1)

        # Correct kpt scale
        mkpts = mkpts * torch.tensor([rw1, rh1], device=mkpts.device).view(1, 1, -1)

        valid = scores > 0

        return [
            {"keypoints": mkpts[b][valid[b]], "scores": scores[b][valid[b]], "descriptors": feats[b][valid[b]]}
            for b in range(batch_size)
        ]
