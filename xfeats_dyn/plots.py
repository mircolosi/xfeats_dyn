import cv2
from pathlib import Path
import torch
from torchvision.utils import make_grid
from matplotlib import cm
import torch.nn.functional as F

import typer
from loguru import logger
from tqdm import tqdm

from xfeats_dyn.config import FIGURES_DIR, PROCESSED_DATA_DIR

app = typer.Typer()


def to_colormap(tensor):
    res = (tensor - tensor.min()) / (tensor.max() - tensor.min())
    res = torch.stack([torch.tensor(cm.turbo(r.squeeze(0).detach().cpu().numpy())[..., :3]) for r in res])
    res = res.permute(0, 3, 1, 2)
    return res

def produce_log_img(x, rel_map, target_rel_map):
    IMGS_PER_ROW = 3
    interleaved_batch: torch.Tensor = torch.empty(
        IMGS_PER_ROW * target_rel_map.size(0),
        3,
        *target_rel_map.size()[2:],
        dtype=target_rel_map.dtype,
        device=target_rel_map.device,
    )
    x_inter = F.interpolate(
        x,
        size=(target_rel_map.shape[2], target_rel_map.shape[3]),
        mode="bilinear",
        align_corners=False,
    ).expand(-1, 3, -1, -1)
    interleaved_batch[0::3] = x_inter
    interleaved_batch[1::3] = to_colormap(target_rel_map)
    interleaved_batch[2::3] = to_colormap(rel_map)

    return make_grid(interleaved_batch, nrow=IMGS_PER_ROW, pad_value=1.0)


def draw_kpts(ref_points, img):
    # Prepare keypoints and matches for drawMatches function
    keypoints1 = [cv2.KeyPoint(p[0], p[1], 5) for p in ref_points]

    out_img = cv2.drawKeypoints(img, keypoints1, 0, color=(0, 255, 0))

    return out_img


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    output_path: Path = FIGURES_DIR / "plot.png",
    # -----------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Generating plot from data...")
    for i in tqdm(range(10), total=10):
        if i == 5:
            logger.info("Something happened for iteration 5.")
    logger.success("Plot generation complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()
