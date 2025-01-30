from pathlib import Path

import typer
import torch
import torch.compiler
from loguru import logger
from lightning import Trainer
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor

from xfeats_dyn.config import MODELS_DIR, MODEL_NAME
from xfeats_dyn.dataset import Coco2017DataModule
from xfeats_dyn.modeling.model import XFeatsDynModule, ALLOWED_LOSSES

import torch._dynamo

torch._dynamo.config.suppress_errors = True

app = typer.Typer()


@app.command()
def main(
    loss_function: str = "mse",
    dataset: str = "coco",
    batch_size: int = 64,
    max_epochs: int = -1,
    min_epochs: int = 10,
    use_existing_ckpt: bool = False,
    checkpoint: Path = None,
    compile: bool = False,
):
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else None
    if device is None:
        logger.error("No CUDA device found.")
        raise typer.Abort()

    print(f"Using {device} device")

    datamodule = None
    if dataset == "coco":
        datamodule = Coco2017DataModule(batch_size=batch_size)
    else:
        logger.error(f"Invalid dataset: {dataset}")
        raise typer.Abort()

    learning_rate = 3e-4

    if loss_function not in ALLOWED_LOSSES:
        logger.error(f"Invalid loss function: {loss_function}")
        raise typer.Abort()

    logger.info(f"Training model {MODEL_NAME} with loss {loss_function}...")
    latest_checkpoint = checkpoint
    if checkpoint is not None:
        logger.info(f"Using checkpoint: {checkpoint}")
        model = XFeatsDynModule.load_from_checkpoint(checkpoint, learning_rate=learning_rate, loss=loss_function)
    else:
        existing_checkpoints = list((MODELS_DIR / loss_function).glob(f"{MODEL_NAME}_epoch*.ckpt"))
        if existing_checkpoints and use_existing_ckpt:
            latest_checkpoint = max(existing_checkpoints, key=lambda p: int(p.stem.split("epoch")[1]))
            logger.info(f"Found existing checkpoint: {latest_checkpoint}")
            model = XFeatsDynModule(loss=loss_function)

        else:
            model = XFeatsDynModule(learning_rate=learning_rate, loss=loss_function)

    if compile:
        model.compile()

    loss_name = "train_loss"
    early_stopping_cb = EarlyStopping(
        monitor=loss_name,
        min_delta=0.0,
        patience=10,
        mode="min",
        verbose=False,
        strict=True,
    )
    ckpt_cb = ModelCheckpoint(
        monitor=loss_name,
        dirpath=MODELS_DIR / loss_function,
        filename=MODEL_NAME + "_epoch{epoch:02d}",
        auto_insert_metric_name=False,
        save_top_k=5,
    )

    trainer = Trainer(
        max_epochs=max_epochs,
        min_epochs=min_epochs,
        logger=model.tb_logger,
        callbacks=[early_stopping_cb, ckpt_cb, LearningRateMonitor(logging_interval="step")],
    )

    # Set the current epoch to the highest epoch found
    trainer.fit(model, datamodule=datamodule, ckpt_path=latest_checkpoint)

    model_path: Path = MODELS_DIR / f"{MODEL_NAME}_{loss_function}.ckpt"
    trainer.save_checkpoint(model_path)
    logger.info(f"Model saved at {model_path}. Reached loss @ {trainer.callback_metrics[loss_name].numpy()}")
    logger.info(f"Find best path model in {ckpt_cb.best_model_path}. Reached loss @ {ckpt_cb.best_model_score}")

    logger.success("Modeling training complete.")


if __name__ == "__main__":
    app()
