import torch
from torchmetrics.classification import AUROC, BinaryAveragePrecision
from torchmetrics.utilities.data import dim_zero_cat
from tqdm import tqdm

from diffusers.models.unets import unet_2d
from typing import Optional, Union

class MaxDSC(BinaryAveragePrecision):
    def compute(self) -> torch.Tensor:  # type: ignore[override]
        """Compute metric."""
        state = (dim_zero_cat(self.preds), dim_zero_cat(self.target))

        if self.thresholds is None:
            min_, max_ = state[0].min(), state[1].max()
            # Diffusion [DICE] curve
            λ = torch.linspace(min_, max_, 200)[None] # Just get 200 points in between max and min linearly spaced
        else:
            λ = self.thresholds[None]
        preds = state[0]
        labels  = state[1]

        DSC_curve = 2 * ((preds[:, None] > λ) * labels.float()[:, None]).sum(0)[None] \
                    / ((preds[:, None] > λ).sum(0)[None] + labels.float().sum()[None])

        return DSC_curve.amax()

#When doing image wise classifaction, defines functions to reduce pixel-wise predictions to image-wise predictions
image_wise_reduction = {
    "mean": lambda x: x.flatten(1).mean(1),
    "max": lambda x: x.flatten(1).amax(1),
    "mean_top100": lambda x: x.flatten(1).topk(100,1).values.mean(1),
}

def evaluate_anomaly_loader( loader, pipe, test_run = False, pipeline_kwargs = {} ):

    eval_iterator_val = tqdm(
        range(len(loader.generator.datalist)) , desc=f"Validate {loader.name}", dynamic_ncols=True,
        position=0, total=len(loader.generator.datalist) , leave=True
    )

    avg_prec_p = BinaryAveragePrecision()
    max_DSC_p = MaxDSC()

    avg_prec_i = {}
    max_DSC_i = {}
    auroc_i = {}

    for kr in image_wise_reduction.keys():
        avg_prec_i[kr] = BinaryAveragePrecision()
        max_DSC_i[kr] = MaxDSC()
        auroc_i[kr] = AUROC(task="binary")

    for i, b in zip(eval_iterator_val,loader):
        # If the loader retrieves full images instead of patches, it might not have concatenated the batch dimension.
        batch_size = None

        if isinstance(b['data'], list):
            if b['data'][0].ndim == 4:
                # If it's evaluating full volumes  (i.e.received as  3D),
                # concatenate along last dimension and move it to the first dimension
                batch_size = len(b['data'])
                b['data'] = torch.cat(b['data'],-1).moveaxis(-1,0)
                b['seg'] = torch.cat(b['seg'],-1).moveaxis(-1,0)
            else:
                #This will throw an error if sizes haven't been normalized in preprocessing... Fix your preprocessing!
                b['data'] = torch.stack(b['data'],0)
                b['seg'] = torch.stack(b['seg'],0)

        # To speed up AP/DSC pixel-wise, get only 10k random pixels
        idx = torch.randint(b['data'].sum(1).nelement(), (10000,))
        labels = (b['seg'] > 0).long()
        predictions = pipe(b['data'], **pipeline_kwargs).anomaly_score.cpu()

        avg_prec_p.update(predictions.flatten()[idx],labels.flatten()[idx])
        max_DSC_p.update(predictions.flatten()[idx],labels.flatten()[idx])

        # Image-wise, if any pixel is anomalous define the slide as anomalous
        label_i = labels.flatten(1).amax(1)
        for kr, fr  in image_wise_reduction.items():
            pred_i = fr(predictions)
            avg_prec_i[kr].update(pred_i,label_i)
            max_DSC_i[kr].update(pred_i,label_i)
            auroc_i[kr].update(pred_i,label_i)

        if test_run and (i >= 2):
            break

        eval_iterator_val.update(batch_size if batch_size is not None else b['data'].shape[0])

    val_to_log = {f"{loader.name}/DSCp": max_DSC_p.compute().item(),
                  f"{loader.name}/APp": avg_prec_p.compute().item()}

    for kr in image_wise_reduction.keys():
        val_to_log[f"{loader.name}/DSCi_{kr}"] = max_DSC_i[kr].compute().item()
        val_to_log[f"{loader.name}/APi_{kr}"] = avg_prec_i[kr].compute().item()
        val_to_log[f"{loader.name}/AUROCi_{kr}"] = auroc_i[kr].compute().item()

    return val_to_log

def evaluate_anomaly_loader_image_only( loader, pipe, test_run = False, pipeline_kwargs = {}):
    """if t is None, residuals between images and restorations are computed"""
    eval_iterator_val = tqdm(
        loader, desc=f"Validate {loader.name} ", dynamic_ncols=True,
        position=0, leave=True, total=len(loader.generator.datalist)
    )

    avg_prec_i = {}
    max_DSC_i = {}
    auroc_i = {}

    for kr in image_wise_reduction.keys():
        avg_prec_i[kr] = BinaryAveragePrecision()
        max_DSC_i[kr] = MaxDSC()
        auroc_i[kr] = AUROC(task="binary")

    for i, b in enumerate(eval_iterator_val):
        label_i = torch.tensor([i['class'] for i in b['metadata']]).long()
        predictions = pipe(b['data'], **pipeline_kwargs).anomaly_score.cpu()

        for kr, fr  in image_wise_reduction.items():
            pred_i = fr(predictions)
            avg_prec_i[kr].update(pred_i,label_i)
            max_DSC_i[kr].update(pred_i,label_i)
            auroc_i[kr].update(pred_i,label_i)

        eval_iterator_val.update(b['data'].shape[0])

        if test_run and (i >= 2):
            break

    val_to_log = {}
    for kr in image_wise_reduction.keys():
        val_to_log[f"{loader.name}/DSCi_{kr}"] = max_DSC_i[kr].compute().item()
        val_to_log[f"{loader.name}/APi_{kr}"] = avg_prec_i[kr].compute().item()
        val_to_log[f"{loader.name}/AUROCi_{kr}"] = auroc_i[kr].compute().item()

    return val_to_log