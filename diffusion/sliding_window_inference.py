
import torch
import torch.nn.functional as F

# Adapted from batch_generators numpy --> torch
def pad_nd_image(image, new_shape=None, mode="constant", kwargs=None, return_slicer=False, shape_must_be_divisible_by=None):
    """
    one padder to pad them all. Documentation? Well okay. A little bit

    :param image: nd image. can be anything
    :param new_shape: what shape do you want? new_shape does not have to have the same dimensionality as image. If
    len(new_shape) < len(image.shape) then the last axes of image will be padded. If new_shape < image.shape in any of
    the axes then we will not pad that axis, but also not crop! (interpret new_shape as new_min_shape)
    Example:
    image.shape = (10, 1, 512, 512); new_shape = (768, 768) -> result: (10, 1, 768, 768). Cool, huh?
    image.shape = (10, 1, 512, 512); new_shape = (364, 768) -> result: (10, 1, 512, 768).

    :param mode: see torch.nn.functional.pad for documentation
    :param return_slicer: if True then this function will also return what coords you will need to use when cropping back
    to original shape
    :param shape_must_be_divisible_by: for network prediction. After applying new_shape, make sure the new shape is
    divisibly by that number (can also be a list with an entry for each axis). Whatever is missing to match that will
    be padded (so the result may be larger than new_shape if shape_must_be_divisible_by is not None)
    :param kwargs: see torch.nn.functional.pad for documentation
    """
    if kwargs is None:
        kwargs = {'value': 0}

    if new_shape is not None:
        old_shape = torch.tensor(image.shape[-len(new_shape):])
    else:
        assert shape_must_be_divisible_by is not None
        assert isinstance(shape_must_be_divisible_by, (list, tuple, torch.Tensor))
        new_shape = torch.tensor(image.shape[-len(shape_must_be_divisible_by):])
        old_shape = new_shape

    num_axes_nopad = len(image.shape) - len(new_shape)

    new_shape = [max(new_shape[i], old_shape[i]) for i in range(len(new_shape))]

    if not isinstance(new_shape, torch.Tensor):
        new_shape = torch.tensor(new_shape)

    if shape_must_be_divisible_by is not None:
        if not isinstance(shape_must_be_divisible_by, (list, tuple, torch.Tensor)):
            shape_must_be_divisible_by = [shape_must_be_divisible_by] * len(new_shape)
        else:
            assert len(shape_must_be_divisible_by) == len(new_shape)

        for i in range(len(new_shape)):
            if new_shape[i] % shape_must_be_divisible_by[i] == 0:
                new_shape[i] -= shape_must_be_divisible_by[i]

        new_shape = torch.tensor \
            ([new_shape[i] + shape_must_be_divisible_by[i] - new_shape[i] % shape_must_be_divisible_by[i] for i in range(len(new_shape))])

    difference = new_shape - old_shape
    pad_below = difference // 2
    pad_above = difference // 2 + difference % 2

    pad = torch.stack((pad_below ,pad_above) ,0).flip(1).T.flatten()
    pad_list = [[0, 0] ] *num_axes_nopad + torch.stack((pad_below, pad_above) ,-1).tolist()

    if not ((all([i == 0 for i in pad_below])) and (all([i == 0 for i in pad_above]))):
        res = F.pad(image, tuple(pad.tolist()), mode, **kwargs)
    else:
        res = image

    if not return_slicer:
        return res
    else:
        pad_list = torch.tensor(pad_list)
        pad_list[:, 1] = torch.tensor(res.shape) - pad_list[:, 1]
        slicer = list(slice(*i) for i in pad_list)
        return res, slicer

# Copied from MONAI
def compute_importance_map(patch_size, mode, sigma_scale = 0.125, device = "cpu", dtype = torch.float32,):
    """Get importance map for different weight modes.

    Args:
        patch_size: Size of the required importance map. This should be either H, W [,D].
        mode: {``"constant"``, ``"gaussian"``}
            How to blend output of overlapping windows. Defaults to ``"constant"``.

            - ``"constant``": gives equal weight to all predictions.
            - ``"gaussian``": gives less weight to predictions on edges of windows.

        sigma_scale: Sigma_scale to calculate sigma for each dimension
            (sigma = sigma_scale * dim_size). Used for gaussian mode only.
        device: Device to put importance map on.
        dtype: Data type of the output importance map.

    Raises:
        ValueError: When ``mode`` is not one of ["constant", "gaussian"].

    Returns:
        Tensor of size patch_size.

    """
    if isinstance(device,str):
        device = torch.device(device)
    if mode == "constant":
        importance_map = torch.ones(patch_size, device=device, dtype=torch.float)
    elif mode == "gaussian":
        sigma_scale = (sigma_scale,) * len(patch_size)
        sigmas = [i * sigma_s for i, sigma_s in zip(patch_size, sigma_scale)]

        for i in range(len(patch_size)):
            x = torch.arange(
                start=-(patch_size[i] - 1) / 2.0, end=(patch_size[i] - 1) / 2.0 + 1, dtype=torch.float, device=device
            )
            x = torch.exp(x**2 / (-2 * sigmas[i] ** 2))  # 1D gaussian
            importance_map = importance_map.unsqueeze(-1) * x[(None,) * i] if i > 0 else x
    else:
        raise ValueError(
            f"Unsupported mode: {mode}, available options are 'constant' and 'gaussian']."
        )
    # handle non-positive weights
    min_non_zero = max(torch.min(importance_map).item(), 1e-3)
    importance_map = torch.clamp_(importance_map.to(torch.float), min=min_non_zero).to(dtype)
    return importance_map


def unfold(image, patch_size, steps):
    # Get the individual patches processed in the sliding window
    # unfold returns (bs,c,sw1,sw2,...,d1,d2,...), being (d1,d2,...) patch size and (sw1, sw2,...) the num of windows
    for id_, (ps, st) in enumerate(zip(patch_size, steps)):
        image = image.unfold(id_ + 2, size=ps, step=st)
    return image

def fold(image, image_size, patch_size, steps):
    """Image is expected (bs,sw1,sw2,...,c,d1,d2,...), being (d1,d2,...) patch size"""

    # Generate output folded. the idea is to use the unfolded indexes and revert the opperation using scatter_add
    # from https://github.com/f-dangel/unfoldNd/blob/main/unfoldNd/fold.py

    numel_image = torch.tensor(image_size).prod()
    numel_patch = torch.tensor(patch_size).prod()

    # (bs, sw1, sw2,...,c,d1,d2,...) > (bs, c, d1 * d2 *... * sw1 * sw2 * ...)
    bs_, c_ = image.shape[0], image.shape[-len(patch_size) - 1]
    image = image.reshape(bs_, -1, c_ * numel_patch)
    image = image.movedim(1, -1).reshape(bs_, c_, -1)

    idx = torch.arange(numel_image,device=image.device)
    idx = idx.reshape(1, 1, *image_size)
    idx = unfold(idx, patch_size, steps)

    idx = idx.reshape(1, 1, -1, numel_patch).long()
    idx = idx.expand(*image.shape[:2], -1, -1)
    idx = idx.movedim(2, -1).reshape(*image.shape[:2], -1)

    output = torch.zeros(*image.shape[:2], numel_image, device=image.device, dtype=image.dtype)
    output.scatter_add_(2, idx, image)
    output = output.reshape(*image.shape[:2], *image_size)

    return output

def sliding_window_inference(image,patch_size,predictor,overlap = 0.25,
                             sw_batch_size=None, sw_device=None,device=None,
                             mode='gaussian',sigma_scale=0.125,importance_map=None,**kwargs):

    if sw_device is None:
        sw_device = image.device
    if device is None:
        device = image.device
    if importance_map is None:
        importance_map = compute_importance_map(patch_size, mode=mode,sigma_scale=sigma_scale,device=device,dtype=torch.float32)

    # Compute step size of the sliding window
    steps = [int(i*overlap) for i in patch_size]
    image_size = image.shape[2:]

    # Compute size so the image is at least as big as the patch
    image_size_padded = [max(p,i) for p,i in zip(patch_size,image_size)]
    # Compute a size so that when applying the sliding window there are no pixels left outside
    image_size_padded = [d+s-(d-p)%s if (p != d) and ((d-p)%s != 0) else d for d,s,p in zip(image_size_padded,steps,patch_size)]
    image_padded,slicer = pad_nd_image(image=image,new_shape=image_size_padded,return_slicer=True)

    # Unfold the image to apply predictor
    image_padded_unfolded = image_padded.clone()
    image_padded_unfolded = unfold(image_padded_unfolded,patch_size,steps)
    # unfold returns (bs,c,sw1,sw2,...,d1,d2,...), being (d1,d2,...) patch size and (sw1, sw2,...) the num of windows
    bs_,c_,*other_d = image_padded_unfolded.shape
    sliding_d = other_d[:-len(patch_size)]

    # (bs,c,sw1,sw2,...,d1,d2,...) > (bs * sw1 * sw2 *...,c,d1,d2,...)
    image_padded_unfolded = image_padded_unfolded.reshape(bs_,c_,-1,*patch_size)
    image_padded_unfolded = image_padded_unfolded.movedim(2,1)
    image_padded_unfolded = image_padded_unfolded.reshape(-1,c_,*patch_size)
    # Get prediction. To supporrt multiple outputs as generated by pipeline turn the items into a dict
    if sw_batch_size is None:
        pred = predictor(image_padded_unfolded.to(sw_device),**kwargs)
        if isinstance(pred, torch.Tensor):
            outputs_unfolded = {None:pred.to(device)}
        else:
            outputs_unfolded = {k:v for k,v in pred.items() if v is not None}
    else:
        outputs_unfolded = {}
        for i in range(0,image_padded_unfolded.shape[0],sw_batch_size):
            pred = predictor(image_padded_unfolded[i:i + sw_batch_size].to(sw_device),**kwargs)
            if isinstance(pred, torch.Tensor):
                pred = {None: pred.to(device)}
            else:
                pred = {k: v.to(device) for k, v in pred.items() if v is not None}
            for k,v in pred.items():
                if k not in outputs_unfolded.keys():
                    outputs_unfolded[k] = []
                outputs_unfolded[k].append(v)

        outputs_unfolded = {k:torch.cat(v, 0) for k,v in outputs_unfolded.items()}

    outputs = {}
    for k, output_unfolded in outputs_unfolded.items():
        # (bs * sw1 * sw2 *...,c,d1,d2,...) > (bs, sw1, sw2,...,c,d1,d2,...)  As expected by fold
        output_unfolded = output_unfolded.reshape(bs_,*sliding_d,c_,*patch_size)
        # Expand so the size of importance map so it matches the output.
        weights_unfolded = importance_map.clone()
        weights_unfolded = weights_unfolded.reshape(
            list((1,) * (len(output_unfolded.shape) - len(patch_size))) + patch_size)
        weights_unfolded = weights_unfolded.expand(
            list(output_unfolded.shape[:-len(patch_size)]) + list((-1,) * len(patch_size)))

        # Apply importance map
        output_unfolded *= weights_unfolded

        # Fold the output and normalize by the weights
        output = fold(output_unfolded,image_size_padded,patch_size,steps)
        weights = fold(weights_unfolded,image_size_padded,patch_size,steps)

        output /= weights

        outputs[k] = output

    # Crop the output to the original size
    outputs = {k:v[slicer] for k,v in outputs.items()}

    # Unpack if needed (i.e. the output was just a Tensor)
    if (len(outputs) == 1) and (None in outputs):
        outputs = outputs[None]

    return outputs