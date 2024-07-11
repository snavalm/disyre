import os
import numpy as np
import nibabel as nib

from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter

from batchgenerators.transforms.crop_and_pad_transforms import RandomCropTransform
from batchgenerators.transforms.spatial_transforms import MirrorTransform
from preprocessing.batchgenerators_custom import SpatialTransform_2
from batchgenerators.transforms.color_transforms import BrightnessMultiplicativeTransform, GammaTransform, ClipValueRange
from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform, GaussianBlurTransform
from batchgenerators.transforms.abstract_transforms import Compose
from batchgenerators.transforms.utility_transforms import NumpyToTensor

from batchgenerators.transforms.abstract_transforms import Compose
from preprocessing.utils import CropForeground,load_dataset_config, DataLoader3D, DataLoader2Dfrom3D, DataLoader2D, default
from preprocessing.mask_generation import GetRandomLocation, CreateRandomShape
from preprocessing.fpi import FPI
from preprocessing.bias import BiasCorruption

def get_train_transform(config, ag_transforms=[]):

    patch_size = default(config,"patch_size",None)
    anom_type = default(config,"anom_type",None)

    tr_transforms = [
        CropForeground(key_input="data", keys_to_apply=["data", ], ),
        SpatialTransform_2(
            patch_size, [p//2 for p in patch_size ],
            do_elastic_deform=True,
            deformation_scale=(0, 0.25),
            do_rotation=True,
            angle_x=(- 15 / 360. * 2 * np.pi, 15 / 360. * 2 * np.pi),
            angle_y=(- 15 / 360. * 2 * np.pi, 15 / 360. * 2 * np.pi),
            angle_z=(- 15 / 360. * 2 * np.pi, 15 / 360. * 2 * np.pi),
            do_scale=True, scale=(0.75, 1.25),
            border_mode_data='constant', border_cval_data=0,
            border_mode_seg='constant', border_cval_seg=0,
            order_seg=1, order_data=3,
            random_crop=True,
            p_el_per_sample=0.1, p_rot_per_sample=0.1, p_scale_per_sample=0.1
        ),
        MirrorTransform(axes=(0, 1, 2)),
    ]

    tr_transforms.extend([
        BrightnessMultiplicativeTransform((0.9, 1.1), per_channel=True, p_per_sample=0.15),
        GammaTransform(gamma_range=(0.5, 2), invert_image=False, per_channel=True, p_per_sample=0.15),
        GammaTransform(gamma_range=(0.5, 2), invert_image=True, per_channel=True, p_per_sample=0.15),
        GaussianNoiseTransform(noise_variance=(0, 0.05), p_per_sample=0.15),
        GaussianBlurTransform(blur_sigma=(0.5, 1.5), different_sigma_per_channel=True, p_per_channel=0.5, p_per_sample=0.15),
        ClipValueRange(min=0.0, max=1.0, ),
    ])

    tr_transforms.extend(ag_transforms)

    keys_to_torch = ['data', 'data_c', ]
    if anom_type in ['dag','dag_no_quant','fpi']:
        keys_to_torch += ['alpha_texture']
    if anom_type in ['dag','dag_no_quant','bias_only']:
        keys_to_torch += ['alpha_bias']

    tr_transforms.append(NumpyToTensor(keys=keys_to_torch,cast_to='float'))

    tr_transforms = Compose(tr_transforms)

    return tr_transforms

def get_ag_transforms(type='dag', anom_patch_size=[64,64,64], no_mask_in_background=False, shape_dataset=None,
                      p_anomaly=1.0,
                      quantized_bias_codebook=[0.00565079, 0.38283867, 0.61224216, 0.76920754, 0.9385473],
                      randomshape_kwargs={}, fpi_kwargs={}, bias_kwargs={}):

    assert type in ['dag','fpi','dag_no_quant','bias_only']

    if shape_dataset is not None and "randommask_kwargs" not in randomshape_kwargs:
        randomshape_kwargs["randommask_kwargs"] = {"dataset": shape_dataset, "spatial_prob": 1.0, "scale_masks": (0.5, 0.75)}


    ag_transforms = [
        GetRandomLocation(anom_patch_size=anom_patch_size),
        CreateRandomShape(anom_patch_size=anom_patch_size, smooth_prob=1.0,
                          no_mask_in_background=no_mask_in_background, **randomshape_kwargs),
        ]

    # Override with None if not needed
    quantized_bias_codebook = quantized_bias_codebook if type in ['dag','bias_only'] else None

    if type == 'fpi':
        ag_transforms.append(
            FPI(image_key='data', anomaly_interpolation='linear', output_key="data_c", p_anomaly=p_anomaly,
                anom_patch_size=anom_patch_size, normalize_fp=False, **fpi_kwargs))


    elif type in ['dag','dag_no_quant']:
        ag_transforms.extend([
            FPI(image_key='data', anomaly_interpolation='linear', output_key="data_c", p_anomaly=p_anomaly,
                anom_patch_size=anom_patch_size, normalize_fp="minmax", **fpi_kwargs),
            BiasCorruption(image_key="data_c", shape_key="shape", output_key="data_c", p_anomaly=p_anomaly,
                           quantized_bias_codebook=quantized_bias_codebook, quantized_mask_key="shape_bias", **bias_kwargs)
        ])

    elif type == 'bias_only':
        ag_transforms.extend([
            BiasCorruption(image_key="data", shape_key="shape", output_key="data_c", p_anomaly=p_anomaly,
                           quantized_bias_codebook=quantized_bias_codebook, quantized_mask_key="shape_bias", **bias_kwargs)
        ])

    return ag_transforms

def create_dataloader(config_file):

    config = load_dataset_config(config_file)
    output_mode = default(config, "output_mode", "2Dfrom3D")
    assert output_mode in ["3D", "2D","2Dfrom3D"]

    patch_size = default(config,"patch_size",None)

    if config.dataset_type == "training":
        ag_transforms = get_ag_transforms(default(config,"anom_type","dag"),
                                          default(config,"anom_patch_size",[64,64]),
                                          default(config, "no_anom_in_background", True),
                                          default(config,"shape_dataset",None),
                                          default(config,"p_anomaly",1.0),
                                          default(config,"quantized_bias_codebook",[0.00565079, 0.38283867, 0.61224216, 0.76920754, 0.9385473]),
                                          default(config,"randomshape_kwargs",{}),
                                          default(config,"fpi_kwargs",{}),
                                          default(config,"bias_kwargs",{}),
                                          )

        tr_transforms = get_train_transform(config, ag_transforms)
    elif config.dataset_type == "evaluation_image":
        tr_transforms = [CropForeground(key_input="data", keys_to_apply=["data",], ),]
        if patch_size:
            tr_transforms.append(RandomCropTransform(patch_size, [p//2 for p in patch_size], label_key=None ))
        tr_transforms += [NumpyToTensor(keys=['data',], cast_to='float')]
        tr_transforms = Compose(tr_transforms)

    else:
        tr_transforms = [CropForeground(key_input="data", keys_to_apply=["data", "seg"], ),]
        if patch_size:
            tr_transforms.append(RandomCropTransform(patch_size, [p//2 for p in patch_size] ))
        tr_transforms += [NumpyToTensor(keys=['data','seg'], cast_to='float')]
        tr_transforms = Compose(tr_transforms)

    if hasattr(config,"datalist_key_val"):
        datalist_keys = [config.datalist_key, config.datalist_key_val]
    else:
        datalist_keys = [config.datalist_key]

    # Get the number of cput available in host
    num_workers = os.cpu_count()
    num_workers = min(num_workers,default(config,"num_workers",24))

    loaders = []
    for i_d, datalist_key in enumerate(datalist_keys):
        if output_mode == "3D":
            dataloader_train = DataLoader3D(getattr(config,f"{config.dataset_name}_json"), datalist_key, "image",
                                            datalist_base_dir=default(config,f"{config.dataset_name}_base_dir",None),
                                            batch_size=config.batch_size,
                                            num_threads_in_multithreaded=num_workers,
                                            intensity_percentile_scaling = default(config, "intensity_percentile_scaling", [0, .98]),
                                            keys_label="label" if config.dataset_type == "evaluation" else None,
                                            infinite= (config.dataset_type == 'training') and (i_d == 0),
                                            meta_keys=default(config, "meta_keys",None),
                                            )
        elif output_mode == "2Dfrom3D":
            dataloader_train = DataLoader2Dfrom3D(getattr(config,f"{config.dataset_name}_json"), datalist_key, "image",
                                                  datalist_base_dir=default(config, f"{config.dataset_name}_base_dir", None),
                                                  batch_size=config.batch_size,
                                                  num_slices=default(config, "num_slices", None),
                                                  num_threads_in_multithreaded=num_workers,
                                                  intensity_percentile_scaling=default(config,"intensity_percentile_scaling",[0, .98]),
                                                  keys_label="label" if config.dataset_type == "evaluation" else None,
                                                  infinite=(config.dataset_type == 'training') and (i_d == 0),
                                                  meta_keys=default(config, "meta_keys", None),
                                                  )
        elif output_mode == "2D":
            dataloader_train = DataLoader2D(getattr(config, f"{config.dataset_name}_json"), datalist_key, "image",
                                            datalist_base_dir=default(config, f"{config.dataset_name}_base_dir", None),
                                            batch_size=config.batch_size,
                                            num_threads_in_multithreaded=num_workers,
                                            intensity_percentile_scaling=default(config, "intensity_percentile_scaling", [0, .98]),
                                            keys_label="label" if config.dataset_type == "evaluation" else None,
                                            infinite=(config.dataset_type == 'training') and (i_d == 0),
                                            image_loader=default(config, "image_loader", "nibabel"),
                                            meta_keys=default(config, "meta_keys",None),
                                            )
        else:
            raise NotImplementedError(f"Output {output_mode} not implemented yet, only ['3D','2D','2Dfrom3D'] are supported.")


        tr_gen = MultiThreadedAugmenter(dataloader_train, tr_transforms,
                                        num_processes=num_workers,
                                        num_cached_per_queue=3,
                                        seeds=default(config,"seeds",None),
                                        pin_memory=True)

        tr_gen.name = config.dataset_name + "_" + config.datalist_key
        loaders.append(tr_gen)

    # Ensure that if we are in training mode, we have two loaders
    if (config.dataset_type == "training") and (not hasattr(config,"datalist_key_val")):
        loaders.append(None)

    return *(loaders), config
