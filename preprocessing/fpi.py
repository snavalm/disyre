import random
import numpy as np

from batchgenerators.transforms.abstract_transforms import AbstractTransform,Compose
from batchgenerators.transforms.crop_and_pad_transforms import RandomCropTransform
from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform
from batchgenerators.transforms.color_transforms import GammaTransform, BrightnessTransform, ClipValueRange
from preprocessing.utils import CropForeground

class FPI( AbstractTransform ):

    def __init__(self,
                 image_key = "data",
                 shape_key = "shape",
                 fp_key = "fp",
                 output_key = "data",
                 p_anomaly = 0.3,
                 alpha_range = [0,1],
                 normalize_fp = None,
                 anom_patch_size = [64,64,64],
                 fp_augmentations=False,
                 size_cache = 512, # This must be > batch_size!
                 anomaly_interpolation = 'linear',
                 ):
        """
        Args:
            keys: keys of the corresponding items to be sampled from.
            anomaly_interpolation: interpolation method for the anomaly. Can be 'linear' or 'poisson'.
                                    If poisson please install pie-torch: https://pypi.org/project/pie-torch/
        """
        self.p = p_anomaly
        self.image_key = image_key
        self.shape_key = shape_key
        self.fp_key = fp_key
        self.output_key = output_key

        self.alpha_range = alpha_range
        self.size_cache = size_cache

        self.anom_patch_size = anom_patch_size
        self.spatial_dims = len(anom_patch_size)

        self.fp_augmentations = fp_augmentations
        self.cache_patches = []

        self.anomaly_interpolation = anomaly_interpolation
        self.normalize_fp = 'mean' if normalize_fp == True else normalize_fp

        assert anomaly_interpolation in ['linear','poisson','none'], "Anomaly interpolation method not recognized. Please choose between 'linear' and 'poisson'."

        if anomaly_interpolation == 'poisson':
            try:
                self.pietorch = __import__('pietorch')
            except:
                raise ImportError("Please install pie-torch: https://pypi.org/project/pie-torch/")

        transforms_fp  = [CropForeground(key_input=self.image_key),
                          RandomCropTransform(crop_size=self.anom_patch_size)
                          ]

        if self.fp_augmentations:
            transforms_fp.extend( [BrightnessTransform( mu = 0.0, sigma=0.1, p_per_sample = 0.2 ),
                                   GaussianNoiseTransform( noise_variance = (0.,0.2), p_per_sample = 0.2 ),
                                   GammaTransform( gamma_range = (0.5, 2), p_per_sample = 0.2 ),
                                   ClipValueRange( min = 0.0, max = 1.0, ),
                                   ] )

        self.patch_sampler = Compose(transforms_fp)

    def push_patch_to_cache(self, patch):
        for p in patch:
            self.cache_patches.append( p )
        # if cache is larger than textures_cache_len, get rid of some from the beginning
        while self.size_cache < len( self.cache_patches ):
            self.cache_patches.pop( 0 )

    def randomize(self ):

        do_anomaly = random.random() < self.p

        if do_anomaly:
            # Pull a foreign patch from cache
            fp = self.cache_patches.pop( random.randrange( len( self.cache_patches ) ) )
            alpha = random.random() * (self.alpha_range[1] - self.alpha_range[0]) + self.alpha_range[0]
            return do_anomaly, fp, alpha
        else:
            return do_anomaly, None, np.array(0.)

    def __call__(self, **data_dict):

        inserted = 0
        while len(self.cache_patches) < self.size_cache:
            patch = self.patch_sampler(**data_dict)
            self.push_patch_to_cache(patch[self.image_key])
            # Avoid filling the whole cache with the same image initially
            inserted +=1
            if inserted >= 2:
                break

        fp_big_s = []
        output_s = []
        alpha_s = []

        for n, (image,shape,anom_center) in enumerate(zip(data_dict[self.image_key],data_dict[self.shape_key], data_dict['anom_center'])):

            # Generate set of patch centres and alpha for the interpolation
            do_anomaly, fp, alpha = self.randomize()

            # Initialize keys with Foreign Patch and Masks
            fp_big = np.zeros_like( image )
            output = image.copy()

            if do_anomaly:

                # Create fp
                if self.spatial_dims == 3:
                    fp_big[:, anom_center[0] - self.anom_patch_size[0] // 2:anom_center[0] + self.anom_patch_size[0] // 2,
                    anom_center[1] - self.anom_patch_size[1] // 2:anom_center[1] + self.anom_patch_size[1] // 2,
                    anom_center[2] - self.anom_patch_size[2] // 2:anom_center[2] + self.anom_patch_size[2] // 2] = fp

                elif self.spatial_dims == 2:
                    fp_big[:, anom_center[0] - self.anom_patch_size[0] // 2:anom_center[0] + self.anom_patch_size[0] // 2,
                    anom_center[1] - self.anom_patch_size[1] // 2:anom_center[1] + self.anom_patch_size[1] // 2] = fp

                mask_non_zeros = shape > 0

                if self.normalize_fp:

                    if mask_non_zeros.any():
                        masked_voxels_image = image[mask_non_zeros]
                        masked_voxels_fp = fp_big[mask_non_zeros]
                        fp_mean, src_mean = masked_voxels_fp.mean(), masked_voxels_image.mean()

                        if self.normalize_fp == 'mean':
                            fp_big = fp_big - fp_mean + src_mean
                        elif self.normalize_fp == 'std':
                            fp_std, src_std = masked_voxels_fp.std(), masked_voxels_image.std()
                            fp_std = fp_std if not np.isnan(fp_std) and fp_std > 0 else 1
                            src_std = src_std if not np.isnan(src_std) and src_std > 0 else 1

                            fp_big = (fp_big - fp_mean) / (fp_std + 1e-3)
                            fp_big = (fp_big * src_std) + src_mean
                        elif self.normalize_fp == 'minmax':
                            fp_min, src_min = masked_voxels_fp.min(), masked_voxels_image.min()
                            fp_max, src_max = masked_voxels_fp.max(), masked_voxels_image.max()

                            fp_big = (fp_big - fp_min) / (fp_max - fp_min + 1e-3)
                            fp_big = (fp_big * (src_max - src_min)) + src_min

                        fp_big = fp_big.clip(0., 1.)

                # Interpolation with foreign patch
                if self.anomaly_interpolation == 'linear':
                    output = image * (1 - shape * alpha) + fp_big * shape * alpha

                    # If the input is equal the ouptut, correct alpha so it's 0
                    if (output == image).all().item():
                        alpha = np.array(0.)

                elif self.anomaly_interpolation == 'poisson':
                    raise NotImplementedError("Poisson interpolation not implemented yet.")
                    # Relook at this, since might not working in numpy, was just copied from previous preprocessing in torch
                    # data_dict[self.output_key].data = self.pietorch.blend(target=data_dict[self.image_key],
                    #                                                  source=data_dict[self.fp_key],
                    #                                                  mask=data_dict[self.shape_key][0], # Blend expects no channel dimension
                    #                                                  corner_coord=np.array((0,)*self.spatial_dims),
                    #                                                  mix_gradients=True, channels_dim=0)

                    # data_dict[self.output_key].data = data_dict[self.output_key].clamp(0., 1.)
                    #
                    # # This is different from the origianl PII. Alpha is a linear interpolation of corrupted and orinal image
                    # d[self.output_key] = d[self.output_key] * (1 - self.alpha) + d[self.output_key] * self.alpha
                    #
                    # if (d[self.output_key] == d[self.image_key]).all().item():
                    #     d['alpha_texture'] = np.array(0.)


            fp_big_s.append( fp_big )
            output_s.append( output )
            alpha_s.append( alpha )

        # Initialize keys with Foreign Patch and Masks
        data_dict[self.fp_key] = np.stack( fp_big_s )
        data_dict[self.output_key] = np.stack( output_s )
        data_dict['alpha_texture'] = np.stack( alpha_s )

        return data_dict
