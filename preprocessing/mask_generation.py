
import random
import numpy as np
from batchgenerators.augmentations.spatial_transformations import augment_spatial_2
from batchgenerators.augmentations.noise_augmentations import augment_gaussian_blur

from batchgenerators.transforms.abstract_transforms import AbstractTransform

class MaskGenerator():
    def __init__(self,
                 patch_size,
                 spatial_prob = 0.0,
                 smooth_prob = 0.0):
        self.patch_size = patch_size
        self.spatial_dims = len(patch_size)
        self.spatial_prob = spatial_prob
        self.smooth_prob = smooth_prob
        self.scale = (1.,1.) # This is here to enable resaling masks

    def generate_mask(self,*args,**kwargs):
        out = np.ones(1,*self.patch_size)
        return out

    def random_spatial(self, mask):
        rot_angle = 15 / 360. * 2 * np.pi
        rot_angle = (-rot_angle,rot_angle)
        mask,_ = augment_spatial_2(
            mask[None], # add batch dimension
            None,
            self.patch_size,
            0,
            angle_x=rot_angle,
            angle_y=rot_angle,
            angle_z=rot_angle,
            scale=self.scale,
            do_elastic_deform=False,
            random_crop=False,
            p_el_per_sample=0.,
            p_rot_per_sample=self.spatial_prob,
            p_scale_per_sample=1.0,
        )
        # Not sure why, but 'nearest' is still producing  some background artifacts
        mask = (mask > 0.5).astype('float')
        return mask[0] # remove batch dimension

    def random_smooth(self, mask):
        mask = augment_gaussian_blur(mask,
                                     sigma_range=(0, 5),
                                     p_per_channel= 0.7
                                     )
        return mask

    def __call__(self, *args, **kwargs):

        out = self.generate_mask(*args,**kwargs)
        if self.spatial_prob > 0:
            out = self.random_spatial(out)
        if self.smooth_prob > 0:
            out = self.random_smooth(out)


        return out

class RandomSquare(MaskGenerator):
    def __init__(self,
                 patch_size,
                 spatial_prob = 0.0,
                 smooth_prob = 0.0,
                 min_anom_sizes=10,
                 max_anom_sizes=None,
                 ):
        super().__init__(patch_size,spatial_prob,smooth_prob)

        if isinstance(min_anom_sizes, (tuple,list)):
            assert len(min_anom_sizes) == self.spatial_dims, "min_anom_sizes must have the same length as spatial_dims or be an int"
        else:
            min_anom_sizes = [min_anom_sizes] * self.spatial_dims

        self.min_anom_sizes = min_anom_sizes

        max_anom_sizes = max_anom_sizes if max_anom_sizes is not None else [p-20 for p in patch_size]

        if isinstance(max_anom_sizes, (tuple,list)):
            assert len(max_anom_sizes) == self.spatial_dims, "max_anom_sizes must have the same length as spatial_dims or be an int"
        else:
            max_anom_sizes = [max_anom_sizes] * self.spatial_dims

        self.max_anom_sizes = max_anom_sizes

    def generate_mask(self,*args,**kwargs):

        anom_size = [random.randint(a,b) for a,b in zip(self.min_anom_sizes,self.max_anom_sizes)]
        anom_center = [random.randint(a_s//2 , p_s - a_s // 2 ) for a_s, p_s in zip(anom_size,self.patch_size)]

        out = np.zeros( (1, *self.patch_size) )
        if self.spatial_dims == 3:
            out[:, anom_center[0] - anom_size[0] // 2:anom_center[0] + anom_size[0] // 2,
            anom_center[1] - anom_size[1] // 2:anom_center[1] + anom_size[1] // 2,
            anom_center[2] - anom_size[2] // 2:anom_center[2] + anom_size[2] // 2] = 1

        elif self.spatial_dims == 2:
            out[:, anom_center[0] - anom_size[0] // 2:anom_center[0] + anom_size[0] // 2,
            anom_center[1] - anom_size[1] // 2:anom_center[1] + anom_size[1] // 2] = 1

        return out

class RandomSphere(MaskGenerator):
    def __init__(self,
                 patch_size,
                 spatial_prob = 0.0,
                 smooth_prob = 0.0,
                 min_anom_size=10,
                 max_anom_size=None,
                 ):
        super().__init__(patch_size, spatial_prob, smooth_prob)
        self.min_anom_size = min_anom_size
        self.max_anom_size = max_anom_size if max_anom_size is not None else min(patch_size) - 20
        self.mesh = np.stack( np.meshgrid( *[np.linspace( 0, i, i ) for i in self.patch_size], indexing ='ij' ) )


    def generate_mask(self,*args,**kwargs):
        anom_size = random.randint( self.min_anom_size, self.max_anom_size )
        anom_center = [random.randint( anom_size // 2 + 10, p_s - anom_size // 2 - 10) for p_s in self.patch_size ]
        anom_center = np.array( anom_center )[(None,) * self.spatial_dims]
        anom_center = np.moveaxis(anom_center, -1,0)

        # Create a mask with the position of the anomaly
        mask_anom = (self.mesh - anom_center)**2
        mask_anom = mask_anom.sum( 0 ) <= (anom_size//2)**2
        mask_anom = mask_anom[None]

        return mask_anom.astype('float')


class RandomMask(MaskGenerator):
    def __init__(self,
                 patch_size,
                 masks_spatial_dims = 3,
                 dataset=None,
                 num_workers=4,
                 spatial_prob = 0.0,
                 smooth_prob = 0.0,
                 scale_masks = (1.,1.),
                 ):
        """ Masks are originally about [64,64,64], use a scale factor if you need to make them bigger,
        please note that this is the inverse so 2. will make them smaller 1/2 the size and 0.5 will make them 2 times bigger
        """
        super().__init__(patch_size,spatial_prob,smooth_prob)
        self.scale = scale_masks

        self.mask_spatial_dims = masks_spatial_dims

        from preprocessing.utils import DataLoader3D

        self.shape_loader = DataLoader3D(dataset,"training","label", batch_size=1,
                                         infinite=True, num_threads_in_multithreaded=num_workers)


    def generate_mask(self,*args,**kwargs):
        mask_anom = next(self.shape_loader)
        # Get only array and remove batch dimension
        mask_anom = mask_anom['data'][0]
        mask_anom = (mask_anom > 0).astype('float')

        if (self.spatial_dims == 2) and (self.mask_spatial_dims == 3):
            return mask_anom[:,:,mask_anom.shape[2]//2]
        else:
            return mask_anom


class GetRandomLocation(AbstractTransform):
    def __init__(self,
                 image_key="data",
                 anom_patch_size=[128, 128, 128],
                 not_in_background=True,
                 ):
        self.image_key = image_key
        self.anom_patch_size = anom_patch_size
        self.not_in_background=not_in_background
    def __call__(self, **data_dict):
        # Pick a random location for the anomaly
        image = data_dict[self.image_key]
        bs, image_size = image.shape[0],image.shape[2:]

        anom_center = []
        for n in range(bs):
            if self.not_in_background:
                indices_nonzero = np.stack(np.nonzero(image[n].sum(0) > 0), axis=-1)
                margin = np.array(self.anom_patch_size)[None] // 2
                valid_indices = ((indices_nonzero >= margin) & (indices_nonzero < np.array(image_size)[None] - margin)).all(1)
                if valid_indices.sum() == 0:
                    anom_center.append(np.array(
                        [random.randint(a // 2, i - (a // 2)) for i, a in zip(image_size, self.anom_patch_size)]))
                else:
                    try:
                        anom_center.append(random.choice(indices_nonzero[valid_indices]))
                    except:
                        print(indices_nonzero[valid_indices].shape)
                        raise ValueError("No valid indices found")

            else:
                anom_center.append(np.array(
                    [random.randint(a // 2, i - (a // 2)) for i, a in zip(image_size, self.anom_patch_size)]))

        data_dict['anom_center'] = np.stack(anom_center)

        if "metadata" not in data_dict:
            data_dict["metadata"] = {}

        return data_dict


class CreateRandomShape(AbstractTransform):
    """
    p_squaremask:
    p_spheremask:
    p_randommask:
    patch_size:
    no_mask_in_background:
    square_kwargs:
    spehere_kwargs:
    randommask_kwargs:
    shape_key:
    """

    def __init__(self, p_squaremask=1, p_spheremask=1, p_randommask=1, anom_patch_size=[128,128,128],
                 square_kwargs={"spatial_prob":0.5,},
                 sphere_kwargs={"spatial_prob":0.0,},
                 randommask_kwargs=None,
                 smooth_prob = 0.0,
                 shape_key='shape',
                 data_key='data',
                 no_mask_in_background=False,):

        self.p_squaremask = p_squaremask
        self.p_spheremask = p_spheremask
        self.p_randommask = p_randommask
        self.shape_key = shape_key
        self.data_key = data_key
        self.anom_patch_size = anom_patch_size
        self.no_mask_in_background = no_mask_in_background

        if self.p_squaremask > 0:
            self.square_mask_gen = RandomSquare(anom_patch_size, smooth_prob = smooth_prob, **square_kwargs)
        else:
            self.square_mask_gen = None
        if self.p_spheremask > 0:
            self.sphere_mask_gen = RandomSphere(anom_patch_size, smooth_prob = smooth_prob, **sphere_kwargs)
        else:
            self.sphere_mask_gen = None
        if self.p_randommask > 0:
            self.random_mask_gen = RandomMask(anom_patch_size, 3, smooth_prob = smooth_prob, **randommask_kwargs)
        else:
            self.random_mask_gen = None
    def __call__(self, **data_dict):
        image = data_dict[self.data_key]
        bs, image_size = image.shape[0], image.shape[2:]

        shapes = []

        for i in range(bs):

            mg = random.choices([self.square_mask_gen, self.sphere_mask_gen, self.random_mask_gen],
                           weights=[self.p_squaremask, self.p_spheremask, self.p_randommask])

            mask_anom = mg[0]()

            assert "anom_center" in data_dict, "Anomaly center not found in data_dict. Please use GetRandomLocation transform to generate it."
            anom_center = data_dict['anom_center'][i]

            shape_i = np.zeros((1, *image_size))

            # Create mask (location of anomaly) x alpha x pattern of anomaly
            if len(self.anom_patch_size) == 3:
                shape_i[:, anom_center[0] - self.anom_patch_size[0] // 2:anom_center[0] + self.anom_patch_size[0] // 2,
                anom_center[1] - self.anom_patch_size[1] // 2:anom_center[1] + self.anom_patch_size[1] // 2,
                anom_center[2] - self.anom_patch_size[2] // 2:anom_center[2] + self.anom_patch_size[2] // 2] = mask_anom
            elif len(self.anom_patch_size) == 2:
                shape_i[:, anom_center[0] - self.anom_patch_size[0] // 2:anom_center[0] + self.anom_patch_size[0] // 2,
                anom_center[1] - self.anom_patch_size[1] // 2:anom_center[1] + self.anom_patch_size[1] // 2] = mask_anom

            # Zero the mask areas where the image is background
            if self.no_mask_in_background:
                shape_i[(image[i] == 0).all(0, keepdims=True)] = 0

            shapes.append(shape_i)

        data_dict[self.shape_key] = np.stack(shapes)
        return data_dict