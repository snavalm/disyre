import random
import numpy as np
from scipy.cluster import vq
from batchgenerators.transforms.abstract_transforms import AbstractTransform
from batchgenerators.augmentations.noise_augmentations import augment_gaussian_blur
def quantize(img,codebook):
    img_q= vq.vq(img.flatten(),code_book=codebook)[0]
    img_q = img_q.reshape(img.shape)
    return img_q
class BiasCorruption(AbstractTransform):

    def __init__(self,
                 image_key= "data",
                 shape_key = "shape",
                 output_key = "data",
                 p_anomaly=0.3,
                 alpha_range=[0, 1],
                 quantized_bias_codebook = None,
                 quantized_mask_key = "mask_bias",
                 ):
        """
        Args:
        """
        self.image_key = image_key
        self.shape_key = shape_key
        self.output_key = output_key
        self.alpha_range = alpha_range
        self.p = p_anomaly

        self.quantized_mask_key = quantized_mask_key

        if quantized_bias_codebook is not None:
            self.quantized_bias_codebook = np.array(quantized_bias_codebook)
            self.codebook_k = len(self.quantized_bias_codebook)
            self.eye_q = np.eye(self.codebook_k)

        else:
            self.quantized_bias_codebook = None

    def randomize(self, image, shape):

        do_anomaly = random.random() < self.p

        if do_anomaly:
            # Generate a mask
            alpha = random.random() * (self.alpha_range[1] - self.alpha_range[0]) + self.alpha_range[0]
            bias_sign = 1 if (random.random() < 0.5) else -1

            if self.quantized_bias_codebook is not None:
                image_q = quantize(image,self.quantized_bias_codebook)
                image_q = np.take(self.eye_q, image_q, axis=0)
                # Compute the pixels for each centroid, the proportion pix_per_centroid / total pixles is used a prob to select centroid
                voxels_per_q = np.matmul((shape > 0).flatten()[None].astype('int'),
                                            image_q.reshape(-1, self.codebook_k))
                voxels_per_q = voxels_per_q.squeeze(0)
                centroids_selected = np.random.rand(self.codebook_k) < ( voxels_per_q / (voxels_per_q.sum() + 1e3))
                # If all centroids are False set all to True
                centroids_selected[centroids_selected.sum() == 0] = True
                mask_ranges = np.matmul(np.expand_dims(image_q,-2),
                                        centroids_selected[None, None, None, :, None].astype('int')).squeeze(-1).squeeze(-1)

                #Smooth the mask to avoid sharp transitions
                mask_bias = augment_gaussian_blur(mask_ranges.astype(np.float32)[None], sigma_range=(1,1))[0]
            else:
                mask_bias = 1.0

            return do_anomaly, alpha, np.array(bias_sign), mask_bias
        else:
            return do_anomaly, 0., np.array(0.), np.zeros_like(shape)
    def __call__(self, **data_dict):

        mask_bias_s = []
        outputs_s = []
        alpha_s = []

        for n, (image, shape) in enumerate(zip(data_dict[self.image_key], data_dict[self.shape_key])):
            do_anomaly, alpha, bias_sign, mask_bias = self.randomize(image, shape)
            bias_shape = shape.copy()

            if do_anomaly:
                bias_shape *= mask_bias
                bias_corrupted = image * (1 - bias_shape) + (image + alpha * bias_sign) * bias_shape
                bias_corrupted = np.clip(bias_corrupted, 0., 1.)

            else:
                bias_corrupted = image

            mask_bias_s.append(bias_shape)
            outputs_s.append(bias_corrupted)
            alpha_s.append(np.abs(image-bias_corrupted).max())

        data_dict[self.output_key] = np.stack(outputs_s)
        data_dict[self.quantized_mask_key] = np.stack(mask_bias_s)
        data_dict['alpha_bias'] = np.array(alpha_s)

        return data_dict
