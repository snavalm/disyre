# Copyright 2024 Sergio Naval Marimont. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# DISCLAIMER: This file is strongly influenced by https://github.com/ermongroup/ddim

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np
import torch

from diffusers import DiffusionPipeline, ImagePipelineOutput
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.utils import BaseOutput
from diffusers.schedulers.scheduling_utils import KarrasDiffusionSchedulers, SchedulerMixin

from diffusion.sliding_window_inference import sliding_window_inference

from torchmetrics import StructuralSimilarityIndexMeasure as SSIM
@dataclass
class DISYRESchedulerOutput(BaseOutput):
    """
    Output class for the scheduler's `step` function output.

    Args:
        prev_sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)` for images):
            Computed sample `(x_{t-1})` of previous timestep. `prev_sample` should be used as next model input in the
            denoising loop.
        pred_original_sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)` for images):
            The predicted denoised sample `(x_{0})` based on the model output from the current timestep.
            `pred_original_sample` can be used to preview progress or for guidance.
    """

    prev_sample: torch.FloatTensor
    pred_original_sample: Optional[torch.FloatTensor] = None
    anomaly_score: Optional[torch.FloatTensor] = None

class DISYREPipelineOutput(BaseOutput):
    """
    Output class for the scheduler's `step` function output.

    Args:
        prev_sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)` for images):
            Computed sample `(x_{t-1})` of previous timestep. `prev_sample` should be used as next model input in the
            denoising loop.
        pred_original_sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)` for images):
            The predicted denoised sample `(x_{0})` based on the model output from the current timestep.
            `pred_original_sample` can be used to preview progress or for guidance.
    """

    image: Optional[Union[np.array,torch.Tensor]] = None
    anomaly_score: Optional[Union[np.array,torch.Tensor]] = None

class DISYREScheduler(SchedulerMixin, ConfigMixin):
    """
    `DDPMScheduler` explores the connections between denoising score matching and Langevin dynamics sampling.

    This model inherits from [`SchedulerMixin`] and [`ConfigMixin`]. Check the superclass documentation for the generic
    methods the library implements for all schedulers such as loading and saving.

    Args:
        num_train_timesteps (`int`, defaults to 1000):
            The number of diffusion steps to train the model.
        beta_start (`float`, defaults to 0.0001):
            The starting `beta` value of inference.
        beta_end (`float`, defaults to 0.02):
            The final `beta` value.
        beta_schedule (`str`, defaults to `"linear"`):
            The beta schedule, a mapping from a beta range to a sequence of betas for stepping the model. Choose from
            `linear`, `scaled_linear`, or `squaredcos_cap_v2`.
        trained_betas (`np.ndarray`, *optional*):
            An array of betas to pass directly to the constructor without using `beta_start` and `beta_end`.
        variance_type (`str`, defaults to `"fixed_small"`):
            Clip the variance when adding noise to the denoised sample. Choose from `fixed_small`, `fixed_small_log`,
            `fixed_large`, `fixed_large_log`, `learned` or `learned_range`.
        clip_sample (`bool`, defaults to `True`):
            Clip the predicted sample for numerical stability.
        clip_sample_range (`float`, defaults to 1.0):
            The maximum magnitude for sample clipping. Valid only when `clip_sample=True`.
        prediction_type (`str`, defaults to `epsilon`, *optional*):
            Prediction type of the scheduler function; can be `epsilon` (predicts the noise of the diffusion process),
            `sample` (directly predicts the noisy sample`) or `v_prediction` (see section 2.4 of [Imagen
            Video](https://imagen.research.google/video/paper.pdf) paper).
        thresholding (`bool`, defaults to `False`):
            Whether to use the "dynamic thresholding" method. This is unsuitable for latent-space diffusion models such
            as Stable Diffusion.
        dynamic_thresholding_ratio (`float`, defaults to 0.995):
            The ratio for the dynamic thresholding method. Valid only when `thresholding=True`.
        sample_max_value (`float`, defaults to 1.0):
            The threshold value for dynamic thresholding. Valid only when `thresholding=True`.
        timestep_spacing (`str`, defaults to `"leading"`):
            The way the timesteps should be scaled. Refer to Table 2 of the [Common Diffusion Noise Schedules and
            Sample Steps are Flawed](https://huggingface.co/papers/2305.08891) for more information.
        steps_offset (`int`, defaults to 0):
            An offset added to the inference steps, as required by some model families.
        rescale_betas_zero_snr (`bool`, defaults to `False`):
            Whether to rescale the betas to have zero terminal SNR. This enables the model to generate very bright and
            dark samples instead of limiting it to samples with medium brightness. Loosely related to
            [`--offset_noise`](https://github.com/huggingface/diffusers/blob/74fd735eb073eb1d774b1ab4154a0876eb82f055/examples/dreambooth/train_dreambooth.py#L506).
    """

    _compatibles = [e.name for e in KarrasDiffusionSchedulers]
    order = 1

    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 100,
        beta_start: float = 1e-3,
        beta_end: float = 2e-1,
        beta_schedule: str = "linear",
        trained_betas: Optional[Union[np.ndarray, List[float]]] = None,
        clip_sample: bool = True,
        prediction_type: str = "epsilon",
        thresholding: bool = False,
        dynamic_thresholding_ratio: float = 0.995,
        clip_sample_range: float = 1.0,
        sample_max_value: float = 1.0,
        timestep_spacing: str = "trailing",
        steps_offset: int = 0,
        anomaly_score_ssim: bool = False,
    ):
        if trained_betas is not None:
            self.betas = torch.tensor(trained_betas, dtype=torch.float32)
        elif beta_schedule == "linear":
            self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=torch.float32)
        elif beta_schedule == "scaled_linear":
            # this schedule is very specific to the latent diffusion model.
            self.betas = torch.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps, dtype=torch.float32) ** 2
        elif beta_schedule == "sigmoid":
            # GeoDiff sigmoid schedule
            betas = torch.linspace(-6, 6, num_train_timesteps)
            self.betas = torch.sigmoid(betas) * (beta_end - beta_start) + beta_start
        else:
            raise NotImplementedError(f"{beta_schedule} does is not implemented for {self.__class__}")


        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.one = torch.tensor(1.0)

        # setable values
        self.custom_timesteps = False
        self.num_inference_steps = None
        self.timesteps = torch.from_numpy(np.arange(0, num_train_timesteps)[::-1].copy())

        self.anomaly_score_ssim = anomaly_score_ssim
        self.ssim = SSIM(return_full_image=True, data_range=(0,255))

    def scale_model_input(self, sample: torch.FloatTensor, timestep: Optional[int] = None) -> torch.FloatTensor:
        return sample

    def set_timesteps(
        self,
        num_inference_steps: Optional[int] = None,
        device: Union[str, torch.device] = None,
        timesteps: Optional[List[int]] = None,
    ):
        """
        Sets the discrete timesteps used for the diffusion chain (to be run before inference).

        Args:
            num_inference_steps (`int`):
                The number of diffusion steps used when generating samples with a pre-trained model. If used,
                `timesteps` must be `None`.
            device (`str` or `torch.device`, *optional*):
                The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
            timesteps (`List[int]`, *optional*):
                Custom timesteps used to support arbitrary spacing between timesteps. If `None`, then the default
                timestep spacing strategy of equal spacing between timesteps is used. If `timesteps` is passed,
                `num_inference_steps` must be `None`.

        """
        if num_inference_steps is not None and timesteps is not None:
            raise ValueError("Can only pass one of `num_inference_steps` or `custom_timesteps`.")

        if timesteps is not None:
            for i in range(1, len(timesteps)):
                if timesteps[i] >= timesteps[i - 1]:
                    raise ValueError("`custom_timesteps` must be in descending order.")

            if timesteps[0] >= self.config.num_train_timesteps:
                raise ValueError(
                    f"`timesteps` must start before `self.config.train_timesteps`:"
                    f" {self.config.num_train_timesteps}."
                )

            timesteps = np.array(timesteps, dtype=np.int64)
            self.custom_timesteps = True
        else:
            if num_inference_steps > self.config.num_train_timesteps:
                raise ValueError(
                    f"`num_inference_steps`: {num_inference_steps} cannot be larger than `self.config.train_timesteps`:"
                    f" {self.config.num_train_timesteps} as the unet model trained with this scheduler can only handle"
                    f" maximal {self.config.num_train_timesteps} timesteps."
                )

            self.num_inference_steps = num_inference_steps
            self.custom_timesteps = False

            # "linspace", "leading", "trailing" corresponds to annotation of Table 2. of https://arxiv.org/abs/2305.08891
            if self.config.timestep_spacing == "linspace":
                timesteps = (
                    np.linspace(0, self.config.num_train_timesteps - 1, num_inference_steps)
                    .round()[::-1]
                    .copy()
                    .astype(np.int64)
                )
            elif self.config.timestep_spacing == "leading":
                step_ratio = self.config.num_train_timesteps // self.num_inference_steps
                # creates integer timesteps by multiplying by ratio
                # casting to int to avoid issues when num_inference_step is power of 3
                timesteps = (np.arange(0, num_inference_steps) * step_ratio).round()[::-1].copy().astype(np.int64)
                timesteps += self.config.steps_offset
            elif self.config.timestep_spacing == "trailing":
                step_ratio = self.config.num_train_timesteps / self.num_inference_steps
                # creates integer timesteps by multiplying by ratio
                # casting to int to avoid issues when num_inference_step is power of 3
                timesteps = np.round(np.arange(self.config.num_train_timesteps, 0, -step_ratio)).astype(np.int64)
                timesteps -= 1
            else:
                raise ValueError(
                    f"{self.config.timestep_spacing} is not supported. Please make sure to choose one of 'linspace', 'leading' or 'trailing'."
                )

        self.timesteps = torch.from_numpy(timesteps).to(device)
    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: int,
        sample: torch.FloatTensor,
        generator=None,
        return_dict: bool = True,
        cold_diffusion_sampling: bool = True,
    ) -> Union[DISYRESchedulerOutput, Tuple]:
        """
        Predict the sample from the previous timestep by reversing the SDE. This function propagates the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            model_output (`torch.FloatTensor`):
                The direct output from learned diffusion model.
            timestep (`float`):
                The current discrete timestep in the diffusion chain.
            sample (`torch.FloatTensor`):
                A current instance of a sample created by the diffusion process.
            generator (`torch.Generator`, *optional*):
                A random number generator.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~schedulers.scheduling_ddpm.DDPMSchedulerOutput`] or `tuple`.

        Returns:
            [`~schedulers.scheduling_ddpm.DDPMSchedulerOutput`] or `tuple`:
                If return_dict is `True`, [`~schedulers.scheduling_ddpm.DDPMSchedulerOutput`] is returned, otherwise a
                tuple is returned where the first element is the sample tensor.

        """
        t = timestep

        prev_t = self.previous_timestep(t)

        # 1. compute alphas, betas
        alpha_prod_t = self.sqrt_alphas_cumprod.flip(0)[t]
        alpha_prod_t_prev =  self.sqrt_alphas_cumprod.flip(0)[prev_t] if prev_t >= 0 else (self.one - 1)

        # 2. compute predicted original sample from predicted noise also called
        x0_bar = model_output

        if self.anomaly_score_ssim:
            if self.ssim.device != x0_bar.device:
                self.ssims = self.ssim.to(x0_bar.device)
            anomaly_score = 1 - self.ssim(x0_bar, sample)[1]
        else:
            anomaly_score = (sample - x0_bar).abs()

        if prev_t <= 0:
            x0_bar.clamp_(0., 1.)
            if not return_dict:
                return (x0_bar,)

            return DISYRESchedulerOutput(prev_sample=x0_bar, pred_original_sample=x0_bar,
                                         anomaly_score=anomaly_score)

        # 3. reintroduce corruption using either cold_diffusion or naive sampling
        if cold_diffusion_sampling:
            xT_bar = (sample - x0_bar * (1 - alpha_prod_t)) / alpha_prod_t
            xt_bar = x0_bar * (1 - alpha_prod_t) + alpha_prod_t * xT_bar
            xt_sub1_bar = x0_bar * (1 - alpha_prod_t_prev) + alpha_prod_t_prev * xT_bar
            pred_prev_sample = sample - xt_bar + xt_sub1_bar
        else:
            pred_prev_sample = (1 - alpha_prod_t_prev) * x0_bar + alpha_prod_t_prev * sample

        pred_prev_sample.clamp_(0.,1.)

        if not return_dict:
            return (pred_prev_sample,)

        return DISYRESchedulerOutput(prev_sample=pred_prev_sample, pred_original_sample=x0_bar, anomaly_score=anomaly_score)


    def map_alpha_to_timestep(self, alpha: torch.FloatTensor) -> torch.LongTensor:
        """
        Maps an alpha value to the corresponding timestep in the diffusion chain.
        """
        return (self.sqrt_alphas_cumprod[None] - alpha[:,None]).abs().argmin(dim=1)

    def __len__(self):
        return self.config.num_train_timesteps

    def previous_timestep(self, timestep):
        if self.custom_timesteps:
            index = (self.timesteps == timestep).nonzero(as_tuple=True)[0][0]
            if index == self.timesteps.shape[0] - 1:
                prev_t = torch.tensor(-1)
            else:
                prev_t = self.timesteps[index + 1]
        else:
            num_inference_steps = (
                self.num_inference_steps if self.num_inference_steps else self.config.num_train_timesteps
            )
            prev_t = timestep - self.config.num_train_timesteps // num_inference_steps

        return prev_t


class DISYREPipeline(DiffusionPipeline):
    r"""
    Pipeline for image healing.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    Parameters:
        unet ([`UNet2DModel`]):
            A `UNet2DModel`/'UNet3DModel' to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image. Can be one of
            [`DDPMScheduler`], or [`DDIMScheduler`].
    """

    model_cpu_offload_seq = "unet"

    def __init__(self, unet, scheduler):
        super().__init__()
        self.register_modules(unet=unet, scheduler=scheduler)

    @torch.no_grad()
    def __call__(
            self,
            image: Union[np.ndarray,torch.Tensor],
            method: Optional[str] = "mean-single-step",
            num_inference_steps: int = 100,
            output_type: Optional[str] = "torch",
            progress_bar: Optional = False,
            weight_foreground: Optional = False,
            sw_in_inner_loop: bool = False,
            sw_kwargs: Optional = {},
    ) -> Union[ImagePipelineOutput, Tuple]:
        r"""
        The call function to the pipeline for generation.

        Args:
            image np.array with dimensions (b,c,h,w,(d))
            method (`str`, *optional*, defaults to `"mean-single-step"`):
                Choose between `"mean-single-step"`, `"single-step-restoration"`, `"multi-step-restoration"`
            batch_size (`int`, *optional*, defaults to 1):
                The number of images to generate.
            num_inference_steps (`int`, *optional*, defaults to 1000):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            sw_in_inner_loop: bool, *optional*, defaults to False.
            sw_kwargs: sliding window kwargs
        Example:

        >>> # run pipeline in inference (sample random noise and denoise)
        >>> image = pipe().images[0]

        >>> # save image
        >>> image.save("ddpm_generated_image.png")
        ```

        Returns:
            [`~DISYREPipelineOutput`]: If sliding window inference is required, only anomaly score is returned
        """

        # If the size doesn't match the sample size, use sliding window inference
        # This should have been a method in this class, but it's not :(
        if isinstance(image, list):
            if isinstance(image[0], np.ndarray):
                image = torch.from_numpy(np.stack(image)).float()
            else:
                image = torch.stack(image)
        elif isinstance(image, np.ndarray):
            image = torch.from_numpy(image).float()

        # If the image size doesn't match the image size of the model, use sliding window inference
        if (not sw_in_inner_loop) and (not all([d_img==d_net for d_img,d_net in zip(image.shape[2:],self.unet.sample_size)])):
            output = sliding_window_inference(image= image, patch_size=self.unet.sample_size,
                                                     predictor=self, **sw_kwargs,
                                                     **{"method":method,"num_inference_steps":num_inference_steps,
                                                        "weight_foreground":weight_foreground,"output_type":"torch"})

            return DISYREPipelineOutput(**output)

        image = image.to(self.device)

        if weight_foreground:
            weight = (image > 0).flatten(1).float().mean(1)[:, None, None, None]
        else:
            weight = 1

        anomaly_score = torch.zeros_like(image)

        # set step values
        self.scheduler.set_timesteps(num_inference_steps)

        timesteps = self.scheduler.timesteps
        if progress_bar:
            timesteps = self.progress_bar(timesteps)
        for t in timesteps:
            # 1. predict noise model_output
            if sw_in_inner_loop and (not all([d_img==d_net for d_img,d_net in zip(image.shape[2:],self.unet.sample_size)])):
                predictor = lambda x: self.unet(x, t).sample
                model_output = sliding_window_inference(image= image, patch_size=self.unet.sample_size,
                                                         predictor=predictor, **sw_kwargs,)
            else:
                model_output = self.unet(image, t).sample

            # 2. compute previous image: x_t -> x_t-1
            schd_output = self.scheduler.step(model_output, t, image, )
            if method == "multi-step-restoration":
                image = schd_output.prev_sample
            anomaly_score += schd_output.anomaly_score

        anomaly_score /= num_inference_steps
        anomaly_score *= weight

        # Post-process to 0 prediction on background!
        anomaly_score = anomaly_score * (image > 0).float()

        if output_type == "torch":
            return DISYREPipelineOutput(images=image, anomaly_score=anomaly_score)

        anomaly_score = anomaly_score.cpu()
        image = image.cpu()
        if output_type == "pil":
            image = self.numpy_to_pil(image.permute(0, 2, 3, 1).numpy())
        else:
            image = image.numpy()

        return DISYREPipelineOutput(images=image, anomaly_score=anomaly_score.numpy())