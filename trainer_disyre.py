## This script is adapted from train_unconditional.py from diffuser

import argparse
import json
import logging
import os
import shutil
from datetime import timedelta
from pathlib import Path

import accelerate
import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
from preprocessing.preprocessing import create_dataloader
from huggingface_hub import create_repo, upload_folder
from packaging import version
from tqdm.auto import tqdm

import diffusers
from diffusers import UNet2DModel
from diffusion.scheduler import DISYREScheduler, DISYREPipeline
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import is_accelerate_version, is_tensorboard_available, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available

import evaluate
import environment_defaults
# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
# check_min_version("0.28.0.dev0")

logger = get_logger(__name__, log_level="INFO")

pipeline_eval_kwargs = {"num_inference_steps":5,
                  "method":"mean-single-step",
                  "output_type":"torch",
                   "weight_foreground":True,
                  }
def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    if not isinstance(arr, torch.Tensor):
        arr = torch.from_numpy(arr)
    res = arr[timesteps].float().to(timesteps.device)
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument('--json', type=str, help='Path to JSON file with experiments specifications')
    parser.add_argument(
        "--dataset_config",
        type=str,
        default=None,
        help="The path to a Dataset config.json",
    )
    parser.add_argument(
        "--model_config_name_or_path",
        type=str,
        default=None,
        help="The config of the UNet model to train, leave as None to use standard DDPM configuration.",
    )
    parser.add_argument(
        "--experiment_id",
        type=str,
        default=None,
        help="The id of the experiment. If not provided will default to json file name.",
    )
    parser.add_argument(
        "--eval_dataset_config",
        type=str,
        nargs='+',
        default=None,
        help="Path of json directories of evaluation datasets.",
    )
    parser.add_argument("--image_evaluation_only", action="store_true")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--overwrite_output_dir", action="store_true")
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--num_steps", type=int, default=100000)
    parser.add_argument("--validation_steps", type=int, default=5000, help="How often to run validation and save images during training.")
    parser.add_argument(
        "--save_model_steps", type=int, default=5000, help="How often to save the model during training."
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="cosine",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--adam_beta1", type=float, default=0.95, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument(
        "--adam_weight_decay", type=float, default=1e-6, help="Weight decay magnitude for the Adam optimizer."
    )
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer.")
    parser.add_argument(
        "--use_ema",
        action="store_true",
        help="Whether to use Exponential Moving Average for the final model weights.",
    )
    parser.add_argument("--ema_inv_gamma", type=float, default=1.0, help="The inverse gamma value for the EMA decay.")
    parser.add_argument("--ema_power", type=float, default=3 / 4, help="The power value for the EMA decay.")
    parser.add_argument("--ema_max_decay", type=float, default=0.9999, help="The maximum decay magnitude for EMA.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--hub_private_repo", action="store_true", help="Whether or not to create a private repository."
    )
    parser.add_argument(
        "--logger",
        type=str,
        default="wandb",
        choices=["tensorboard", "wandb"],
        help=(
            "Whether to use [tensorboard](https://www.tensorflow.org/tensorboard) or [wandb](https://www.wandb.ai)"
            " for experiment tracking and logging of model metrics and model checkpoints"
        ),
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose"
            "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
            "and an Nvidia Ampere GPU."
        ),
    )
    parser.add_argument("--ddpm_num_steps", type=int, default=100)
    parser.add_argument("--ddpm_num_inference_steps", type=int, default=5)
    parser.add_argument("--ddpm_beta_schedule", type=str, default="linear")
    parser.add_argument("--ddpm_beta_start", type=float, default=1e-3)
    parser.add_argument("--ddpm_beta_end", type=float, default=2e-1)
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=20000,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=10,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )

    args = parser.parse_args()
    if args.json:
        with open(args.json, 'r') as f:
            json_args = json.load(f)
        for key, value in json_args.items():
            if hasattr(args, key):
                setattr(args, key, value)

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank


    if args.experiment_id is None:
        # From "aaaa/bbb/cccc.json" json filename, get the experiment id ccc
        args.experiment_id = os.path.basename(args.json).split(".")[0] if args.json is not None else "default"

    if args.output_dir is None:
        args.output_dir = os.path.join(environment_defaults.env_default["checkpoint_dir"], args.experiment_id)

    return args


def main(args):
    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=7200))  # a big number for high resolution or big dataset
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.logger,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
    )

    if args.logger == "tensorboard":
        if not is_tensorboard_available():
            raise ImportError("Make sure to install tensorboard if you want to use it for logging during training.")

    elif args.logger == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")
        import wandb

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                if args.use_ema:
                    ema_model.save_pretrained(os.path.join(output_dir, "unet_ema"))

                for i, model in enumerate(models):
                    model.save_pretrained(os.path.join(output_dir, "unet"))

                    # make sure to pop weight so that corresponding model is not saved again
                    weights.pop()

        def load_model_hook(models, input_dir):
            if args.use_ema:
                load_model = EMAModel.from_pretrained(os.path.join(input_dir, "unet_ema"), UNet2DModel)
                ema_model.load_state_dict(load_model.state_dict())
                ema_model.to(accelerator.device)
                del load_model

            for i in range(len(models)):
                # pop models so that they are not loaded again
                model = models.pop()

                # load diffusers style into model
                load_model = UNet2DModel.from_pretrained(input_dir, subfolder="unet")
                model.register_to_config(**load_model.config)

                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        diffusers.utils.logging.set_verbosity_info()
    else:
        diffusers.utils.logging.set_verbosity_error()

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id

    train_dataloader, val_loader, dataset_config = create_dataloader(args.dataset_config)
    args.train_dataset_config = dataset_config

    eval_loaders = [create_dataloader(edset)[0] for edset in args.eval_dataset_config] if args.eval_dataset_config else None

    # Initialize the model
    if args.model_config_name_or_path is None:
        model = UNet2DModel(
            sample_size=args.train_dataset_config.patch_size,
            in_channels=args.train_dataset_config.num_channels,
            out_channels=args.train_dataset_config.num_channels,
            layers_per_block=2,
            block_out_channels=(32, 64, 96, 128, 256, 256), # These have changed so it matches original experiments
            down_block_types=(
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "AttnDownBlock2D",
                "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D",
                "AttnUpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
            ),
        )
    else:
        config = UNet2DModel.load_config(args.model_config_name_or_path)
        model = UNet2DModel.from_config(config)

    # Torch compile
    # model = torch.compile(model, mode="default", fullgraph=True)

    # Create EMA for the model.
    if args.use_ema:
        ema_model = EMAModel(
            model.parameters(),
            decay=args.ema_max_decay,
            use_ema_warmup=True,
            inv_gamma=args.ema_inv_gamma,
            power=args.ema_power,
            model_cls=UNet2DModel,
            model_config=model.config,
        )

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        args.mixed_precision = accelerator.mixed_precision
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        args.mixed_precision = accelerator.mixed_precision

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warning(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            model.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    # Initialize the scheduler
    noise_scheduler = DISYREScheduler(num_train_timesteps=args.ddpm_num_steps,
                                      beta_schedule=args.ddpm_beta_schedule,
                                      beta_start=args.ddpm_beta_start,
                                      beta_end=args.ddpm_beta_end,)

    # TODO Fix this!
    noise_scheduler.sqrt_alphas_cumprod = noise_scheduler.sqrt_alphas_cumprod.to(accelerator.device)

    # Initialize the optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    alphas_keys = []
    if args.train_dataset_config.anom_type in ['dag', 'fpi', 'dag_no_quant', ]:
        alphas_keys.append("alpha_texture")
    if args.train_dataset_config.anom_type in ['dag', 'dag_no_quant', 'bias_only']:
        alphas_keys.append("alpha_bias")

    # Initialize the learning rate scheduler
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.num_steps,
    )

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, lr_scheduler, noise_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler, noise_scheduler
    )

    if args.use_ema:
        ema_model.to(accelerator.device)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers(project_name="disyre",
                                  config = vars(args),
                                  init_kwargs={"wandb": {"entity": environment_defaults.env_default['wandb_entity'],
                                                         "name":args.experiment_id,}})

    per_device_batch_size = args.train_dataset_config.batch_size
    total_batch_size = args.train_dataset_config.batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    if args.train_dataset_config.output_mode == "2Dfrom3D":
        num_slices = args.train_dataset_config.num_slices if hasattr(args.train_dataset_config, "num_slices") else 1
        total_batch_size *= num_slices
        per_device_batch_size *= num_slices

    max_train_steps = args.num_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataloader.generator.datalist)}")
    logger.info(f"  Num Steps = {args.num_steps}")
    logger.info(f"  Anomaly Generation = {args.train_dataset_config.anom_type}")
    logger.info(f"  Instantaneous batch size per device = {per_device_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_train_steps}")

    global_step = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            # resume_global_step = global_step * args.gradient_accumulation_steps
            # first_epoch = global_step // num_update_steps_per_epoch
            # resume_step = resume_global_step % (num_update_steps_per_epoch * args.gradient_accumulation_steps)

    # Train!
    progress_bar = tqdm(total=max_train_steps, disable=not accelerator.is_local_main_process)
    while global_step < max_train_steps:
        progress_bar.set_description(f"Training step {global_step}")
        model.train()
        batch = next(train_dataloader)

        # to device shouldn't be required! why is this happening?
        # TODO Check if to(device) is needed? How to make it work with accelerate...
        normal_images = batch["data"].to(weight_dtype).to(accelerator.device)
        anom_images = batch["data_c"].to(weight_dtype).to(accelerator.device)

        for alpha_n, alpha_k in enumerate(alphas_keys):
            if alpha_n == 0:
                alpha_dag = batch[alpha_k].to(weight_dtype).to(accelerator.device)
            else:
                alpha_dag = torch.stack([alpha_dag,batch[alpha_k].to(weight_dtype).to(accelerator.device)])
                alpha_dag = torch.amax(alpha_dag, dim=0)

        # Get the timestep matching for a given alpha_dag
        timesteps = noise_scheduler.map_alpha_to_timestep(alpha_dag)

        with accelerator.accumulate(model):
            # Predict the noise residual
            model_output = model(anom_images, timesteps).sample
            loss = F.mse_loss(model_output.float(), normal_images.float().to(model_output.device))

            accelerator.backward(loss)

            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        # Checks if the accelerator has performed an optimization step behind the scenes
        if accelerator.sync_gradients:
            if args.use_ema:
                ema_model.step(model.parameters())
            progress_bar.update(1)
            global_step += 1

            if accelerator.is_main_process:
                if global_step % args.checkpointing_steps == 0:
                    # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                    if args.checkpoints_total_limit is not None:
                        checkpoints = os.listdir(args.output_dir)
                        checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                        checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                        # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                        if len(checkpoints) >= args.checkpoints_total_limit:
                            num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                            removing_checkpoints = checkpoints[0:num_to_remove]

                            logger.info(
                                f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                            )
                            logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                            for removing_checkpoint in removing_checkpoints:
                                removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                shutil.rmtree(removing_checkpoint)

                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    accelerator.save_state(save_path)
                    logger.info(f"Saved state to {save_path}")

        logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
        if args.use_ema:
            logs["ema_decay"] = ema_model.cur_decay_value
        progress_bar.set_postfix(**logs)
        accelerator.log(logs, step=global_step)
        # progress_bar.close()

        accelerator.wait_for_everyone()

        # Generate sample images for visual inspection
        if accelerator.is_main_process:
            if (global_step % args.validation_steps == 0 or global_step == args.num_steps - 1) and val_loader is not None :
                unet = accelerator.unwrap_model(model)

                if args.use_ema:
                    ema_model.store(unet.parameters())
                    ema_model.copy_to(unet.parameters())

                val_loss = 0
                val_n_counter = 0
                for batch in val_loader:
                    # to device shouldn't be required! why is this happening?
                    # TODO Check if to(device) is needed? How to make it work with accelerate...
                    normal_images = batch["data"].to(weight_dtype).to(accelerator.device)
                    anom_images = batch["data_c"].to(weight_dtype).to(accelerator.device)

                    for alpha_n, alpha_k in enumerate(alphas_keys):
                        if alpha_n == 0:
                            alpha_dag = batch[alpha_k].to(weight_dtype).to(accelerator.device)
                        else:
                            alpha_dag = torch.stack([alpha_dag, batch[alpha_k].to(weight_dtype).to(accelerator.device)])
                            alpha_dag = torch.amax(alpha_dag, dim=0)

                    # Get the timestep matching for a given alpha_dag
                    timesteps = noise_scheduler.map_alpha_to_timestep(alpha_dag)

                    with torch.no_grad():
                        # Predict the noise residual
                        model_output = model(anom_images, timesteps).sample
                        loss = F.mse_loss(model_output.float(), normal_images.float().to(model_output.device), reduction="sum")
                        val_loss += loss.item() * anom_images.shape[0]
                        val_n_counter += anom_images.shape[0]

                accelerator.log({"val_loss":val_loss/val_n_counter}, step=global_step)

                pipeline = DISYREPipeline(unet=unet, scheduler=noise_scheduler,)

                batch = next(val_loader)
                # Get just 8 images for validation
                rnd_idx = np.random.choice(len(batch['data_c']), 8, replace=False)

                # run pipeline in inference (sample random noise and denoise)
                images = pipeline(
                    image=batch['data_c'][rnd_idx],
                    method="multi-step-restoration",
                    num_inference_steps=args.ddpm_num_inference_steps,
                    output_type="np",
                ).images

                if args.use_ema:
                    ema_model.restore(unet.parameters())

                # denormalize the images and save to tensorboard
                images = np.concatenate([batch['data_c'][rnd_idx], images], axis=0)
                images_processed = (images * 255).round().astype("uint8")

                if args.logger == "tensorboard":
                    if is_accelerate_version(">=", "0.17.0.dev0"):
                        tracker = accelerator.get_tracker("tensorboard", unwrap=True)
                    else:
                        tracker = accelerator.get_tracker("tensorboard")
                    tracker.add_images("test_samples", images_processed.transpose(0, 3, 1, 2), global_step)
                elif args.logger == "wandb":
                    # Upcoming `log_images` helper coming in https://github.com/huggingface/accelerate/pull/962/files
                    accelerator.get_tracker("wandb").log(
                        {"test_samples": [wandb.Image(img) for img in images_processed], "step": global_step},
                        step=global_step,
                    )

            if (global_step % args.validation_steps == 0 or global_step == args.num_steps - 1) and eval_loaders is not None :

                unet = accelerator.unwrap_model(model)

                if args.use_ema:
                    ema_model.store(unet.parameters())
                    ema_model.copy_to(unet.parameters())

                pipeline = DISYREPipeline( unet=unet, scheduler=noise_scheduler, )

                for eval_loader in eval_loaders:
                    if args.image_evaluation_only:
                        logs = evaluate.evaluate_anomaly_loader_image_only(eval_loader, pipeline, pipeline_kwargs=pipeline_eval_kwargs )
                    else:
                        logs = evaluate.evaluate_anomaly_loader(eval_loader, pipeline, pipeline_kwargs=pipeline_eval_kwargs )
                    accelerator.log(logs, step=global_step)

                if args.use_ema:
                    ema_model.restore(unet.parameters())


            if global_step % args.save_model_steps == 0 or global_step == args.num_steps - 1:
                # save the model
                unet = accelerator.unwrap_model(model)

                if args.use_ema:
                    ema_model.store(unet.parameters())
                    ema_model.copy_to(unet.parameters())

                pipeline = DISYREPipeline( unet=unet, scheduler=noise_scheduler,)
                pipeline.save_pretrained(args.output_dir)

                if args.use_ema:
                    ema_model.restore(unet.parameters())

                if args.push_to_hub:
                    upload_folder(
                        repo_id=repo_id,
                        folder_path=args.output_dir,
                        commit_message=f"Step {global_step}",
                        ignore_patterns=["step_*", "epoch_*"],
                    )

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)