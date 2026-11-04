"""Checkpoint-compatible EDGAR refiner. No baseline or ablation branches."""

import copy
import re
from types import MethodType
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, CLIPTextModel
from diffusers import AutoencoderKL, DDPMScheduler
from diffusers.models.attention import BasicTransformerBlock
from peft import LoraConfig

from .unet import UNet2DConditionModel, moe_forward


def my_vae_encoder_fwd(self, sample):
    sample = self.conv_in(sample)
    l_blocks = []
    # down
    for down_block in self.down_blocks:
        l_blocks.append(sample)
        sample = down_block(sample)
    # middle
    sample = self.mid_block(sample)
    sample = self.conv_norm_out(sample)
    sample = self.conv_act(sample)
    sample = self.conv_out(sample)
    self.current_down_blocks = l_blocks
    return sample


def my_vae_decoder_fwd(self, sample, latent_embeds=None):
    sample = self.conv_in(sample)
    upscale_dtype = next(iter(self.up_blocks.parameters())).dtype
    # middle
    sample = self.mid_block(sample, latent_embeds)
    sample = sample.to(upscale_dtype)

    skip_convs = [
        self.skip_conv_1,
        self.skip_conv_2,
        self.skip_conv_3,
        self.skip_conv_4,
    ]
    # up
    for idx, up_block in enumerate(self.up_blocks):
        skip_in = skip_convs[idx](self.incoming_skip_acts[::-1][idx] * self.gamma)
        # add skip
        sample = sample + skip_in
        sample = up_block(sample, latent_embeds)

    # post-process
    if latent_embeds is None:
        sample = self.conv_norm_out(sample)
    else:
        sample = self.conv_norm_out(sample, latent_embeds)
    sample = self.conv_act(sample)
    sample = self.conv_out(sample)
    return sample


class GlobalRouter(torch.nn.Module):
    def __init__(self, dim, num_experts, hidden=256):
        super().__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.LayerNorm(dim),
            torch.nn.Linear(dim, hidden),
            torch.nn.GELU(),
            torch.nn.Linear(hidden, num_experts),
        )

    def forward(self, x):
        return F.softmax(self.mlp(x.mean(dim=1)), dim=-1)


def read_checkpoint(path):
    """Load a trusted tensor checkpoint; reject incomplete legacy weights."""
    state = torch.load(path, map_location="cpu", weights_only=True)
    required = {
        "rank_vae",
        "vae_lora_target_modules",
        "state_dict_unet",
        "state_dict_vae",
        "state_dict_ref_proj",
    }
    missing = required.difference(state)
    if missing:
        raise ValueError(
            f"Incomplete EDGAR checkpoint; missing {sorted(missing)}. "
            "Use a complete checkpoint; missing trained parameters cannot be reconstructed."
        )
    experts = {
        int(m.group(1))
        for key in state["state_dict_unet"]
        if (m := re.search(r"\.experts\.(\d+)\.", key))
    }
    if experts != set(range(6)):
        raise ValueError("This release supports the full six-expert EDGAR model only.")
    return state


class EDGAR(torch.nn.Module):
    def __init__(
        self,
        *,
        sapiens_path,
        checkpoint=None,
        base_model="stabilityai/sd-turbo",
        timestep=249,
        lora_rank_vae=4,
        device="cuda",
    ):
        super().__init__()
        device = torch.device(device)
        if device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("EDGAR requires a CUDA-enabled PyTorch installation.")
        if not Path(sapiens_path).is_file():
            raise FileNotFoundError(
                f"Sapiens TorchScript model not found: {sapiens_path}"
            )
        state = read_checkpoint(checkpoint) if checkpoint else None
        if state and "model_config" in state:
            config = state["model_config"]
            timestep = config["timestep"]
            base_model = config["base_model"]
        self.model_config = dict(
            base_model=base_model, timestep=timestep, num_experts=6
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model, subfolder="tokenizer"
        )
        self.text_encoder = CLIPTextModel.from_pretrained(
            base_model, subfolder="text_encoder"
        )
        self.sched = DDPMScheduler.from_pretrained(base_model, subfolder="scheduler")
        self.sched.set_timesteps(1, device=device)
        self.sched.alphas_cumprod = self.sched.alphas_cumprod.to(device)
        vae = AutoencoderKL.from_pretrained(base_model, subfolder="vae")
        vae.encoder.forward = MethodType(my_vae_encoder_fwd, vae.encoder)
        vae.decoder.forward = MethodType(my_vae_decoder_fwd, vae.decoder)
        for i, (channels_in, channels_out) in enumerate(
            [(512, 512), (256, 512), (128, 512), (128, 256)], 1
        ):
            layer = torch.nn.Conv2d(channels_in, channels_out, 1, bias=False)
            torch.nn.init.constant_(layer.weight, 1e-5)
            setattr(vae.decoder, f"skip_conv_{i}", layer)
        unet = UNet2DConditionModel.from_pretrained(
            base_model,
            subfolder="unet",
            low_cpu_mem_usage=False,
            ignore_mismatched_sizes=True,
        )
        # Bind only EDGAR's instances, never modify diffusers' global classes.
        for block in unet.modules():
            if isinstance(block, BasicTransformerBlock):
                block.forward = MethodType(moe_forward, block)
                block.experts = None
        dims = [320, 640, 1280]
        for i in range(2):
            for attention in unet.down_blocks[i].attentions:
                self._add_experts(attention.transformer_blocks[0], dims[i])
        for i in range(2, 4):
            for attention in unet.up_blocks[i].attentions:
                self._add_experts(attention.transformer_blocks[0], dims[3 - i])
        self._add_experts(unet.mid_block.attentions[0].transformer_blocks[0], 1280)
        if state:
            targets = state["vae_lora_target_modules"]
            lora_rank_vae = state["rank_vae"]
        else:
            suffixes = [
                "conv1",
                "conv2",
                "conv_in",
                "conv_shortcut",
                "conv",
                "conv_out",
                "skip_conv_1",
                "skip_conv_2",
                "skip_conv_3",
                "skip_conv_4",
                "to_k",
                "to_q",
                "to_v",
                "to_out.0",
            ]
            targets = [
                name
                for name, _ in vae.named_modules()
                if "decoder" in name and any(name.endswith(s) for s in suffixes)
            ]
        vae.requires_grad_(False)
        vae.add_adapter(
            LoraConfig(
                r=lora_rank_vae, init_lora_weights="gaussian", target_modules=targets
            ),
            adapter_name="vae_skip",
        )
        self.lora_rank_vae, self.target_modules_vae = lora_rank_vae, targets
        self.ref_proj = torch.nn.Conv2d(1024, 320, 1, bias=False)
        if state:
            unet.load_state_dict(state["state_dict_unet"], strict=True)
            expected = {k for k in vae.state_dict() if "lora" in k or "skip" in k}
            if set(state["state_dict_vae"]) != expected:
                raise ValueError("VAE adapter checkpoint keys do not match the model.")
            vae.load_state_dict(
                {**vae.state_dict(), **state["state_dict_vae"]}, strict=True
            )
            self.ref_proj.load_state_dict(state["state_dict_ref_proj"], strict=True)
        self.unet, self.vae = unet, vae
        self.vae.decoder.gamma = 1
        self.register_buffer(
            "timesteps", torch.tensor([timestep], dtype=torch.long), persistent=False
        )
        self.sapiens = torch.jit.load(str(sapiens_path), map_location="cpu")
        for parameter in self.sapiens.parameters():
            parameter.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.to(device)
        self.eval()

    @staticmethod
    def _add_experts(block, dim):
        block.experts = torch.nn.ModuleList([copy.deepcopy(block.ff) for _ in range(6)])
        block.router = GlobalRouter(dim, 6)
        del block.ff

    def train(self, mode=True):
        super().train(mode)
        self.sapiens.eval()
        self.text_encoder.eval()
        return self

    def set_train(self):
        self.train()
        self.unet.requires_grad_(True)
        for name, module in self.unet.named_modules():
            if any(
                part in name
                for part in (
                    "time_proj",
                    "time_embedding",
                    "attn2",
                    "norm2",
                    "resnets",
                    "downsamplers",
                    "upsamplers",
                    "conv",
                )
            ):
                module.requires_grad_(False)
        self.vae.requires_grad_(False)
        for name, param in self.vae.named_parameters():
            param.requires_grad_("lora" in name or "skip_conv" in name)
        self.ref_proj.requires_grad_(True)

    def set_eval(self):
        self.eval()
        self.requires_grad_(False)

    def encode_reference(self, reference):
        with torch.no_grad():
            features = self.sapiens(reference)
            features = features[0] if isinstance(features, (tuple, list)) else features
        return self.ref_proj(features)

    def forward(
        self,
        image,
        reference=None,
        *,
        prompt="High resolution portrait",
        prompt_tokens=None,
        reference_features=None,
        generator=None,
    ):
        if prompt_tokens is None:
            prompt_tokens = self.tokenizer(
                prompt,
                max_length=self.tokenizer.model_max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            ).input_ids
        prompt_tokens = prompt_tokens.reshape(-1, prompt_tokens.shape[-1]).to(
            image.device
        )
        with torch.no_grad():
            caption = self.text_encoder(prompt_tokens)[0]
        if caption.shape[0] == 1 and image.shape[0] > 1:
            caption = caption.expand(image.shape[0], -1, -1)
        z = (
            self.vae.encode(image).latent_dist.sample(generator=generator)
            * self.vae.config.scaling_factor
        )
        if reference_features is None:
            if reference is None:
                raise ValueError("A subject reference image is required.")
            reference_features = self.encode_reference(reference)
        if reference_features.shape[0] == 1 and image.shape[0] > 1:
            reference_features = reference_features.expand(image.shape[0], -1, -1, -1)
        prediction = self.unet(
            z,
            self.timesteps,
            encoder_hidden_states=caption,
            ref_feat=reference_features,
        ).sample
        denoised = self.sched.step(prediction, self.timesteps, z).prev_sample
        self.vae.decoder.incoming_skip_acts = self.vae.encoder.current_down_blocks
        latent = (denoised / self.vae.config.scaling_factor).to(dtype=self.vae.dtype)
        return self.vae.decode(latent).sample.clamp(-1, 1)

    def checkpoint_state(self):
        return {
            "format_version": 1,
            "model_config": self.model_config,
            "rank_vae": self.lora_rank_vae,
            "vae_lora_target_modules": self.target_modules_vae,
            "state_dict_unet": self.unet.state_dict(),
            "state_dict_vae": {
                k: v
                for k, v in self.vae.state_dict().items()
                if "lora" in k or "skip" in k
            },
            "state_dict_ref_proj": self.ref_proj.state_dict(),
        }
