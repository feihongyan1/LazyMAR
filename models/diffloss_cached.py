import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
import math

from diffusion_cached import create_diffusion


class CachedDiffLoss(nn.Module):
    """
    Diffusion Loss with caching optimization for LazyMAR acceleration.
    
    This variant caches intermediate computations to avoid redundant calculations
    during autoregressive token generation. The cache is updated based on the
    cache_threshold parameter.
    """

    def __init__(self, target_channels, z_channels, depth, width, num_sampling_steps, grad_checkpointing=False):
        super(CachedDiffLoss, self).__init__()
        self.in_channels = target_channels
        self.net = SimpleMLPAdaLN(
            in_channels=target_channels,
            model_channels=width,
            out_channels=target_channels * 2,  # for vlb loss
            z_channels=z_channels,
            num_res_blocks=depth,
            grad_checkpointing=grad_checkpointing
        )

        self.train_diffusion = create_diffusion(timestep_respacing="", noise_schedule="cosine")
        self.gen_diffusion = create_diffusion(timestep_respacing=num_sampling_steps, noise_schedule="cosine")
        self.block_cos = []
        self.block_mse = []

    def forward_(self, target, z, mask=None):
        t = torch.randint(0, self.train_diffusion.num_timesteps, (target.shape[0],), device=target.device)
        model_kwargs = dict(c=z)
        loss_dict = self.train_diffusion.training_losses(self.net, target, t, model_kwargs)
        loss = loss_dict["loss"]
        if mask is not None:
            loss = (loss * mask).sum() / mask.sum()
        return loss.mean()

    def sample(self, z, temperature=1.0, cfg=1.0, diffloss_d=0, cache_type='default', step=0, cache_threshold=0, device=None):
        if not cfg == 1.0:
            noise = torch.randn(z.shape[0] // 2, self.in_channels).to(device)
            noise = torch.cat([noise, noise], dim=0)
            model_kwargs = dict(c=z, cfg_scale=cfg)
            sample_fn = self.net.forward_with_cfg
        else:
            noise = torch.randn(z.shape[0], self.in_channels).to(device)
            model_kwargs = dict(c=z)
            sample_fn = self.net.forward
        self.block_cos = []
        self.block_mse = []
        # cache修改
        sampled_token_latent = self.gen_diffusion.p_sample_loop(
            sample_fn, noise.shape, noise=noise, diffloss_d=diffloss_d, cache_type=cache_type, clip_denoised=False,
            model_kwargs=model_kwargs, device=device, progress=False,
            temperature=temperature, py_model=self.net, block_cos=self.block_cos, block_mse=self.block_mse,
            cache_threshold=cache_threshold
        )
        return sampled_token_latent

    def forward(self, z, temperature=1.0, cfg=1.0, diffloss_d=0, cache_type='default', step=0, cache_threshold=0, device=None):
        if not cfg == 1.0:
            noise = torch.randn(z.shape[0] // 2, self.in_channels).to(device)
            noise = torch.cat([noise, noise], dim=0)
            model_kwargs = dict(c=z, cfg_scale=cfg)
            sample_fn = self.net.forward_with_cfg
        else:
            noise = torch.randn(z.shape[0], self.in_channels).to(device)
            model_kwargs = dict(c=z)
            sample_fn = self.net.forward
        self.block_cos = []
        self.block_mse = []
        sampled_token_latent = self.gen_diffusion.p_sample_loop(
            sample_fn, noise.shape, noise=noise, diffloss_d=diffloss_d, cache_type=cache_type, clip_denoised=False,
            model_kwargs=model_kwargs, device=device, progress=False,
            temperature=temperature, py_model=self.net, block_cos=self.block_cos, block_mse=self.block_mse,
            cache_threshold=cache_threshold
        )
        return sampled_token_latent


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class ResBlock(nn.Module):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    """

    def __init__(
            self,
            channels
    ):
        super().__init__()
        self.channels = channels

        self.in_ln = nn.LayerNorm(channels, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels, bias=True),
            nn.SiLU(),
            nn.Linear(channels, channels, bias=True),
        )

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(channels, 3 * channels, bias=True)
        )

    def forward(self, x, y, cache, cache_type, timestep, num_sampling_steps, block_idx, use_lazy_mar,
                cache_threshold=7):
        shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(y).chunk(3, dim=-1)
        h = modulate(self.in_ln(x), shift_mlp, scale_mlp)
        if cache_type == 'default':
            # 添加预热期的特殊处理
            if timestep > 0.9 * num_sampling_steps or (timestep == int(num_sampling_steps - 1)) or (timestep % int(cache_threshold) == 0):
                cache[-1][block_idx]['mlp'] = self.mlp(h)
                cache_mlp = cache[-1][block_idx]['mlp']
                cache_mlp = cache[-1][block_idx]['mlp'][:x.size(0)] if x.size(0) < cache_mlp.size(0) else torch.cat(
                    [cache_mlp, cache_mlp]) if x.size(0) > cache_mlp.size(0) else cache_mlp
                out = x + gate_mlp * cache_mlp
            else:
                cache_mlp = cache[-1][block_idx]['mlp']
                cache_mlp = cache[-1][block_idx]['mlp'][:x.size(0)] if x.size(0) < cache_mlp.size(0) else torch.cat(
                    [cache_mlp, cache_mlp]) if x.size(0) > cache_mlp.size(0) else cache_mlp
                out = x + gate_mlp * cache_mlp
        else:
            h = self.mlp(h)
            out = x + gate_mlp * h
        return out


class FinalLayer(nn.Module):
    """
    The final layer adopted from DiT.
    """

    def __init__(self, model_channels, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(model_channels, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(model_channels, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(model_channels, 2 * model_channels, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class SimpleMLPAdaLN(nn.Module):
    """
    The MLP for Diffusion Loss.
    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param z_channels: channels in the condition.
    :param num_res_blocks: number of residual blocks per downsample.
    """

    def __init__(
            self,
            in_channels,
            model_channels,
            out_channels,
            z_channels,
            num_res_blocks,
            grad_checkpointing=False
    ):
        super().__init__()

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.grad_checkpointing = grad_checkpointing

        self.time_embed = TimestepEmbedder(model_channels)
        self.cond_embed = nn.Linear(z_channels, model_channels)

        self.input_proj = nn.Linear(in_channels, model_channels)

        res_blocks = []
        for i in range(num_res_blocks):
            res_blocks.append(ResBlock(
                model_channels,
            ))

        self.res_blocks = nn.ModuleList(res_blocks)
        self.final_layer = FinalLayer(model_channels, out_channels)

        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize timestep embedding MLP
        nn.init.normal_(self.time_embed.mlp[0].weight, std=0.02)
        nn.init.normal_(self.time_embed.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers
        for block in self.res_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, x, t, c, cache, cache_type, time_step, num_sampling_steps, use_lazy_mar, cache_threshold=7):
        """
        Apply the model to an input batch.
        :param x: an [N x C] Tensor of inputs.
        :param t: a 1-D batch of timesteps.
        :param c: conditioning from AR transformer.
        :return: an [N x C] Tensor of outputs.
        """
        x = self.input_proj(x)
        t = self.time_embed(t)
        c = self.cond_embed(c)
        y = t + c

        if self.grad_checkpointing and not torch.jit.is_scripting():
            for block in self.res_blocks:
                x = checkpoint(block, x, y)
        else:
            for block_idx, block in enumerate(self.res_blocks):
                if cache_type == 'default':
                    if (time_step == int(num_sampling_steps - 1)) or (time_step % int(cache_threshold) == 0):
                        x = block(x, y, cache, cache_type, time_step, num_sampling_steps, block_idx, use_lazy_mar,
                                  cache_threshold=cache_threshold)
                        if block_idx == 1 or block_idx == 2 or block_idx == 3 or block_idx == 4:
                            cache[-1][block_idx]['block_delta_x'] = x - last_x
                        last_x = x
                    else:
                        if block_idx == 1 or block_idx == 2 or block_idx == 3 or block_idx == 4:
                            block_delta_x = cache[-1][block_idx]['block_delta_x']
                            block_delta_x = cache[-1][block_idx]['block_delta_x'][:last_x.size(0)] if last_x.size(
                                0) < block_delta_x.size(0) else torch.cat(
                                [block_delta_x, block_delta_x]) if last_x.size(0) > block_delta_x.size(
                                0) else block_delta_x
                            x = last_x + block_delta_x
                        else:
                            x = block(x, y, cache, cache_type, time_step, num_sampling_steps, block_idx, use_lazy_mar,
                                      cache_threshold=cache_threshold)
                        last_x = x
                else:
                    x = block(x, y, cache, cache_type, time_step, num_sampling_steps, block_idx, use_lazy_mar,
                              cache_threshold=cache_threshold)
        return self.final_layer(x, y)

    def forward_with_cfg(self, x, t, cache, cache_type, time_step, num_sampling_steps, cache_threshold, c, cfg_scale,
                         lazy_mar_threshold=7):
        half = x[: len(x) // 2]
        if cache_type == 'default':
            if (time_step == int(num_sampling_steps - 1)) or (time_step % int(lazy_mar_threshold) == 0):
                use_lazy_mar = False
                combined = torch.cat([half, half], dim=0)
                model_out = self.forward(combined, t, c, cache, cache_type, time_step=time_step,
                                         num_sampling_steps=num_sampling_steps, use_lazy_mar=use_lazy_mar,
                                         cache_threshold=cache_threshold)
                eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
                cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
                delta_cfg = cond_eps - uncond_eps
                cache[-1]['cfg'] = delta_cfg
                half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
                eps = torch.cat([half_eps, half_eps], dim=0)
                return torch.cat([eps, rest], dim=1)
            else:
                use_lazy_mar = True
                c = c[: len(c) // 2]
                t = t[: len(t) // 2]
                model_out = self.forward(half, t, c, cache, cache_type, time_step=time_step,
                                         num_sampling_steps=num_sampling_steps, use_lazy_mar=use_lazy_mar,
                                         cache_threshold=cache_threshold)
                cond_eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
                uncond_eps = cond_eps - cache[-1]['cfg']
                half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
                eps = torch.cat([half_eps, half_eps], dim=0)
                rest = torch.cat([rest, rest], dim=0)
                return torch.cat([eps, rest], dim=1)
        else:
            use_lazy_mar = False
            combined = torch.cat([half, half], dim=0)
            model_out = self.forward(combined, t, c, cache, cache_type, time_step=time_step,
                                     num_sampling_steps=num_sampling_steps, use_lazy_mar=use_lazy_mar,
                                     cache_threshold=cache_threshold)
            eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
            cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
            half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
            eps = torch.cat([half_eps, half_eps], dim=0)
            return torch.cat([eps, rest], dim=1)
