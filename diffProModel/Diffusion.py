import math
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

def get_timestep_embedding(timesteps, embedding_dim):
    """Build sinusoidal timestep embeddings.

    Args:
        timesteps (torch.Tensor): A 1-D Tensor of N timesteps.
        embedding_dim (int): The dimension of the embedding.

    Returns:
        torch.Tensor: N x embedding_dim Tensor of positional embeddings.
    """
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = np.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.cuda() # Move embedding to CUDA device
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # Zero pad if embedding_dim is odd
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb

class Attention(nn.Module):
    """A simple attention layer to get weights for attributes."""
    def __init__(self, embedding_dim):
        super(Attention, self).__init__()
        self.fc = nn.Linear(embedding_dim, 1)

    def forward(self, x):
        # x shape: (batch_size, num_attributes, embedding_dim)
        weights = self.fc(x)  # shape: (batch_size, num_attributes, 1)
        # Apply softmax along the attributes dimension to get attention weights.
        weights = F.softmax(weights, dim=1)
        return weights
    
class WideAndDeep(nn.Module):
    """Network to combine attribute (start/end points) and prototype embeddings."""
    def __init__(self, in_channels, embedding_dim=512):
        super(WideAndDeep, self).__init__()
        
        # Process start point and end point independently
        self.start_fc1 = nn.Linear(in_channels, embedding_dim)
        self.start_fc2 = nn.Linear(embedding_dim, embedding_dim)
        
        self.end_fc1 = nn.Linear(in_channels, embedding_dim)
        self.end_fc2 = nn.Linear(embedding_dim, embedding_dim)
        
        # Process prototype features
        self.prototype_fc1 = nn.Linear(512, embedding_dim)
        self.prototype_fc2 = nn.Linear(embedding_dim, embedding_dim)
        
        self.relu = nn.ReLU()
        
    def forward(self, attr, prototype):
        # attr shape: (batch_size, num_features, traj_length)
        # prototype shape: (batch_size, prototype_embedding_dim) - assuming N_CLUSTER is handled before or it's single prototype
        start_point = attr[:, :, 0].float() # First point in trajectory features
        end_point = attr[:, :, -1].float()  # Last point in trajectory features
        
        # Process start point features
        start_x = self.start_fc1(start_point)
        start_x = self.relu(start_x)
        start_embed = self.start_fc2(start_x)
        
        # Process end point features
        end_x = self.end_fc1(end_point)
        end_x = self.relu(end_x)
        end_embed = self.end_fc2(end_x)
        
        # Combine the processed start and end point features
        attr_embed = start_embed + end_embed
        
        # Process prototype features
        proto_x = self.prototype_fc1(prototype)
        proto_x = self.relu(proto_x)
        proto_embed = self.prototype_fc2(proto_x)
        
        # Combine the processed attribute and prototype features
        combined_embed = attr_embed + proto_embed # Simple addition for combination
        
        return combined_embed


def nonlinearity(x):
    # Swish activation function (SiLU)
    return x * torch.sigmoid(x)

def Normalize(in_channels):
    """Group normalization."""
    return torch.nn.GroupNorm(num_groups=32,
                              num_channels=in_channels,
                              eps=1e-6,
                              affine=True)
    
class Upsample(nn.Module):
    """Upsampling layer, optionally with a 1D convolution."""
    def __init__(self, in_channels, with_conv=True):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv1d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        x = torch.nn.functional.interpolate(x,
                                            scale_factor=2.0,
                                            mode="nearest") # Upsample using nearest neighbor
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    """Downsampling layer, optionally with a 1D convolution."""
    def __init__(self, in_channels, with_conv=True):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # No asymmetric padding in torch.nn.Conv1d, must do it ourselves via F.pad.
            self.conv = torch.nn.Conv1d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=2,
                                        padding=0)

    def forward(self, x):
        if self.with_conv:
            pad = (1, 1) # Padding for kernel_size=3, stride=2 to maintain roughly half size
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2) # Avg pool if no conv
        return x


class ResnetBlock(nn.Module):
    """Residual block for the U-Net."""
    def __init__(self,
                 in_channels,
                 out_channels=None,
                 conv_shortcut=False,
                 dropout=0.1,
                 temb_channels=512):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv1d(in_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        self.temb_proj = torch.nn.Linear(temb_channels, out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv1d(out_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv1d(in_channels,
                                                     out_channels,
                                                     kernel_size=3, # Convolutional shortcut
                                                     stride=1,
                                                     padding=1)
            else:
                self.nin_shortcut = torch.nn.Conv1d(in_channels,
                                                    out_channels,
                                                    kernel_size=1, # 1x1 convolution (Network-in-Network) shortcut
                                                    stride=1,
                                                    padding=0)

    def forward(self, x, temb):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)
        h = h + self.temb_proj(nonlinearity(temb))[:, :, None]
        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x + h


class AttnBlock(nn.Module):
    """Self-attention block for the U-Net."""
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv1d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv1d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv1d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv1d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_) # Query
        k = self.k(h_) # Key
        v = self.v(h_) # Value
        b, c, w = q.shape
        q = q.permute(0, 2, 1)  # b,w,c (sequence_length, channels)
        w_ = torch.bmm(q, k)  # b,w,w (attention scores: q @ k.T)
        w_ = w_ * (int(c)**(-0.5)) # Scale by sqrt(channel_dim)
        w_ = torch.nn.functional.softmax(w_, dim=2) # Softmax over scores
        # attend to values
        w_ = w_.permute(0, 2, 1)  # b,w,w (transpose back for v @ w_ if v is b,c,w)
        h_ = torch.bmm(v, w_) # Weighted sum of values
        h_ = h_.reshape(b, c, w)

        h_ = self.proj_out(h_)

        return x + h_ # Add residual connection
    
    
class Model(nn.Module):
    """The core U-Net model for the diffusion process."""
    def __init__(self, config):
        super(Model, self).__init__()
        self.config = config
        ch, out_ch, ch_mult = config.model.ch, config.model.out_ch, tuple(config.model.ch_mult)
        num_res_blocks = config.model.num_res_blocks
        attn_resolutions = config.model.attn_resolutions
        dropout = config.model.dropout
        in_channels = config.model.in_channels
        resolution = config.data.traj_length
        resamp_with_conv = config.model.resamp_with_conv
        num_timesteps = config.diffusion.num_diffusion_timesteps
        
        if config.model.type == 'bayesian':
            self.logvar = nn.Parameter(torch.zeros(num_timesteps))
            
        self.ch = ch
        self.temb_ch = self.ch * 4
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        
        # timestep embedding
        self.temb = nn.Module()
        self.temb.dense = nn.ModuleList([
            torch.nn.Linear(self.ch, self.temb_ch),
            torch.nn.Linear(self.temb_ch, self.temb_ch),
        ])

        # downsampling
        self.conv_in = torch.nn.Conv1d(in_channels, # in_channels related to embedding_dim, not traj_length. Input format (batch_size, embedding_dim, traj_length)
                                       self.ch,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        curr_res = resolution
        in_ch_mult = (1, ) + ch_mult
        self.down = nn.ModuleList()
        block_in = None
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(
                    ResnetBlock(in_channels=block_in,
                                out_channels=block_out,
                                temb_channels=self.temb_ch,
                                dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle block
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            skip_in = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                if i_block == self.num_res_blocks:
                    skip_in = ch * in_ch_mult[i_level]
                block.append(
                    ResnetBlock(in_channels=block_in + skip_in,
                                out_channels=block_out,
                                temb_channels=self.temb_ch,
                                dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up)  # Prepend to get consistent order for upsampling path

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv1d(block_in,
                                        out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)
        
    def forward(self, x, t, extra_embed=None):
        assert x.shape[2] == self.resolution # Ensure input trajectory length matches model resolution

        # timestep embedding
        temb = get_timestep_embedding(t, self.ch)
        temb = self.temb.dense[0](temb)
        temb = nonlinearity(temb)
        temb = self.temb.dense[1](temb)
        if extra_embed is not None:
            temb = temb + extra_embed

        # downsampling
        hs = [self.conv_in(x)] # List to store hidden states for skip connections
        # print(hs[-1].shape)
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                # print(i_level, i_block, h.shape)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        # print(hs[-1].shape)
        # print(len(hs))
        h = hs[-1]  # Last hidden state from downsampling path
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)
        # print(h.shape)
        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                ht = hs.pop() # Get corresponding hidden state from downsampling path
                if ht.size(-1) != h.size(-1):
                    # Pad if spatial dimensions do not match (can happen with odd resolutions)
                    h = torch.nn.functional.pad(h,
                                                (0, ht.size(-1) - h.size(-1)))
                h = self.up[i_level].block[i_block](torch.cat([h, ht], dim=1), # Concatenate skip connection
                                                    temb)
                # print(i_level, i_block, h.shape)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h
    
class Guide_UNet(nn.Module):
    """A U-Net model guided by attribute and prototype embeddings."""
    def __init__(self, config):
        super(Guide_UNet, self).__init__()
        self.config = config
        self.in_channels = config.model.in_channels
        self.ch = config.model.ch * 4
        self.attr_dim = config.model.attr_dim
        self.guidance_scale = config.model.guidance_scale
        self.unet = Model(config)
        self.guide_emb = WideAndDeep(self.in_channels, self.ch)
        self.place_emb = WideAndDeep(self.in_channels, self.ch)
        
    def forward(self, x, t, attr, prototype):
        guide_emb = self.guide_emb(attr, prototype) # Conditional embedding
        
        target_device = attr.device # Get device from an existing input tensor
        place_vector = torch.zeros(attr.shape, device=target_device)
        place_prototype = torch.zeros(prototype.shape, device=target_device)
        
        place_emb = self.place_emb(place_vector, place_prototype) # Unconditional embedding
        
        cond_noise = self.unet(x, t, guide_emb) # Conditioned UNet pass
        uncond_noise = self.unet(x, t, place_emb) # Unconditioned UNet pass (for classifier-free guidance)
        
        # Classifier-free guidance
        pred_noise = cond_noise + self.guidance_scale * (cond_noise -
                                                         uncond_noise)
        return pred_noise
    
    
class WeightedLoss(nn.Module):
    """Base class for weighted losses."""
    def __init__(self):
        super(WeightedLoss, self).__init__()
        
    def forward(self, pred, target, weighted=1.0):
        """ 
        pred, target:[batch_size, 2, traj_length]
        """
        loss = self._loss(pred, target)
        weightedLoss = (loss * weighted).mean() # Apply weights and average
        # loss = self._loss(weighted * pred, weighted * target)
        # weightedLoss = loss.mean()
        return weightedLoss
    
class WeightedL1(WeightedLoss):
    """Weighted L1 Loss (Mean Absolute Error)."""
    def _loss(self, pred, target):
        return torch.abs(pred - target)


class WeightedL2(WeightedLoss):
    """Weighted L2 Loss (Mean Squared Error)."""
    def _loss(self, pred, target):
        return F.mse_loss(pred, target, reduction='none')

class WeightedL3(WeightedLoss):
    """A custom weighted L3-like loss, where weights depend on the error magnitude."""
    def __init__(self, base_weight=1000.0, scale_factor=10000.0):
        super(WeightedL3, self).__init__()
        self.base_weight = base_weight
        self.scale_factor = scale_factor

    def _loss(self, pred, target):
        error = F.mse_loss(pred, target, reduction='none')
        weight = self.base_weight + self.scale_factor * error
        loss = weight * torch.abs(pred - target)
        return loss
Losses = {
    'l1': WeightedL1,
    'l2': WeightedL2,
    'l3': WeightedL3,
}

def extract(a, t, x_shape):
    """Extracts values from a (typically constants like alphas) at given timesteps t 
       and reshapes them to match the batch shape x_shape.
    """
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1))) # Reshape to (b, 1, 1, ...) for broadcasting


class Diffusion(nn.Module):
    """Denoising Diffusion Probabilistic Model (DDPM).
    This class now also includes DDIM sampling capabilities.
    """
    def __init__(self, loss_type, config, clip_denoised=True, predict_epsilon=True, **kwargs):
        super(Diffusion, self).__init__()
        self.predict_epsilon = predict_epsilon
        self.T = config.diffusion.num_diffusion_timesteps
        self.model = Guide_UNet(config)
        self.beta_schedule = config.diffusion.beta_schedule
        self.beta_start = config.diffusion.beta_start
        self.beta_end = config.diffusion.beta_end
        
        if self.beta_schedule == "linear":
            betas = torch.linspace(self.beta_start, self.beta_end, self.T, dtype=torch.float32)
        elif self.beta_schedule == "cosine":
            # Implement cosine schedule
            pass
        else:
            raise ValueError(f"Unsupported beta_schedule: {self.beta_schedule}")
            
        alphas = 1.0 - betas
        alpha_cumprod = torch.cumprod(alphas, axis=0)
        alpha_cumprod_prev = torch.cat([torch.ones(1, device=betas.device), alpha_cumprod[:-1]])

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_cumprod", alpha_cumprod)
        self.register_buffer("alpha_cumprod_prev", alpha_cumprod_prev)
        
        # Parameters for q(x_t | x_0) (forward process - DDPM & DDIM)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alpha_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alpha_cumprod))
        
        # Parameters for DDPM reverse process posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1.0 - alpha_cumprod_prev) / (1.0 - alpha_cumprod)
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer("posterior_log_variance_clipped", torch.log(posterior_variance.clamp(min=1e-20)))
        self.register_buffer("posterior_mean_coef1", betas * torch.sqrt(alpha_cumprod_prev) / (1.0 - alpha_cumprod))
        self.register_buffer("posterior_mean_coef2", (1.0 - alpha_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alpha_cumprod))
        
        # Parameters for computing x_0 from x_t and noise (used in DDPM prediction and DDIM sampling)
        self.register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alpha_cumprod))
        self.register_buffer("sqrt_recipminus_alphas_cumprod", torch.sqrt(1.0 / alpha_cumprod - 1))
        
        self.loss_fn = Losses[loss_type]()
        
    def q_posterior(self, x_start, x, t):
        """Compute the mean, variance, and log variance of the posterior q(x_{t-1} | x_t, x_0)."""
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x.shape) * x_start
            + extract(self.posterior_mean_coef2, t, x.shape) * x
        )
        posterior_variance = extract(self.posterior_variance, t, x.shape)
        posterior_log_variance = extract(self.posterior_log_variance_clipped, t, x.shape)
        return posterior_mean, posterior_variance, posterior_log_variance

    def predict_start_from_noise(self, x, t, pred_noise):
        """Compute x_0 from x_t and predicted noise epsilon_theta(x_t, t).
           Used by both DDPM and DDIM.
        """
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x.shape) * x
            - extract(self.sqrt_recipminus_alphas_cumprod, t, x.shape) * pred_noise
        )

    def p_mean_variance(self, x, t, attr, prototype):
        """Compute the mean and variance of the reverse process p_theta(x_{t-1} | x_t)."""
        pred_noise = self.model(x, t, attr, prototype)
        x_recon = self.predict_start_from_noise(x, t, pred_noise) # Predict x0
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_recon, x, t)
        return model_mean, posterior_log_variance

    def p_sample(self, x, t, attr, prototype, start_end_info):
        """Sample x_{t-1} from the model p_theta(x_{t-1} | x_t) (DDPM step)."""
        b = x.shape[0]
        model_mean, model_log_variance = self.p_mean_variance(x, t, attr, prototype)
        noise = torch.randn_like(x)

        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1))) # No noise when t=0
        x = model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

        # Fix the first and last point for trajectory interpolation
        x[:, :, 0] = start_end_info[:, :, 0]
        x[:, :, -1] = start_end_info[:, :, -1]
        return x

    def p_sample_loop(self, test_x0, attr, prototype, *args, **kwargs):
        """DDPM sampling loop to generate x_0 from x_T (noise)."""
        batch_size = attr.shape[0]
        device = attr.device # Assuming attr is on the correct device

        x = torch.randn(attr.shape, requires_grad=False, device=device) # Start with pure noise
        start_end_info = test_x0.clone() # Contains the ground truth start and end points
        
        # Fix the first and last point from the start
        x[:, :, 0] = start_end_info[:, :, 0]
        x[:, :, -1] = start_end_info[:, :, -1]
        
        for i in reversed(range(0, self.T)): # Iterate from T-1 down to 0
            t = torch.full((batch_size,), i, dtype=torch.long, device=device)
            x = self.p_sample(x, t, attr, prototype, start_end_info)
        return x

    # --------------------- DDIM Sampling Methods ---------------------
    def ddim_sample(self, x, t, t_prev, attr, prototype, start_end_info, eta=0.0):
        """
        DDIM sampling step from t to t_prev.
        eta: Controls stochasticity. 0 for DDIM (deterministic), 1 for DDPM-like (stochastic).
        """
        # Ensure model is on the same device as x
        self.model.to(x.device)
        
        pred_noise = self.model(x, t, attr, prototype)
        x_0_pred = self.predict_start_from_noise(x, t, pred_noise)
        
        x_0_pred[:, :, 0] = start_end_info[:, :, 0]
        x_0_pred[:, :, -1] = start_end_info[:, :, -1]
        
        alpha_cumprod_t = extract(self.alpha_cumprod, t, x.shape)
        alpha_cumprod_t_prev = extract(self.alpha_cumprod, t_prev, x.shape) if t_prev.all() >= 0 else torch.ones_like(alpha_cumprod_t)
        
        sigma_t = eta * torch.sqrt((1 - alpha_cumprod_t_prev) / (1 - alpha_cumprod_t) * (1 - alpha_cumprod_t / alpha_cumprod_t_prev))
        
        c1 = torch.sqrt(alpha_cumprod_t_prev)
        c2 = torch.sqrt(1 - alpha_cumprod_t_prev - sigma_t**2)
        
        noise_cond = torch.zeros_like(x)
        if eta > 0:
            noise_cond = torch.randn_like(x)
            noise_cond[:, :, 0] = 0
            noise_cond[:, :, -1] = 0
        
        x_prev = c1 * x_0_pred + c2 * pred_noise + sigma_t * noise_cond
        
        x_prev[:, :, 0] = start_end_info[:, :, 0]
        x_prev[:, :, -1] = start_end_info[:, :, -1]
        
        return x_prev
        
    def ddim_sample_loop(self, test_x0, attr, prototype, num_steps=50, eta=0.0):
        """
        DDIM sampling loop. Can use fewer steps than original diffusion process.
        num_steps: Number of sampling steps (can be less than self.T).
        eta: Controls stochasticity (0 for deterministic, 1 for fully stochastic).
        """
        batch_size = attr.shape[0]
        device = attr.device # Assuming attr is on the correct device
        
        x = torch.randn(attr.shape, requires_grad=False, device=device)
        start_end_info = test_x0.clone()
        
        x[:, :, 0] = start_end_info[:, :, 0]
        x[:, :, -1] = start_end_info[:, :, -1]
        
        times = torch.linspace(self.T - 1, 0, num_steps + 1, device=device).long() # Ensure times tensor is on the same device
        
        for i in range(num_steps):
            t = times[i]
            t_next = times[i + 1]
            # Create full tensors for t and t_next for batch processing
            t_tensor = torch.full((batch_size,), t.item(), dtype=torch.long, device=device)
            t_next_tensor = torch.full((batch_size,), t_next.item(), dtype=torch.long, device=device)
            
            x = self.ddim_sample(x, t_tensor, t_next_tensor, attr, prototype, start_end_info, eta)
            
        return x

    # --------------------- Unified Sampling Entry Point ---------------------
    def sample(self, test_x0, attr, prototype, sampling_type='ddpm', 
               ddim_num_steps=50, ddim_eta=0.0, *args, **kwargs):
        """Generate samples using either DDPM or DDIM.

        Args:
            test_x0 (torch.Tensor): Tensor containing ground truth data, primarily used for start/end points.
            attr (torch.Tensor): Attributes for conditioning.
            prototype (torch.Tensor): Prototypes for conditioning.
            sampling_type (str, optional): 'ddpm' or 'ddim'. Defaults to 'ddpm'.
            ddim_num_steps (int, optional): Number of steps for DDIM sampling. Defaults to 50.
            ddim_eta (float, optional): Eta for DDIM sampling. Defaults to 0.0.
        """
        self.model.eval() # Set model to evaluation mode for sampling
        with torch.no_grad():
            if sampling_type == 'ddpm':
                return self.p_sample_loop(test_x0, attr, prototype, *args, **kwargs)
            elif sampling_type == 'ddim':
                return self.ddim_sample_loop(test_x0, attr, prototype, 
                                             num_steps=ddim_num_steps, eta=ddim_eta)
            else:
                raise ValueError(f"Unsupported sampling_type: {sampling_type}. Choose 'ddpm' or 'ddim'.")

    #----------------------------------training----------------------------------#
    def q_sample(self, x_start, t, noise):
        """Sample x_t from x_0 using q(x_t | x_0) = sqrt(alpha_bar_t)x_0 + sqrt(1-alpha_bar_t)noise."""
        sample = (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )
        # Keep start and end points fixed during noising process as well (for interpolation task)
        sample[:, :, 0] = x_start[:, :, 0]
        sample[:, :, -1] = x_start[:, :, -1]
        return sample

    def p_losses(self, x_start, attr, prototype, t, weights=1.0):
        """Calculate the diffusion loss (typically MSE between predicted noise and actual noise).
           This is common for both DDPM and DDIM training.
        """
        noise = torch.randn_like(x_start)
        # For interpolation, noise is not added to the fixed start/end points
        noise[:, :, 0] = 0
        noise[:, :, -1] = 0
        
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        x_recon = self.model(x_noisy, t, attr, prototype) # Model predicts noise or x0
        assert noise.shape == x_recon.shape

        if self.predict_epsilon:
            # Loss on the predicted noise, excluding start/end points
            loss = self.loss_fn(x_recon[:, :, 1:-1], noise[:, :, 1:-1], weights)
        else:
            # Loss on the predicted x0, excluding start/end points
            loss = self.loss_fn(x_recon[:, :, 1:-1], x_start[:, :, 1:-1], weights)

        return loss

    def trainer(self, x, attr, prototype, weights=1.0):
        """Performs a single training step. Common for DDPM and DDIM."""
        self.model.train() # Set model to training mode
        batch_size = len(x)
        t = torch.randint(0, self.T, (batch_size,), device=x.device).long() # Sample random timesteps on the same device as x
        return self.p_losses(x, attr, prototype, t, weights)

    def forward(self, test_x0, attr, prototype, sampling_type='ddpm', 
                ddim_num_steps=50, ddim_eta=0.0, *args, **kwargs):
        """Default forward pass calls the unified sampling method."""
        return self.sample(test_x0, attr, prototype, 
                           sampling_type=sampling_type, 
                           ddim_num_steps=ddim_num_steps, 
                           ddim_eta=ddim_eta, 
                           *args, **kwargs)
