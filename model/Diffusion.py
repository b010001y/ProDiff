import math
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

def get_timestep_embedding(timesteps, embedding_dim):
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = np.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.cuda()
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb

class Attention(nn.Module):
    def __init__(self, embedding_dim):
        super(Attention, self).__init__()
        self.fc = nn.Linear(embedding_dim, 1)

    def forward(self, x):
        # x shape: (batch_size, num_attributes, embedding_dim)
        weights = self.fc(x)  # shape: (batch_size, num_attributes, 1)
        # apply softmax along the attributes dimension
        weights = F.softmax(weights, dim=1)
        return weights

# class WideAndDeep(nn.Module):
#     def __init__(self, in_channels, embedding_dim=128):
#         super(WideAndDeep, self).__init__()
        
#         # Combine attribute dimensions and prototype dimensions
#         combined_dim = 2 * in_channels + 512
        
#         self.fc1 = nn.Linear(combined_dim, embedding_dim)
#         self.fc2 = nn.Linear(embedding_dim, embedding_dim)
#         self.relu = nn.ReLU()
        
#     def forward(self, attr, prototype):
#         start_point = attr[:, 0, :].float()
#         end_point = attr[:, -1, :].float()
        
#         # Combine the features of start point, end point, and prototype
#         combined_features = torch.cat((start_point, end_point, prototype), dim=1)
        
#         # Apply fully connected layers
#         x = self.fc1(combined_features)
#         x = self.relu(x)
#         combined_embed = self.fc2(x)
        
#         return combined_embed
class WideAndDeep(nn.Module):
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
        start_point = attr[:, :, 0].float()
        end_point = attr[:, :, -1].float()
        
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
        combined_embed = attr_embed + proto_embed
        
        return combined_embed


def nonlinearity(x):
    # swish
    return x * torch.sigmoid(x)

def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32,
                              num_channels=in_channels,
                              eps=1e-6,
                              affine=True)
    
class Upsample(nn.Module):
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
                                            mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv=True):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv1d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=2,
                                        padding=0)

    def forward(self, x):
        if self.with_conv:
            pad = (1, 1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class ResnetBlock(nn.Module):
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
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1)
            else:
                self.nin_shortcut = torch.nn.Conv1d(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
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
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)
        b, c, w = q.shape
        q = q.permute(0, 2, 1)  # b,hw,c
        w_ = torch.bmm(q, k)  # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)
        # attend to values
        w_ = w_.permute(0, 2, 1)  # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v, w_)
        h_ = h_.reshape(b, c, w)

        h_ = self.proj_out(h_)

        return x + h_
    
    
class Model(nn.Module):
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
        self.conv_in = torch.nn.Conv1d(in_channels, #in_channels只与embedding_dim有关，跟traj_length无关，并且输入格式为(batch_size, embedding_dim, traj_length)
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

        # middle
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
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv1d(block_in,
                                        out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)
        
    def forward(self, x, t, extra_embed=None):
        assert x.shape[2] == self.resolution

        # timestep embedding
        temb = get_timestep_embedding(t, self.ch)
        temb = self.temb.dense[0](temb)
        temb = nonlinearity(temb)
        temb = self.temb.dense[1](temb)
        if extra_embed is not None:
            temb = temb + extra_embed

        # downsampling
        hs = [self.conv_in(x)]
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
        h = hs[-1]  # [10, 256, 4, 4]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)
        # print(h.shape)
        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                ht = hs.pop()
                if ht.size(-1) != h.size(-1):
                    h = torch.nn.functional.pad(h,
                                                (0, ht.size(-1) - h.size(-1)))
                h = self.up[i_level].block[i_block](torch.cat([h, ht], dim=1),
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
        guide_emb = self.guide_emb(attr, prototype)
        place_vector = torch.zeros(attr.shape).cuda()
        place_prototype = torch.zeros(prototype.shape).cuda()
        place_emb = self.place_emb(place_vector, place_prototype)
        cond_noise = self.unet(x, t, guide_emb)
        uncond_noise = self.unet(x, t, place_emb)
        pred_noise = cond_noise + self.guidance_scale * (cond_noise -
                                                         uncond_noise)
        return pred_noise
    
    
class WeightedLoss(nn.Module):
    def __init__(self):
        super(WeightedLoss, self).__init__()
        
    def forward(self, pred, target, weighted=1.0):
        """ 
        pred, target:[batch_size, 2, traj_length]
        """
        loss = self._loss(pred, target)
        weightedLoss = (loss * weighted).mean()
        # loss = self._loss(weighted * pred, weighted * target)
        # weightedLoss = loss.mean()
        return weightedLoss
    
class WeightedL1(WeightedLoss):
    def _loss(self, pred, target):
        return torch.abs(pred - target)


class WeightedL2(WeightedLoss):
    def _loss(self, pred, target):
        return F.mse_loss(pred, target, reduction='none')

class WeightedL3(WeightedLoss):
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
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


class Diffusion(nn.Module):
    def __init__(self, loss_type, config, clip_denoised=True, predict_epsilon=True, **kwargs):
        super(Diffusion, self).__init__()
        self.predict_epsilon = predict_epsilon
        self.T = config.diffusion.num_diffusion_timesteps
        # self.device = torch.device(kwargs["device"])
        self.model = Guide_UNet(config)
        self.beta_schedule = config.diffusion.beta_schedule
        self.beta_start = config.diffusion.beta_start
        self.beta_end = config.diffusion.beta_end
        
        if self.beta_schedule == "linear":
            betas = torch.linspace(self.beta_start, self.beta_end, self.T, dtype=torch.float32)
            
        alphas = 1.0 - betas
        alpha_cumprod = torch.cumprod(alphas, axis=0) #[1, 2, 3] -> [1, 2, 6] alpha_t_bar
        alpha_cumprod_prev = torch.cat([torch.ones(1), alpha_cumprod[:-1]]) #alpha_t-1_bar，统一时间长度。对应的位置是不同的值，所以t-1最开始填补一个1

        """ 
        构建参数buffer，用于前向、反向计算的参数
        """
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_cumprod", alpha_cumprod)
        self.register_buffer("alpha_cumprod_prev", alpha_cumprod_prev)
        
        #前向过程，知道x0计算xt
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alpha_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alpha_cumprod))
        
        #反向过程
        posterior_variance = betas * (1.0 - alpha_cumprod_prev) / (1.0 - alpha_cumprod)
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer("posterior_log_variance_clipped", torch.log(posterior_variance.clamp(min=1e-20)))
        
        #计算x0需要的，在已知XT的情况下，一步反向求得X0
        self.register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alpha_cumprod))
        self.register_buffer("sqrt_recipminus_alphas_cumprod", torch.sqrt(1.0 / alpha_cumprod - 1))
        
        self.register_buffer("posterior_mean_coef1", betas * torch.sqrt(alpha_cumprod_prev) / (1.0 - alpha_cumprod))
        self.register_buffer("posterior_mean_coef2", (1.0 - alpha_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alpha_cumprod))
        
        self.loss_fn = Losses[loss_type]()
        
        """
        采样和训练主要流程
        """
        
    def q_posterior(self, x_start, x, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x.shape) * x_start
            + extract(self.posterior_mean_coef2, t, x.shape) * x
        )
        posterior_variance = extract(self.posterior_variance, t, x.shape)
        posterior_log_variance = extract(self.posterior_log_variance_clipped, t, x.shape)
        return posterior_mean, posterior_variance, posterior_log_variance

    def predict_start_from_noise(self, x, t, pred_noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x.shape) * x
            - extract(self.sqrt_recipminus_alphas_cumprod, t, x.shape) * pred_noise
        )

    def p_mean_variance(self, x, t, attr, prototype):
        pred_noise = self.model(x, t, attr, prototype)
        x_recon = self.predict_start_from_noise(x, t, pred_noise)
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_recon, x, t)
        return model_mean, posterior_log_variance

    def p_sample(self, x, t, attr, prototype, start_end_info):
        b = x.shape[0]
        model_mean, model_log_variance = self.p_mean_variance(x, t, attr, prototype)
        noise = torch.randn_like(x)

        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        x = model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

        # Fix the first and last point
        x[:, :, 0] = start_end_info[:, :, 0]
        x[:, :, -1] = start_end_info[:, :, -1]
        return x

    def p_sample_loop(self, test_x0, attr, prototype, *args, **kwargs):
        batch_size = attr.shape[0]
        # device = attr.device

        x = torch.randn(attr.shape, requires_grad=False).cuda()
        start_end_info = test_x0.clone()
        # Fix the first and last point only once at the start
        x[:, :, 0] = start_end_info[:, :, 0]
        x[:, :, -1] = start_end_info[:, :, -1]
        for i in reversed(range(0, self.T)):
            t = torch.full((batch_size,), i, dtype=torch.long).cuda()
            x = self.p_sample(x, t, attr, prototype, start_end_info)
        return x

    def sample(self, test_x0, attr, prototype, *args, **kwargs):
        x0 = self.p_sample_loop(test_x0, attr, prototype, *args, **kwargs)
        return x0

    #----------------------------------training----------------------------------#
    def q_sample(self, x_start, t, noise):
        sample = (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )
        sample[:, :, 0] = x_start[:, :, 0]
        sample[:, :, -1] = x_start[:, :, -1]
        return sample

    def p_losses(self, x_start, attr, prototype, t, weights=1.0):
        noise = torch.randn_like(x_start)
        noise[:, :, 0] = 0
        noise[:, :, -1] = 0
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        x_recon = self.model(x_noisy, t, attr, prototype)
        assert noise.shape == x_recon.shape

        if self.predict_epsilon:
            loss = self.loss_fn(x_recon[:, :, 1:-1], noise[:, :, 1:-1], weights)
        else:
            loss = self.loss_fn(x_recon[:, :, 1:-1], x_start[:, :, 1:-1], weights)

        return loss

    def trainer(self, x, attr, prototype, weights=1.0):
        batch_size = len(x)
        t = torch.randint(0, self.T, (batch_size,)).long().cuda()
        return self.p_losses(x, attr, prototype, t, weights)

    def forward(self, test_x0, attr, prototype, *args, **kwargs):
        return self.sample(test_x0, attr, prototype, *args, **kwargs)
