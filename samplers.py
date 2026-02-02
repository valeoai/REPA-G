import torch
import numpy as np

from potential import feature_dir_update


def expand_t_like_x(t, x_cur):
    """Function to reshape time t to broadcastable dimension of x
    Args:
      t: [batch_dim,], time vector
      x: [batch_dim,...], data point
    """
    dims = [1] * (len(x_cur.size()) - 1)
    t = t.view(t.size(0), *dims)
    return t


def get_score_from_velocity(vt, xt, t, path_type="linear"):
    """Wrapper function: transfrom velocity prediction model to score
    Args:
        velocity: [batch_dim, ...] shaped tensor; velocity model output
        x: [batch_dim, ...] shaped tensor; x_t data point
        t: [batch_dim,] time tensor
    """
    t = expand_t_like_x(t, xt)
    if path_type == "linear":
        alpha_t, d_alpha_t = 1 - t, torch.ones_like(xt, device=xt.device) * -1
        sigma_t, d_sigma_t = t, torch.ones_like(xt, device=xt.device)
    elif path_type == "cosine":
        alpha_t = torch.cos(t * np.pi / 2)
        sigma_t = torch.sin(t * np.pi / 2)
        d_alpha_t = -np.pi / 2 * torch.sin(t * np.pi / 2)
        d_sigma_t =  np.pi / 2 * torch.cos(t * np.pi / 2)
    elif path_type == "reverse_linear":
        alpha_t, d_alpha_t = t, torch.ones_like(xt, device=xt.device)
        sigma_t, d_sigma_t = 1 - t, torch.ones_like(xt, device=xt.device) * -1
    else:
        raise NotImplementedError

    mean = xt
    reverse_alpha_ratio = alpha_t / d_alpha_t
    var = sigma_t**2 - reverse_alpha_ratio * d_sigma_t * sigma_t
    score = (reverse_alpha_ratio * vt - mean) / var

    return score


def compute_diffusion(t_cur):
    return 2 * t_cur


def euler_sampler(
        model,
        latents,
        y,
        potential=None,
        num_steps=20,
        heun=False,
        cfg_scale=1.0,
        guidance_low=0.0,
        guidance_high=1.0,
        path_type="linear", # not used, just for compatability
        y_bis=None,
        gibbs=True,
        use_projector=True,
    ):
    # setup conditioning
    if cfg_scale > 1.0 or y_bis is not None:
        y_null = torch.tensor([1000] * y.size(0), device=y.device)
    _dtype = latents.dtype    
    t_steps = torch.linspace(1, 0, num_steps+1, dtype=torch.float64)
    if path_type == "reverse_linear":
        t_steps = torch.flip(t_steps, dims=[0])
    x_next = latents.to(torch.float64)
    device = x_next.device

    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        x_cur = x_next
        
        # --- 1. Prepare Inputs (Main Step) ---
        if (cfg_scale > 1.0 or y_bis is not None) and t_cur <= guidance_high and t_cur >= guidance_low:
            model_input = torch.cat([x_cur] * 2 if y_bis is None else [x_cur] * 3, dim=0)
            if y_bis is None:
                y_cur = torch.cat([y, y_null], dim=0)
            else:
                y_cur = torch.cat([y, y_bis, y_null], dim=0)
        else:
            model_input = x_cur
            y_cur = y            
        kwargs = dict(y=y_cur)
        time_input = torch.ones(model_input.size(0)).to(device=device, dtype=torch.float64) * t_cur
        if potential is not None and t_cur <= guidance_high and t_cur >= guidance_low: 
            # Apply transport potential only when we are past a certain threshold of steps
            d_cur, zs = model.inference_feats(model_input.to(dtype=_dtype), time_input.to(dtype=_dtype), use_projector=use_projector, **kwargs)
            d_feat = feature_dir_update(model_input, zs[0], potential).to(torch.float64)
            if gibbs:
                d_feat *= (t_cur-1) if path_type == "reverse_linear" else t_cur
            d_cur = d_cur.to(torch.float64) + d_feat
        else:
            d_cur = model.inference(
                model_input.to(dtype=_dtype), time_input.to(dtype=_dtype), **kwargs
                ).to(torch.float64)
        if (cfg_scale > 1. or y_bis is not None) and t_cur <= guidance_high and t_cur >= guidance_low:
            if y_bis is None:
                d_cur_cond, d_cur_uncond = d_cur.chunk(2)
                d_cur = d_cur_uncond + cfg_scale * (d_cur_cond - d_cur_uncond)
            else:
                d_cur_cond, d_cur_bis, d_cur_uncond = d_cur.chunk(3)
                d_cur = d_cur_uncond + cfg_scale * (d_cur_cond - d_cur_uncond) + cfg_scale * (d_cur_bis - d_cur_uncond)

        # Euler Update
        x_next = x_cur + (t_next - t_cur) * d_cur
        if heun and (i < num_steps - 1):
            # Prepare Inputs (Heun Step)
            if (cfg_scale > 1.0 or y_bis is not None) and t_cur <= guidance_high and t_cur >= guidance_low:
                model_input = torch.cat([x_next] * 2 if y_bis is None else [x_next] * 3)
                if y_bis is None:
                    y_cur = torch.cat([y, y_null], dim=0)
                else:
                    y_cur = torch.cat([y, y_bis, y_null], dim=0)
            else:
                model_input = x_next
                y_cur = y
            kwargs = dict(y=y_cur)
            time_input = torch.ones(model_input.size(0)).to(
                device=model_input.device, dtype=torch.float64
                ) * t_next

            if potential is not None and t_cur <= guidance_high and t_cur >= guidance_low:
                d_prime, zs = model.inference_feats(model_input.to(dtype=_dtype), time_input.to(dtype=_dtype), use_projector=use_projector, **kwargs)
                d_feat = feature_dir_update(model_input, zs[0], potential).to(torch.float64)
                if gibbs:
                    d_feat *= (t_cur-1) if path_type == "reverse_linear" else t_cur
                d_prime = d_prime.to(torch.float64) + d_feat
            else:
                d_prime = model.inference(
                    model_input.to(dtype=_dtype), time_input.to(dtype=_dtype), **kwargs
                    ).to(torch.float64)
            
            # Guidance Composition (Heun Step)
            if (cfg_scale > 1.0 or y_bis is not None) and t_cur <= guidance_high and t_cur >= guidance_low:
                if y_bis is None:
                    d_prime_cond, d_prime_uncond = d_prime.chunk(2)
                    d_prime = d_prime_uncond + cfg_scale * (d_prime_cond - d_prime_uncond)
                else:
                    d_prime_cond, d_prime_bis, d_prime_uncond = d_prime.chunk(3)
                    d_prime = d_prime_uncond + cfg_scale * (d_prime_cond - d_prime_uncond) + cfg_scale * (d_prime_bis - d_prime_uncond)
            
            # Average Update
            x_next = x_cur + (t_next - t_cur) * (0.5 * d_cur + 0.5 * d_prime)

    return x_next


def euler_maruyama_sampler(
        model,
        latents,
        y,
        num_steps=20,
        potential=None,
        heun=False,  # not used, just for compatability
        cfg_scale=1.0,
        guidance_low=0.0,
        guidance_high=1.0,
        path_type="linear",
        y_bis=None,
        gibbs=True,
        use_projector=True,
    ):
    # setup conditioning
    if cfg_scale > 1.0 or y_bis is not None:
        y_null = torch.tensor([1000] * y.size(0), device=y.device)
            
    _dtype = latents.dtype
    
    t_steps = torch.linspace(1., 0.04, num_steps, dtype=torch.float64)
    t_steps = torch.cat([t_steps, torch.tensor([0.], dtype=torch.float64)])
    if path_type == "reverse_linear":
        t_steps = torch.flip(t_steps, dims=[0])
    x_next = latents.to(torch.float64)
    device = x_next.device

    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-2], t_steps[1:-1])):
        dt = t_next - t_cur
        x_cur = x_next
        if (cfg_scale > 1.0 or y_bis is not None) and t_cur <= guidance_high and t_cur >= guidance_low:
            model_input = torch.cat([x_cur] * 2 if y_bis is None else [x_cur] * 3, dim=0)
            if y_bis is None:
                y_cur = torch.cat([y, y_null], dim=0)
            else:
                y_cur = torch.cat([y, y_bis,y_null], dim=0)
        else:
            model_input = x_cur
            y_cur = y            
        kwargs = dict(y=y_cur)
        time_input = torch.ones(model_input.size(0)).to(device=device, dtype=torch.float64) * t_cur
        diffusion = compute_diffusion((1-t_cur) if path_type == "reverse_linear" else t_cur)  
        sign_score = -1 if path_type == "reverse_linear" else 1          
        eps_i = torch.randn_like(x_cur).to(device)
        deps = eps_i * torch.sqrt(torch.abs(dt))

        # compute drift
        if potential is not None and t_cur <= guidance_high and t_cur >= guidance_low:
            v_cur, zs = model.inference_feats(model_input.to(dtype=_dtype), time_input.to(dtype=_dtype), use_projector=use_projector, **kwargs)
            s_cur = get_score_from_velocity(v_cur, model_input, time_input, path_type=path_type)
            d_cur = v_cur - 0.5 * diffusion * s_cur * sign_score
            d_feat = feature_dir_update(model_input, zs[0], potential).to(torch.float64)
            if gibbs:
                d_feat *= 0.5 * diffusion * sign_score
            d_cur = d_cur.to(torch.float64) + d_feat
        else:
            v_cur = model.inference(
                model_input.to(dtype=_dtype), time_input.to(dtype=_dtype), **kwargs
            ).to(torch.float64)
            s_cur = get_score_from_velocity(v_cur, model_input, time_input, path_type=path_type)
            d_cur = v_cur - 0.5 * diffusion * s_cur *  sign_score
        if (cfg_scale > 1.0 or y_bis is not None) and t_cur <= guidance_high and t_cur >= guidance_low:
            if y_bis is None:
                d_cur_cond, d_cur_uncond = d_cur.chunk(2)
                d_cur = d_cur_uncond + cfg_scale * (d_cur_cond - d_cur_uncond)
            else:
                d_cur_cond, d_cur_bis, d_cur_uncond = d_cur.chunk(3)
                d_cur = d_cur_uncond + cfg_scale * (d_cur_cond - d_cur_uncond) + cfg_scale * (d_cur_bis - d_cur_uncond)

        x_next =  x_cur + d_cur * dt + torch.sqrt(diffusion) * deps

    # last step
    t_cur, t_next = t_steps[-2], t_steps[-1]
    dt = t_next - t_cur
    x_cur = x_next
    if (cfg_scale > 1.0 or y_bis is not None) and t_cur <= guidance_high and t_cur >= guidance_low:
        model_input = torch.cat([x_cur] * 2 if y_bis is None else [x_cur] * 3, dim=0)
        if y_bis is None:
            y_cur = torch.cat([y, y_null], dim=0)
        else:
            y_cur = torch.cat([y, y_bis,y_null], dim=0)
    else:
        model_input = x_cur
        y_cur = y            
    kwargs = dict(y=y_cur)
    time_input = torch.ones(model_input.size(0)).to(
        device=device, dtype=torch.float64
        ) * t_cur
    
    # compute drift
    if potential is not None and t_cur <= guidance_high and t_cur >= guidance_low:
        v_cur, zs = model.inference_feats(model_input.to(dtype=_dtype), time_input.to(dtype=_dtype), use_projector=use_projector, **kwargs)
        s_cur = get_score_from_velocity(v_cur, model_input, time_input, path_type=path_type)
        d_cur = v_cur - 0.5 * diffusion * s_cur * sign_score
        d_feat = feature_dir_update(model_input, zs[0], potential).to(torch.float64)
        if gibbs:
            d_feat *= 0.5 * diffusion * sign_score
        d_cur = d_cur.to(torch.float64) + d_feat
    else:
        v_cur = model.inference(
            model_input.to(dtype=_dtype), time_input.to(dtype=_dtype), **kwargs
        ).to(torch.float64)
        s_cur = get_score_from_velocity(v_cur, model_input, time_input, path_type=path_type)
        d_cur = v_cur - 0.5 * diffusion * s_cur * sign_score
    if (cfg_scale > 1.0 or y_bis is not None) and t_cur <= guidance_high and t_cur >= guidance_low:
        if y_bis is None:
            d_cur_cond, d_cur_uncond = d_cur.chunk(2)
            d_cur = d_cur_uncond + cfg_scale * (d_cur_cond - d_cur_uncond)
        else:
            d_cur_cond, d_cur_bis, d_cur_uncond = d_cur.chunk(3)
            d_cur = d_cur_uncond + cfg_scale * (d_cur_cond - d_cur_uncond) + cfg_scale * (d_cur_bis - d_cur_uncond)

    mean_x = x_cur + dt * d_cur
    return mean_x

def euler_sampler_with_intermediates(
        model,
        latents,
        y,
        potential=None,
        num_steps=20,
        heun=False,
        cfg_scale=1.0,
        guidance_low=0.0,
        guidance_high=1.0,
        path_type="linear", # not used, just for compatability
    ):
    # setup conditioning
    if cfg_scale > 1.0:
        y_null = torch.tensor([1000] * y.size(0), device=y.device)
    _dtype = latents.dtype    
    t_steps = torch.linspace(1, 0, num_steps+1, dtype=torch.float64)
    x_next = latents.to(torch.float64)
    device = x_next.device

    intermediate_x = [x_next.unsqueeze(0)]

    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        x_cur = x_next
        if cfg_scale > 1.0 and t_cur <= guidance_high and t_cur >= guidance_low:
            model_input = torch.cat([x_cur] * 2, dim=0)
            y_cur = torch.cat([y, y_null], dim=0)
        else:
            model_input = x_cur
            y_cur = y            
        kwargs = dict(y=y_cur)
        time_input = torch.ones(model_input.size(0)).to(device=device, dtype=torch.float64) * t_cur
        if potential is not None and t_cur <= guidance_high and t_cur >= guidance_low: 
            # Apply transport potential only when we are past a certain threshold of steps
            d_cur, zs = model.inference_feats(model_input.to(dtype=_dtype), time_input.to(dtype=_dtype), **kwargs)
            d_feat = feature_dir_update(model_input, zs[0], potential).to(torch.float64)
            d_cur = d_cur.to(torch.float64) + d_feat
        else:
            d_cur = model.inference(
                model_input.to(dtype=_dtype), time_input.to(dtype=_dtype), **kwargs
                ).to(torch.float64)
        if cfg_scale > 1. and t_cur <= guidance_high and t_cur >= guidance_low:
            d_cur_cond, d_cur_uncond = d_cur.chunk(2)
            d_cur = d_cur_uncond + cfg_scale * (d_cur_cond - d_cur_uncond)
        x_next = x_cur + (t_next - t_cur) * d_cur
        intermediate_x.append(x_next.unsqueeze(0))
        if heun and (i < num_steps - 1):
            if cfg_scale > 1.0 and t_cur <= guidance_high and t_cur >= guidance_low:
                model_input = torch.cat([x_next] * 2)
                y_cur = torch.cat([y, y_null], dim=0)
            else:
                model_input = x_next
                y_cur = y
            kwargs = dict(y=y_cur)
            time_input = torch.ones(model_input.size(0)).to(
                device=model_input.device, dtype=torch.float64
                ) * t_next
            d_prime = model.inference(
                model_input.to(dtype=_dtype), time_input.to(dtype=_dtype), **kwargs
                ).to(torch.float64)
            if potential is not None and t_cur <= guidance_high and t_cur >= guidance_low:
                d_prime, zs = model.inference_feats(model_input.to(dtype=_dtype), time_input.to(dtype=_dtype), **kwargs)
                d_feat = feature_dir_update(model_input, zs[0], potential).to(torch.float64)
                d_prime = d_prime.to(torch.float64) + d_feat
            else:
                d_prime = model.inference(
                    model_input.to(dtype=_dtype), time_input.to(dtype=_dtype), **kwargs
                    ).to(torch.float64)
            if cfg_scale > 1.0 and t_cur <= guidance_high and t_cur >= guidance_low:
                d_prime_cond, d_prime_uncond = d_prime.chunk(2)
                d_prime = d_prime_uncond + cfg_scale * (d_prime_cond - d_prime_uncond)
            x_next = x_cur + (t_next - t_cur) * (0.5 * d_cur + 0.5 * d_prime)

    return x_next, torch.cat(intermediate_x, dim=0)


def euler_maruyama_sampler_with_intermediates(
        model,
        latents,
        y,
        num_steps=20,
        potential=None,
        heun=False,  # not used, just for compatability
        cfg_scale=1.0,
        guidance_low=0.0,
        guidance_high=1.0,
        path_type="linear",
    ):
    # setup conditioning
    if cfg_scale > 1.0:
        y_null = torch.tensor([1000] * y.size(0), device=y.device)
            
    _dtype = latents.dtype
    
    t_steps = torch.linspace(1., 0.04, num_steps, dtype=torch.float64)
    t_steps = torch.cat([t_steps, torch.tensor([0.], dtype=torch.float64)])
    x_next = latents.to(torch.float64)
    device = x_next.device

    intermediate_x = [x_next.unsqueeze(0)]

    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-2], t_steps[1:-1])):
        dt = t_next - t_cur
        x_cur = x_next
        if cfg_scale > 1.0 and t_cur <= guidance_high and t_cur >= guidance_low:
            model_input = torch.cat([x_cur] * 2, dim=0)
            y_cur = torch.cat([y, y_null], dim=0)
        else:
            model_input = x_cur
            y_cur = y            
        kwargs = dict(y=y_cur)
        time_input = torch.ones(model_input.size(0)).to(device=device, dtype=torch.float64) * t_cur
        diffusion = compute_diffusion(t_cur)            
        eps_i = torch.randn_like(x_cur).to(device)
        deps = eps_i * torch.sqrt(torch.abs(dt))

        # compute drift
        if potential is not None and t_cur <= guidance_high and t_cur >= guidance_low:
            v_cur, zs = model.inference_feats(model_input.to(dtype=_dtype), time_input.to(dtype=_dtype), **kwargs)
            s_cur = get_score_from_velocity(v_cur, model_input, time_input, path_type=path_type)
            d_cur = v_cur - 0.5 * diffusion * s_cur
            d_feat = feature_dir_update(model_input, zs[0], potential).to(torch.float64)
            d_cur = d_cur.to(torch.float64) + d_feat
        else:
            v_cur = model.inference(
                model_input.to(dtype=_dtype), time_input.to(dtype=_dtype), **kwargs
            ).to(torch.float64)
            s_cur = get_score_from_velocity(v_cur, model_input, time_input, path_type=path_type)
            d_cur = v_cur - 0.5 * diffusion * s_cur
        if cfg_scale > 1. and t_cur <= guidance_high and t_cur >= guidance_low:
            d_cur_cond, d_cur_uncond = d_cur.chunk(2)
            d_cur = d_cur_uncond + cfg_scale * (d_cur_cond - d_cur_uncond)

        x_next =  x_cur + d_cur * dt + torch.sqrt(diffusion) * deps
        intermediate_x.append(x_next.unsqueeze(0))

    # last step
    t_cur, t_next = t_steps[-2], t_steps[-1]
    dt = t_next - t_cur
    x_cur = x_next
    if cfg_scale > 1.0 and t_cur <= guidance_high and t_cur >= guidance_low:
        model_input = torch.cat([x_cur] * 2, dim=0)
        y_cur = torch.cat([y, y_null], dim=0)
    else:
        model_input = x_cur
        y_cur = y            
    kwargs = dict(y=y_cur)
    time_input = torch.ones(model_input.size(0)).to(
        device=device, dtype=torch.float64
        ) * t_cur
    
    # compute drift
    if potential is not None and t_cur <= guidance_high and t_cur >= guidance_low:
        v_cur, zs = model.inference_feats(model_input.to(dtype=_dtype), time_input.to(dtype=_dtype), **kwargs)
        s_cur = get_score_from_velocity(v_cur, model_input, time_input, path_type=path_type)
        d_cur = v_cur - 0.5 * diffusion * s_cur
        d_feat = feature_dir_update(model_input, zs[0], potential).to(torch.float64)
        d_cur = d_cur.to(torch.float64) + d_feat
    else:
        v_cur = model.inference(
            model_input.to(dtype=_dtype), time_input.to(dtype=_dtype), **kwargs
        ).to(torch.float64)
        s_cur = get_score_from_velocity(v_cur, model_input, time_input, path_type=path_type)
        d_cur = v_cur - 0.5 * diffusion * s_cur
    if cfg_scale > 1. and t_cur <= guidance_high and t_cur >= guidance_low:
        d_cur_cond, d_cur_uncond = d_cur.chunk(2)
        d_cur = d_cur_uncond + cfg_scale * (d_cur_cond - d_cur_uncond)

    mean_x = x_cur + dt * d_cur
    intermediate_x.append(mean_x.unsqueeze(0))
    return mean_x, torch.cat(intermediate_x, dim=0)



def compute_single_step_drift_euler(
        model,
        x_cur,
        y,
        t_cur,
        potential=None,
        cfg_scale=1.0,
        guidance_low=0.0,
        guidance_high=1.0,
        _dtype=torch.float32,
    ):
    """
    Computes the drift `d_cur` for a single step of the Euler sampler.

    Args:
        model: The diffusion model.
        x_cur (torch.Tensor): The current latent state.
        y (torch.Tensor): The class label.
        t_cur (float): The current time step.
        potential (optional): The guidance potential.
        cfg_scale (float): The classifier-free guidance scale.
        guidance_low (float): The lower bound of the time range for guidance.
        guidance_high (float): The upper bound of the time range for guidance.
        _dtype (torch.dtype): The data type for model inference.

    Returns:
        torch.Tensor: The computed drift `d_cur`.
    """
    # Ensure inputs are float64 for precision
    x_cur = x_cur.to(torch.float64)
    t_cur = torch.tensor(t_cur, dtype=torch.float64)
    device = x_cur.device

    # Setup conditioning for CFG
    if cfg_scale > 1.0 and t_cur <= guidance_high and t_cur >= guidance_low:
        y_null = torch.tensor([1000] * y.size(0), device=y.device)
        model_input = torch.cat([x_cur] * 2, dim=0)
        y_cur = torch.cat([y, y_null], dim=0)
    else:
        model_input = x_cur
        y_cur = y

    kwargs = dict(y=y_cur)
    time_input = torch.ones(model_input.size(0), device=device, dtype=torch.float64) * t_cur

    # Compute drift
    d_cur, zs = model.inference_feats(model_input.to(dtype=_dtype), time_input.to(dtype=_dtype), **kwargs)
    d_feat = feature_dir_update(model_input, zs[0], potential).to(torch.float64).squeeze(0)

    d_feat = torch.norm(d_feat, dim=0)
    return d_feat

def compute_single_step_drift_euler_maruyama(
        model,
        x_cur,
        y,
        t_cur,
        potential=None,
        cfg_scale=1.0,
        guidance_low=0.0,
        guidance_high=1.0,
        path_type="linear",
        _dtype=torch.float32,
    ):
    """
    Computes the drift `d_cur` for a single step of the Euler-Maruyama sampler.

    Args:
        model: The diffusion model.
        x_cur (torch.Tensor): The current latent state.
        y (torch.Tensor): The class label.
        t_cur (float): The current time step.
        potential (optional): The guidance potential.
        cfg_scale (float): The classifier-free guidance scale.
        guidance_low (float): The lower bound of the time range for guidance.
        guidance_high (float): The upper bound of the time range for guidance.
        path_type (str): The type of path for score calculation.
        _dtype (torch.dtype): The data type for model inference.

    Returns:
        torch.Tensor: The computed drift `d_cur`.
    """
    # Ensure inputs are float64 for precision
    x_cur = x_cur.to(torch.float64)
    t_cur = torch.tensor(t_cur, dtype=torch.float64)
    device = x_cur.device

    # Setup conditioning for CFG
    if cfg_scale > 1.0 and t_cur <= guidance_high and t_cur >= guidance_low:
        y_null = torch.tensor([1000] * y.size(0), device=y.device)
        model_input = torch.cat([x_cur] * 2, dim=0)
        y_cur = torch.cat([y, y_null], dim=0)
    else:
        model_input = x_cur
        y_cur = y

    kwargs = dict(y=y_cur)
    time_input = torch.ones(model_input.size(0), device=device, dtype=torch.float64) * t_cur
    diffusion = compute_diffusion(t_cur)

    # Compute drift
    v_cur, zs = model.inference_feats(model_input.to(dtype=_dtype), time_input.to(dtype=_dtype), **kwargs)
    s_cur = get_score_from_velocity(v_cur, model_input, time_input, path_type=path_type)
    d_cur = v_cur - 0.5 * diffusion * s_cur
    d_feat = feature_dir_update(model_input, zs[0], potential).to(torch.float64).squeeze(9)

    return torch.norm(d_feat, dim=0)
