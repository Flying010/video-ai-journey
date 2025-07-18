import torch
import numpy as np
from tqdm import tqdm
from sampler import Sampler

WIDTH = 512 #stable diffusion only takes in this dimension
HEIGHT = 512
LATENTS_WIDTH = WIDTH // 8
LATENTS_HEIGHT = HEIGHT // 8

#strength is how much attention we want to put into the starting image
def generate(prompt: str, 
            uncond_prompt: str, # Negative prompt or empty string
            input_image=None, 
            strength=0.8, 
            do_cfg=True, 
            cfg_scale=7.5, 
            sampler_name="ddpm", 
            n_inference_steps=50, 
            models={}, 
            seed=None, 
            device=None,
            idle_device=None,
            tokenizer=None,
            eta = 0.0):
    with torch.no_grad():
        if not (0 < strength <= 1):
            raise ValueError("strength must be between 0 and 1")
        if idle_device:
            to_idle = lambda x: x.to(idle_device)
        else: 
            to_idle = lambda x: x
        generator = torch.Generator(device=device)
        if seed is None:
            generator.seed()
        else:
            generator.manual_seed(seed)
        
        clip = models["clip"]
        clip.to(device)

        if do_cfg:
            # Convert the prompt into tokens using tokenizer
            cond_tokens = tokenizer.batch_encode_plus(
                [prompt], padding="max_length", max_length=77
            ).input_ids
            # (Batch_Size, Seq_Len)
            cond_tokens = torch.tensor(cond_tokens, dtype=torch.long, device=device)
            # (Batch_Size, Seq_Len) -> (Batch_Size, Seq_Len, Dim)
            cond_context = clip(cond_tokens)

            #with no conditioins now
            uncond_tokens = tokenizer.batch_encode_plus(
                [uncond_prompt], padding="max_length", max_length=77
            ).input_ids
            uncond_tokens = torch.tensor(uncond_tokens, dtype=torch.long, device=device)
            uncond_context = clip(uncond_tokens)
            
            # (Batch_2, Seq_Len, Dim) = (2, 77, 768)
            context = torch.cat([cond_context, uncond_context])
        else:
            # Convert it into a list of tokens
            tokens = tokenizer.batch_encode_plus(
                [prompt], padding="max_length", max_length=77
            ).input_ids
            tokens = torch.tensor(tokens, dtype=torch.long, device=device)
            # (1, 77, 768)
            context = clip(tokens)
        to_idle(clip) #very useful if you have a limited gpu and want to offload to cpu

        if sampler_name in ["ddpm", "ddim"]:
            sampler = Sampler(generator)
            sampler.set_inference_timesteps(n_inference_steps)
        else:
            raise ValueError(f"Unknown sampler name")
        
        latents_shape = (1, 4, LATENTS_HEIGHT, LATENTS_WIDTH)

        if input_image:
            encoder = models["encoder"]
            encoder.to(device)

            input_image_tensor = input_image.resize((WIDTH, HEIGHT))
            input_image_tensor = np.array(input_image_tensor)
            # (Height, Width, Channel)
            input_image_tensor = torch.tensor(input_image_tensor, dtype=torch.float32, device=device)
            input_image_tensor = rescale(input_image_tensor, (0,255), (-1,1))
            # (Height, Width, Channel) -> (Batch_Size, Height, Width, Channel)
            input_image_tensor = input_image_tensor.unsqueeze(0)
            # (Batch_Size, Height, Width, Channel) -> (Batch_Size, Channel, Height, Width)
            input_image_tensor = input_image_tensor.permute(0,3,1,2)

            encoder_noise = torch.randn(latents_shape, generator=generator, device=device)
            #run the image through the encoder of the VAE
            latents = encoder(input_image_tensor, encoder_noise)

            #Add noise to the latent, the more the strength, the stronger the noise, making the model more creative
            sampler.set_strength(strength=strength)
            latents = sampler.add_noise(latents, sampler.timesteps[0])

            to_idle(encoder)
        else:
            # If we are doing text to image, start with random noise N(0,1)
            latents = torch.randn(latents_shape, generator=generator, device=device)
        
        #999...0
        #1000 980 940 920 900 880....0, each of these time steps indicates a nosie level
        #can tell the scheduler to reduce noise according to particular time steps, defined by n_inference_steps 
        diffusion = models["diffusion"]
        diffusion.to(device)

        timesteps = tqdm(sampler.timesteps)
        for i, timestep in enumerate(timesteps):
            # (1, 320)
            time_embedding = get_time_embedding(timestep).to(device)

            # (Batch_Size, 4, Latents_Height, Latents_Width)
            model_input = latents

            if do_cfg:
                # (Batch_Size, 4, Latents_Height, Latents_Width) -> (4 * Batch_Size, 4, Latents_Height, Latents_Width)
                model_input = model_input.repeat(2, 1, 1, 1)

            # model_output is the predicted noise by the UNET
            model_output = diffusion(model_input, context, time_embedding)

            if do_cfg:
                output_cond, output_uncond = model_output.chunk(2)
                model_output = cfg_scale * (output_cond - output_uncond) + output_uncond

            # how to remove the noise from image? using the scheduler
            #Remove noise predicted by the UNET
            if sampler.sampler_name == "ddpm":
                latents = sampler.ddpm_step(timestep, latents, model_output)
            elif sampler.sampler_name == "ddim":
                latents = sampler.ddim_step(timestep, latents, model_output, eta=eta)
            else:
                raise ValueError(f"Unknown sampler name {sampler.sampler_name}")
        
        to_idle(diffusion)

        decoder = models["decoder"]
        decoder.to(device)

        images = decoder(latents)
        to_idle(decoder)

        images = rescale(images, (-1, 1), (0, 255), clamp=True)
        # (Batch_Size, Channel, Height, Width) -> (Batch_Size, Height, Width, Channel)
        images = images.permute(0, 2, 3, 1)
        images = images.to("cpu", torch.uint8).numpy()
        return images[0]

def rescale(x, old_range, new_range, clamp=False):
    old_min, old_max = old_range
    new_min, new_max = new_range
    x -= old_min
    x *= (new_max - new_min) / (old_max - old_min)
    x += new_min
    if clamp:
        x = x.clamp(new_min, new_max)
    return x

def get_time_embedding(timestep):
    freqs = torch.pow(10000, -torch.arange(start=0, end=160, dtype=torch.float32) / 160)
    # (1, 160)
    x = torch.tensor([timestep], dtype=torch.float32)[:, None] * freqs[None]
    # (1, 320)
    return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)




            



