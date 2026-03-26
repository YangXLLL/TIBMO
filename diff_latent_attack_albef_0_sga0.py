from typing import Optional
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from torch import optim
import ruamel.yaml as yaml
from models.model_retrieval import ALBEF
from models.tokenization_bert import BertTokenizer
import torch.nn.functional as F
from transformers import BertForMaskedLM
import os
from attack.attacker import TextAttacker
import torchvision.transforms as transforms
from utils import view_images, aggregate_attention
from distances import LpDistance

def preprocess(image, res=512):
    image = image.resize((res, res), resample=Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)[:, :3, :, :].cuda()
    return 2.0 * image - 1.0

def encoder(image, model, res=512):
    generator = torch.Generator().manual_seed(8888)
    image = preprocess(image, res)
    gpu_generator = torch.Generator(device=image.device)
    gpu_generator.manual_seed(generator.initial_seed())
    return 0.18215 * model.vae.encode(image).latent_dist.sample(generator=gpu_generator)

@torch.no_grad()
def ddim_reverse_sample(image, prompt, model, num_inference_steps: int = 20,
                        guidance_scale: float = 2.5, res=512):
    batch_size = 1
    max_length = 77

    uncond_input = model.tokenizer(
        [""], padding="max_length", max_length=max_length, return_tensors="pt"
    )
    uncond_embeds = model.text_encoder(uncond_input.input_ids.to(model.device))[0]

    text_input = model.tokenizer(
        prompt[0], padding="max_length", max_length=max_length,
        truncation=True, return_tensors="pt"
    )
    text_embeds = model.text_encoder(text_input.input_ids.to(model.device))[0]

    context = torch.cat([uncond_embeds, text_embeds])
    model.scheduler.set_timesteps(num_inference_steps)

    latents = encoder(image, model, res=res)
    timesteps = model.scheduler.timesteps.flip(0)
    all_latents = [latents]

    for t in tqdm(timesteps[:-1], desc="DDIM_inverse"):
        latents_input = torch.cat([latents] * 2)
        noise_pred = model.unet(latents_input, t, encoder_hidden_states=context)["sample"]

        noise_uncond, noise_text = noise_pred.chunk(2)
        noise_pred = noise_uncond + guidance_scale * (noise_text - noise_uncond)

        step_size = model.scheduler.config.num_train_timesteps // model.scheduler.num_inference_steps
        next_t = t + step_size
        alpha_bar_next = model.scheduler.alphas_cumprod[next_t] if next_t <= model.scheduler.config.num_train_timesteps else torch.tensor(0.0)

        alpha_t = model.scheduler.alphas_cumprod[t]
        reverse_x0 = (1 / torch.sqrt(alpha_t)) * (latents - noise_pred * torch.sqrt(1 - alpha_t))
        latents = reverse_x0 * torch.sqrt(alpha_bar_next) + torch.sqrt(1 - alpha_bar_next) * noise_pred
        all_latents.append(latents)

    return latents, all_latents

def register_attention_control(model, controller):
    def ca_forward(self, place_in_unet):
        def forward(hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None):
            if self.spatial_norm is not None:
                hidden_states = self.spatial_norm(hidden_states, temb)

            batch_size, seq_len, _ = hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape

            if attention_mask is not None:
                attention_mask = self.prepare_attention_mask(attention_mask, seq_len, batch_size)
                attention_mask = attention_mask.view(batch_size, self.heads, -1, attention_mask.shape[-1])

            if self.group_norm is not None:
                hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

            query = self.to_q(hidden_states)
            is_cross = encoder_hidden_states is not None

            if encoder_hidden_states is None:
                encoder_hidden_states = hidden_states
            elif self.norm_cross:
                encoder_hidden_states = self.norm_encoder_hidden_states(encoder_hidden_states)

            key = self.to_k(encoder_hidden_states)
            value = self.to_v(encoder_hidden_states)

            def reshape_heads(tensor):
                b, s, d = tensor.shape
                tensor = tensor.reshape(b, s, self.heads, d//self.heads).permute(0,2,1,3).reshape(b*self.heads, s, d//self.heads)
                return tensor

            query = reshape_heads(query)
            key = reshape_heads(key)
            value = reshape_heads(value)

            sim = torch.einsum("b i d, b j d -> b i j", query, key) * self.scale
            attn = sim.softmax(dim=-1)
            attn = controller(attn, is_cross, place_in_unet)
            out = torch.einsum("b i j, b j d -> b i d", attn, value)

            def reshape_batch(tensor):
                b, s, d = tensor.shape
                tensor = tensor.reshape(b//self.heads, self.heads, s, d).permute(0,2,1,3).reshape(b//self.heads, s, d*self.heads)
                return tensor

            out = reshape_batch(out)
            out = self.to_out[0](out)
            out = self.to_out[1](out)
            out = out / self.rescale_output_factor
            return out
        return forward

    def register_recr(net_, count, place):
        if net_.__class__.__name__ == "Attention":
            net_.forward = ca_forward(net_, place)
            return count + 1
        for child in net_.children():
            count = register_recr(child, count, place)
        return count

    cross_count = 0
    for name, net in model.unet.named_children():
        if "down" in name:
            cross_count += register_recr(net, 0, "down")
        elif "up" in name:
            cross_count += register_recr(net, 0, "up")
        elif "mid" in name:
            cross_count += register_recr(net, 0, "mid")
    controller.num_att_layers = cross_count

def reset_attention_control(model):
    def ca_forward(self):
        def forward(hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None):
            if self.spatial_norm is not None:
                hidden_states = self.spatial_norm(hidden_states, temb)

            batch_size, seq_len, _ = hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape

            if attention_mask is not None:
                attention_mask = self.prepare_attention_mask(attention_mask, seq_len, batch_size)
                attention_mask = attention_mask.view(batch_size, self.heads, -1, attention_mask.shape[-1])

            if self.group_norm is not None:
                hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

            query = self.to_q(hidden_states)
            if encoder_hidden_states is None:
                encoder_hidden_states = hidden_states
            elif self.norm_cross:
                encoder_hidden_states = self.norm_encoder_hidden_states(encoder_hidden_states)

            key = self.to_k(encoder_hidden_states)
            value = self.to_v(encoder_hidden_states)

            def reshape_heads(tensor):
                b, s, d = tensor.shape
                tensor = tensor.reshape(b, s, self.heads, d//self.heads).permute(0,2,1,3).reshape(b*self.heads, s, d//self.heads)
                return tensor

            query = reshape_heads(query)
            key = reshape_heads(key)
            value = reshape_heads(value)

            sim = torch.einsum("b i d, b j d -> b i j", query, key) * self.scale
            attn = sim.softmax(dim=-1)
            out = torch.einsum("b i j, b j d -> b i d", attn, value)

            def reshape_batch(tensor):
                b, s, d = tensor.shape
                tensor = tensor.reshape(b//self.heads, self.heads, s, d).permute(0,2,1,3).reshape(b//self.heads, s, d*self.heads)
                return tensor

            out = reshape_batch(out)
            out = self.to_out[0](out)
            out = self.to_out[1](out)
            out = out / self.rescale_output_factor
            return out
        return forward

    def register_recr(net_):
        if net_.__class__.__name__ == "Attention":
            net_.forward = ca_forward(net_)
        for child in net_.children():
            register_recr(child)

    for name, net in model.unet.named_children():
        if name in ["down", "up", "mid"]:
            register_recr(net)

def init_latent(latent, model, height, width, batch_size):
    latents = latent.expand(batch_size, model.unet.in_channels, height//8, width//8).to(model.device)
    return latent, latents

def diffusion_step(model, latents, context, t, guidance_scale):
    latents_input = torch.cat([latents] * 2)
    noise_pred = model.unet(latents_input, t, encoder_hidden_states=context)["sample"]
    noise_uncond, noise_text = noise_pred.chunk(2)
    noise_pred = noise_uncond + guidance_scale * (noise_text - noise_uncond)
    latents = model.scheduler.step(noise_pred, t, latents)["prev_sample"]
    return latents

def latent2image(vae, latents):
    latents = 1 / 0.18215 * latents
    image = vae.decode(latents)['sample']
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()
    return (image * 255).astype(np.uint8)

def load_albef(config_path, checkpoint_path):
    config = yaml.load(open(config_path, 'r'), Loader=yaml.Loader)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    model = ALBEF(config=config, text_encoder='bert-base-uncased', tokenizer=tokenizer)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint

    for key in list(state_dict.keys()):
        if 'bert' in key:
            state_dict[key.replace('bert.', '')] = state_dict[key]
            del state_dict[key]

    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model, tokenizer, config

# Global Model Initialization
config_path = 'configs/Retrieval_flickr.yaml'
ckpt_path = 'checkpoint/flickr30k.pth'
albef_model, albef_tokenizer, albef_config = load_albef(config_path, ckpt_path)
albef_model = albef_model.cuda()

ref_bert = BertForMaskedLM.from_pretrained('bert-base-uncased').cuda()

txt_attacker = TextAttacker(ref_bert, albef_tokenizer, cls=False, max_length=30,
                            number_perturbation=1, topk=10, threshold_pred_score=0.3)

img_norm = transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                (0.26862954, 0.26130258, 0.27577711))
img_attacker = ImageAttacker(img_norm, eps=2/255, steps=10, step_size=0.5/255)

@torch.enable_grad()
def diff_attack(out, model, label, controller, num_inference_steps=20,
                guidance_scale=2.5, image=None, model_name="inception",
                save_path=r"C:\Users\PC\Desktop\output", res=224, start_step=15,
                iterations=30, verbose=True, topN=1, args=None):

    if args.Surrogate_Model == "albef":
        from dataset_caption import clean_text_shunxu as imagenet_label
    else:
        raise NotImplementedError

    label = torch.from_numpy(label).long().cuda()
    model.vae.requires_grad_(False)
    model.text_encoder.requires_grad_(False)
    model.unet.requires_grad_(False)

    albef_norm = transforms.Normalize(
        (0.48145466, 0.4578275, 0.40821073),
        (0.26862954, 0.26130258, 0.27577711)
    )

    def compute_albef_loss(image, target_text):
        image_norm = (image / 2 + 0.5).clamp(0, 1)
        proc_img = albef_norm(image_norm)
        proc_img = F.interpolate(proc_img, size=(albef_config['image_res'], albef_config['image_res']),
                                mode='bilinear', align_corners=False)

        with torch.no_grad():
            img_feat = albef_model.inference_image(img_attacker.normalization(proc_img))['image_feat']

        adv_text = txt_attacker.img_guided_attack(albef_model, target_text, img_embeds=img_feat)

        try:
            with open("a1111.txt", "a+") as f:
                f.write('\n'.join(adv_text) + "\n" if isinstance(adv_text, list) else adv_text + "\n")
        except Exception as e:
            print(f"Save adv text failed: {e}")

        print(adv_text)
        text_input = albef_tokenizer(adv_text, padding='max_length', truncation=True,
                                     max_length=30, return_tensors="pt").to(image.device)

        output = albef_model.inference(proc_img, text_input)
        sim = (output['image_feat'] @ output['text_feat'].T).mean()
        print(f"Image-text similarity: {sim.item():.4f}")
        return -sim, adv_text

    height = width = res
    target_text = [imagenet_label.refined_Label[label.item()]]
    prompt = [imagenet_label.refined_Label[label.item()]] * 2
    true_token_ids = model.tokenizer.encode(imagenet_label.refined_Label[label.item()])

    # DDIM Inversion
    latent, inv_latents = ddim_reverse_sample(image, prompt, model, num_inference_steps, 0, height)
    inv_latents = inv_latents[::-1]
    latent = inv_latents[start_step - 1]

    # Optimize unconditional embeddings
    batch_size = 1
    max_length = 77
    uncond_input = model.tokenizer([""], padding="max_length", max_length=max_length, return_tensors="pt")
    uncond_embeds = model.text_encoder(uncond_input.input_ids.to(model.device))[0]

    text_input = model.tokenizer(prompt[:1], padding="max_length", return_tensors="pt")
    text_embeds = model.text_encoder(text_input.input_ids.to(model.device))[0]

    all_uncond_embeds = []
    latent, latents = init_latent(latent, model, height, width, batch_size)

    uncond_embeds.requires_grad_(True)
    optimizer = optim.AdamW([uncond_embeds], lr=1e-1)
    mse_loss = torch.nn.MSELoss()
    context = torch.cat([uncond_embeds, text_embeds])

    for idx, t in enumerate(tqdm(model.scheduler.timesteps[1+start_step-1:], desc="Optimize_uncond")):
        for _ in range(10 + 2*idx):
            pred_latents = diffusion_step(model, latents, context, t, guidance_scale)
            optimizer.zero_grad()
            loss = mse_loss(pred_latents, inv_latents[start_step + idx])
            loss.backward()
            optimizer.step()
            context = torch.cat([uncond_embeds, text_embeds])

        with torch.no_grad():
            latents = diffusion_step(model, latents, context, t, guidance_scale).detach()
            all_uncond_embeds.append(uncond_embeds.detach().clone())

    # Latent Adversarial Attack
    uncond_embeds.requires_grad_(False)
    register_attention_control(model, controller)

    text_input = model.tokenizer(prompt, padding="max_length", return_tensors="pt")
    text_embeds = model.text_encoder(text_input.input_ids.to(model.device))[0]

    context_list = [torch.cat([emb.expand(batch_size, -1, -1), text_embeds]) for emb in all_uncond_embeds]
    orig_latent = latent.clone()
    latent.requires_grad_(True)
    optimizer = optim.AdamW([latent], lr=1e-2)
    init_img = preprocess(image, res)

    apply_mask = args.is_apply_mask
    hard_mask = args.is_hard_mask
    init_mask = torch.ones([1, 1, *init_img.shape[-2:]]).cuda() if not apply_mask else None

    pbar = tqdm(range(iterations), desc="Attack iterations")
    for _ in pbar:
        controller.reset()
        latents = torch.cat([orig_latent, latent])

        for idx, t in enumerate(model.scheduler.timesteps[1+start_step-1:]):
            latents = diffusion_step(model, latents, context_list[idx], t, guidance_scale)

        before_attn = aggregate_attention(prompt, controller, args.res//32, ("up", "down"), True, 0, False)
        after_attn = aggregate_attention(prompt, controller, args.res//32, ("up", "down"), True, 1, False)

        before_label_attn = before_attn[:, :, 1: len(true_token_ids)-1]
        after_label_attn = after_attn[:, :, 1: len(true_token_ids)-1]

        adv_img = model.vae.decode(1/0.18215 * latents)['sample'][1:] * init_mask + (1 - init_mask) * init_img

        if init_mask is None:
            mask = before_label_attn.detach().mean(-1) / before_label_attn.detach().mean(-1).max()
            init_mask = torch.nn.functional.interpolate(mask.unsqueeze(0).unsqueeze(0),
                                                       adv_img.shape[-2:], mode="bilinear").clamp(0,1)
            if hard_mask:
                init_mask = init_mask.gt(0.5).float()

        albef_loss, adv_text = compute_albef_loss(adv_img, target_text)
        attack_loss = -albef_loss * 0.5
        self_attn_loss = controller.loss
        total_loss = self_attn_loss + attack_loss

        if verbose:
            print(f"Attack: {attack_loss.item():.3f} | Self-attn: {self_attn_loss.item():.3f} | Total: {total_loss.item():.3f}")

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    # Generate final adversarial image
    with torch.no_grad():
        controller.reset()
        latents = torch.cat([orig_latent, latent])
        for idx, t in enumerate(model.scheduler.timesteps[1+start_step-1:]):
            latents = diffusion_step(model, latents, context_list[idx], t, guidance_scale)

    final_img = latent2image(model.vae, latents.detach())
    real_img = (init_img / 2 + 0.5).clamp(0,1).permute(0,2,3,1).cpu().numpy()
    pert_img = final_img[1:].astype(np.float32)/255 * init_mask.squeeze()[...,None].cpu().numpy() + (1 - init_mask.squeeze()[...,None].cpu().numpy()) * real_img

    view_images(pert_img * 255, show=False, save_path=save_path + "_adv_image.png")

    # Calculate distortion metrics
    l1_dist = LpDistance(1)(real_img, pert_img)
    l2_dist = LpDistance(2)(real_img, pert_img)
    linf_dist = LpDistance(float("inf"))(real_img, pert_img)
    print(f"L1: {l1_dist:.4f}\tL2: {l2_dist:.4f}\tLinf: {linf_dist:.4f}")

    reset_attention_control(model)
    return pert_img[0].astype(np.uint8), 0, 0