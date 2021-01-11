import os

import numpy as np
from torchvision.utils import save_image

from mapper_network import VGG16_Perceptual
from models.GAN import Generator
import matplotlib.pyplot as plt
from config import cfg as opt
from torch import optim, nn
import torch
from torchvision import transforms
from PIL import Image

def load(model, cpk_file):
    pretrained_dict = torch.load(cpk_file)
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    return model

def read_img(path, resize=None):
    with open(path, 'rb') as f:
        image = Image.open(f)
        image = image.convert("RGB")
    if resize is not None:
        image = image.resize((resize, resize))
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    image = transform(image)
    image = image.unsqueeze(0)
    return image


def caluclate_loss(synth_img, img, perceptual_net, img_p, MSE_Loss, upsample2d):
    # calculate MSE Loss
    # print(synth_img.size(), img.size())
    mse_loss = MSE_Loss(synth_img, img)  # (lamda_mse/N)*||G(w)-I||^2

    # calculate Perceptual Loss
    real_0, real_1, real_2, real_3 = perceptual_net(img_p)
    synth_p = upsample2d(synth_img)  # (1,3,256,256)
    synth_0, synth_1, synth_2, synth_3 = perceptual_net(synth_p)

    perceptual_loss = MSE_Loss(synth_0, real_0)
    perceptual_loss += MSE_Loss(synth_1, real_1)
    perceptual_loss += MSE_Loss(synth_2, real_2)
    perceptual_loss += MSE_Loss(synth_3, real_3)

    return mse_loss, perceptual_loss

def adjust_dynamic_range(data, drange_in=(-1, 1), drange_out=(0, 1)):
    """
    adjust the dynamic colour range of the given input data
    :param data: input image data
    :param drange_in: original range of input
    :param drange_out: required range of output
    :return: img => colour range adjusted images
    """
    if drange_in != drange_out:
        scale = (np.float32(drange_out[1]) - np.float32(drange_out[0])) / (
                np.float32(drange_in[1]) - np.float32(drange_in[0]))
        bias = (np.float32(drange_out[0]) - np.float32(drange_in[0]) * scale)
        data = data * scale + bias
    return torch.clamp(data, min=0, max=1)

def map_to_W(img_path, resolution, n_iter=1000, device="cuda:0"):
    out_depth = int(np.log2(resolution)) - 2

    img = read_img(img_path, resize=resolution)
    img = img.to(device)

    MSE_loss = nn.MSELoss(reduction='mean')
    img_p = img.clone()
    upsample = nn.Upsample(scale_factor=256 / resolution, mode='bilinear')
    img_p = upsample(img_p)

    map_net = VGG16_Perceptual(n_layers=[2, 4, 14, 21]).to(device)
    # (log2(resolution)-1) * 2
    dlatent = torch.zeros((1, np.int((np.log2(resolution) - 1) * 2), 512), requires_grad=True, device=device)
    optimizer = optim.Adam({dlatent}, lr=0.01, betas=(0.9, 0.999), eps=1e-8)

    iterations = n_iter
    name = img_path.split("/")[-1].split(".")[0]
    for i in range(iterations):
        optimizer.zero_grad()
        synth_img = g_synthesis(dlatent, depth=out_depth, alpha=1)
        synth_img = adjust_dynamic_range(synth_img)
        mse_loss, perceptual_loss = caluclate_loss(synth_img, img, map_net, img_p, MSE_loss, upsample)
        loss = mse_loss + perceptual_loss

        loss.backward()
        optimizer.step()

        loss_np = loss.detach().cpu().numpy()
        loss_p = perceptual_loss.detach().cpu().numpy()
        loss_m = mse_loss.detach().cpu().numpy()

        if (i + 1) % 200 == 0:
            print("iter{}: loss -- {},  mse_loss --{},  percep_loss --{}".format(i, loss_np, loss_m, loss_p))
            if not os.path.exists(f"save_image/{name}"):
                os.makedirs(f"save_image/{name}")
            save_image(synth_img, f"save_image/{name}/{i}.png")
    if not os.path.exists("latent_W/"):
        os.makedirs("latent_W/")
    np.save("latent_W/{}.npy".format(name), dlatent.detach().cpu().numpy())

if __name__ == '__main__':
    device = "cuda:0" if torch.cuda.is_available() else 'cpu'
    opt.merge_from_file("configs/sample_ffhq_1024.yaml")
    opt.freeze()
    gen = Generator(resolution=opt.dataset.resolution,
                    num_channels=opt.dataset.channels,
                    structure=opt.structure,
                    **opt.model.gen)
    # print(torch.load("weights/GAN_GEN_SHADOW_5_80.pth").keys())
    # gen.load_state_dict(torch.load("weights/GAN_GEN_SHADOW_5_80.pth"))
    gen = load(gen, "weights/ffhq_1024_gen.pth")
    gen = gen.to(device)
    g_mapping, g_synthesis = gen.g_mapping, gen.g_synthesis
    resolution = opt.dataset.resolution
    n_iter = 200 if resolution == 1024 else 1000
    # for img in os.listdir("output"):
    #     map_to_W(f"output/{img}", resolution, n_iter)
    map_to_W(f"output/9wall.jpg", resolution, n_iter=5000)