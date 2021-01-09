import argparse
import torch
from models.GAN import Generator
from models.CustomLayers import Truncation
import numpy as np
from torchvision import utils


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


def create_vectors(args):
    ckpt = torch.load(args.ckpt)

    modulate = {
        k: v
        for k, v in ckpt.items()
        if "style_mod" in k and "weight" in k
    }

    weight_mat = []
    for k, v in modulate.items():
        weight_mat.append(v)

    W = torch.cat(weight_mat, 0)
    eigvec = torch.svd(W).V.to("cpu")

    torch.save({"ckpt": args.ckpt, "eigvec": eigvec}, args.out)
    print("Done.")
    return {"ckpt": args.ckpt, "eigvec": eigvec}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Extract factor/eigenvectors of latent spaces using closed form factorization"
    )
    parser.add_argument(
        "--out", type=str, default="factor.pt", help="name of the result factor file"
    )
    parser.add_argument("--ckpt", type=str, required=True, help="stylegan2 checkpoints")

    args = parser.parse_args()
    vectors = create_vectors(args)
