import torch
import os
import random
import einops
import numpy as np

from min_res import get_boards

import matplotlib.pyplot as plt

REPO_ID = "facebookresearch/dinov2"
MODEL_ID = "dinov2_vits14"
# IMAGE_SIZE = (224, 224)
IMAGE_SIZE = (3*14, 3*14)
CHANNEL_MODE = "gray"
# CHANNEL_MODE = "rgb"

# We want to optimize a set of images to be far apart from each other in the embedding space of the model      

class OptimizableImages(torch.nn.Module):
    def __init__(self, image_initializer, model, device):
        super().__init__()
        self.device = device
        self.images = torch.nn.Parameter(image_initializer().to(device))
        self.optimizer = torch.optim.Adam([self.images], lr=0.1)
        self.model = model

    def model_input_images(self):
        if CHANNEL_MODE == "gray":
            input_images = einops.repeat(self.images, "b 1 h w -> b 3 h w")
        else:
            input_images = self.images
        return input_images

    def embedding_distances(self):
        embeddings = self.model(self.model_input_images())
        distances = torch.cdist(embeddings.unsqueeze(0), embeddings.unsqueeze(0)).squeeze(0)

        # assert that the distances on the diagonal are all zero (should be comparing each embedding to itself)
        assert torch.all(distances.diagonal().abs() < 0.1)  # TODO: i'm not sure why these are not exactly zero. this behavior only happens for n_images > 20 or so

        return distances

    def loss_downscale(self):
        """
        Loss term to impose the requirement that coarse-graining the images leads to a certain result (the base characters from min_res)
        """

    def loss_min(self):
        """
        The loss is the minimum distance between any two embeddings {x_i}
        """
        distances = self.embedding_distances()

        # so that the self-comparisons do not affect the loss, add to the diagonal
        # an identity matrix * a value that is not smaller than the minimum of the other elements
        # (here just the first off-diagonal element)
        return torch.min(distances + distances[0, 1] * torch.eye(distances.shape[0]).to(self.device))

    def loss_energy(self):
        """
        The loss is the "potential energy" of the embeddings {x_i}:
        sum_{i,j, i =/= j} || x_i - x_j ||^-1
        """
        distances = self.embedding_distances()

        # add in an identity matrix
        # when we sum the reciprocals of the elements of distances,
        # this will prevent division by zero but add a constant offset of \sum_i 1^-1 = |{x_i}| to the loss
        # only changes in loss matter for training, so this doesn't matter, but we adjust for it anyway
        return torch.sum(1.0 / (distances + torch.eye(distances.shape[0]).to(self.device))) - len(self.images)

    def loss(self):
        return self.loss_energy()

def upscale_tensor(tensor: torch.Tensor, scale_factor: int) -> torch.Tensor:
    return tensor.repeat_interleave(scale_factor, dim=0).repeat_interleave(scale_factor, dim=1)

def make_random_initializer(n_images):
    def random_initializer():
        assert CHANNEL_MODE in ["gray", "rgb"]
        if CHANNEL_MODE == "gray":
            n_channels = 1
        else:
            n_channels = 3
        return torch.randn(n_images, n_channels, *IMAGE_SIZE)
    return random_initializer

def make_min_res_initializer(boards):
    def min_res_initializer():
        assert CHANNEL_MODE in ["gray", "rgb"]
        if CHANNEL_MODE == "gray":
            n_channels = 1
        else:
            n_channels = 3

        scale_factor = IMAGE_SIZE[0] // boards[0].shape[0]
        # otherwise we'd have to do non-integer scaling to satisfy both the constraint that the aspect ratios are equal
        # and the constraint that the image dimensions are divisible by 14
        assert scale_factor % 14 == 0, f"scale factor must be a multiple of 14; got {scale_factor}"

        images = torch.zeros(len(boards), n_channels, *IMAGE_SIZE)
        for i, board in enumerate(boards):
            tns = upscale_tensor(torch.tensor(board), scale_factor)
            if CHANNEL_MODE == "gray":
                images[i, 0] = tns
            else:
                # boards are black and white, so we repeat across the rgb channels
                images[i] = torch.tensor([tns, tns, tns])
        return images
    return min_res_initializer

def main():
    boards, (n, m) = get_boards()
    # check that the boards have the same aspect ratio as the images
    x, y = IMAGE_SIZE
    assert n / m == x / y, f"boards have aspect ratio {n/m}, images have aspect ratio {x/y}"

    n_images = len(boards)
    # image_initializer = make_random_initializer(n_images)
    image_initializer = make_min_res_initializer(boards)

    model = torch.hub.load(REPO_ID, MODEL_ID)

    device = "cpu"
    model = torch.hub.load(REPO_ID, MODEL_ID)
    model.eval()
    model.to(device)

    opt_images = OptimizableImages(image_initializer, model, device)

    height = int(n_images**0.5)
    width = n_images // height + 1
    fig, axes = plt.subplots(height, width, figsize=(height/2, width/2))

    for i in range(1000):
        if i % 1 == 0:
            # display the current images with matplotlib
            images = opt_images.images.detach().cpu()
            images = (images - images.min()) / (images.max() - images.min())
            for j, ax in enumerate(axes.flatten()):
                ax.clear()  # Clear previous content from the axis
                ax.axis('off')
                if j >= len(images):
                    continue

                if CHANNEL_MODE == "gray":
                    ax.imshow(images[j, 0], cmap="gray")
                else:
                    ax.imshow(images[j].permute(1, 2, 0))

            plt.pause(0.1)  # Pause for a short moment to update the plot

        loss = opt_images.loss()
        print(f"loss at step {i}: {loss}")
        loss.backward()
        opt_images.optimizer.step()
        opt_images.optimizer.zero_grad()
        
    plt.show()


if __name__ == "__main__":
    main()
