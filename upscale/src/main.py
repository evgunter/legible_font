import torch
import os
import random
import einops
import numpy as np

import matplotlib.pyplot as plt

REPO_ID = "facebookresearch/dinov2"
MODEL_ID = "dinov2_vits14"
# IMAGE_SIZE = (224, 224)
IMAGE_SIZE = (14, 14)
CHANNEL_MODE = "gray"
# CHANNEL_MODE = "rgb"

# We want to optimize a set of images to be far apart from each other in the embedding space of the model      

class OptimizableImages(torch.nn.Module):
    def __init__(self, n_images, model, device):
        super().__init__()
        self.device = device
        assert CHANNEL_MODE in ["gray", "rgb"]
        if CHANNEL_MODE == "gray":
            n_channels = 1
        else:
            n_channels = 3
        self.images = torch.nn.Parameter(torch.randn(n_images, n_channels, *IMAGE_SIZE).to(device))
        self.optimizer = torch.optim.Adam([self.images], lr=0.1)
        self.model = model

    def model_input_images(self):
        if CHANNEL_MODE == "gray":
            input_images = einops.repeat(self.images, "b 1 h w -> b 3 h w")
        else:
            input_images = self.images
        return input_images

    def loss_min(self):
        """
        The loss is the minimum distance between any two embeddings {x_i}
        """
        embeddings = self.model(self.model_input_images())
        print(f"shape of embeddings: {embeddings.shape}")
        distances = torch.cdist(embeddings.unsqueeze(0), embeddings.unsqueeze(0)).squeeze(0)
        print(f"shape of distances: {distances.shape}")

        # assert that the distances on the diagonal are all zero (should be comparing to themselves)
        assert torch.all(distances.diagonal() == 0)
        # now, so that the self-comparisons do not affect the loss, add to the diagonal
        # an identity matrix * a value that is not smaller than the minimum of the other elements (here just the first off-diagonal element)
        out = torch.min(distances + distances[0, 1] * torch.eye(distances.shape[0]).to(self.device))
        
        return out

    def loss_energy(self):
        """
        The loss is the "potential energy" of the embeddings {x_i}:
        sum_{i,j, i =/= j} || x_i - x_j ||^-1
        """
        embeddings = self.model(self.model_input_images())
        print(f"shape of embeddings: {embeddings.shape}")
        distances = torch.cdist(embeddings.unsqueeze(0), embeddings.unsqueeze(0)).squeeze(0)
        print(f"shape of distances: {distances.shape}")

        # assert that the distances on the diagonal are all zero (should be comparing to themselves)
        assert torch.all(distances.diagonal() == 0)
        # now add in an identity matrix
        # when we sum the reciprocals of the elements of distances,
        # this will prevent division by zero but add a constant offset of \sum_i 1^-1 = |{x_i}| to the loss
        # since only changes in loss matter, this won't affect the outcome, but we adjust for it anyway
        offset = embeddings.shape[0]
        return torch.sum(1.0 / (distances + torch.eye(distances.shape[0]).to(self.device))) - offset 

    def loss(self):
        return self.loss_energy()

def main():
    model = torch.hub.load(REPO_ID, MODEL_ID)
    print(model)

    n_images = 10
    device = "cpu"
    model = torch.hub.load(REPO_ID, MODEL_ID)
    model.eval()
    model.to(device)

    opt_images = OptimizableImages(n_images, model, device)

    fig, axes = plt.subplots(1, n_images, figsize=(15, 5))

    for i in range(1000):
        if i % 10 == 0:
            # display the current images with matplotlib
            images = opt_images.images.detach().cpu()
            images = (images - images.min()) / (images.max() - images.min())
            for j in range(n_images):
                ax = axes[j] if n_images > 1 else axes
                ax.clear()  # Clear previous content from the axis        

                if CHANNEL_MODE == "gray":
                    ax.imshow(images[j, 0], cmap="gray")
                else:
                    ax.imshow(images[j].permute(1, 2, 0))

                ax.axis('off')

            plt.pause(0.1)  # Pause for a short moment to update the plot

        loss = opt_images.loss()
        print(f"loss at step {i}: {loss}")
        loss.backward()
        opt_images.optimizer.step()
        opt_images.optimizer.zero_grad()
        
    plt.show()


if __name__ == "__main__":
    main()
