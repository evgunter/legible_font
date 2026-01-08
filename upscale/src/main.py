import torch
import torch.nn.functional as F
from torchvision import transforms
import einops
from abc import abstractmethod
import math

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
    def __init__(self, structure, model, device):
        super().__init__()
        self.device = device
        self.structure = structure
        self.image_data = torch.nn.Parameter(structure.initializer().to(device))
        self.n_images = self.image_data.shape[0]
        self.optimizer = torch.optim.Adam([self.image_data], lr=0.1)
        self.model = model

    def images(self):
        return sigmoid_images(self.image_data)

    def model_input_images(self):
        if CHANNEL_MODE == "gray":
            input_images_unscaled = einops.repeat(self.images(), "b 1 h w -> b 3 h w")
        else:
            input_images_unscaled = self.images()

        # per https://github.com/facebookresearch/dinov2/blob/main/notebooks/depth_estimation.ipynb,
        # dinov2 takes images in rgb format, with the following normalization
        input_images = transforms.Normalize(
            mean=(123.675, 116.28, 103.53),
            std=(58.395, 57.12, 57.375),
        )(input_images_unscaled * 255.0)

        return input_images
    
    def get_embeddings(self):
        return self.model(self.model_input_images())

    def embedding_distances(self):
        embeddings = self.get_embeddings()
        distances = einops.rearrange(torch.cdist(einops.rearrange(embeddings, "b d -> 1 b d"), einops.rearrange(embeddings, "b d -> 1 b d"), compute_mode='donot_use_mm_for_euclid_dist'), "1 b1 b2 -> b1 b2")

        # assert that the distances on the diagonal are all zero (should be comparing each embedding to itself)
        assert torch.all(distances.diagonal().abs() < 1e-4), f"large diagonal elements: {distances.diagonal()}"
        return distances

    def loss_min(self):
        """
        the loss is the minimum distance between any two embeddings {x_i}.
        this is simpler than loss_energy(), and might have better results since it focuses on the most similar pairs
        (which may matter more than overall similarities, since the characters only need to meet a certain threshold of distinguishability),
        but is quite slow for large sets of boards since it only affects the boards that are closest together
        (almost always just two at once)
        """
        distances = self.embedding_distances()

        # so that the self-comparisons do not affect the loss, add to the diagonal
        # an identity matrix * a value that is not smaller than the minimum of the other elements
        # (here just the first off-diagonal element)
        return - torch.min(distances + distances[0, 1] * torch.eye(distances.shape[0]).to(self.device))

    def loss_energy(self):
        """
        the loss is the "potential energy" of the embeddings {x_i}:
        sum_{i,j, i =/= j} || x_i - x_j ||^-1
        """
        distances = self.embedding_distances()

        # add in an identity matrix
        # when we sum the reciprocals of the elements of distances,
        # this will prevent division by zero but add a constant offset of 1/|{x_i}| \sum_i 1^-1 = 1 to the loss
        # only changes in loss matter for training, so this doesn't matter, but we adjust for it anyway
        return torch.mean(1.0 / (distances + torch.eye(distances.shape[0]).to(self.device))) - 1

    def loss_embeddings(self):
        # return self.loss_min()
        out = self.loss_energy()
        print(f"embedding loss  {out}")  # TODO remove
        return out

    def loss(self):
        return 10 * self.loss_embeddings() + self.structure.loss(self) + loss_continuity(self) + loss_figural(self)
    
class StructureAndInitializer:
    @abstractmethod
    def initializer(self):
        pass

    @abstractmethod
    def loss(self, optimizable_images):
        pass

def sigmoid_images(images: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(images)
class DownscaleAndInitialUpscale(StructureAndInitializer):
    def __init__(self, boards, noise_scale=0):
        self.boards = einops.rearrange(torch.tensor(boards, dtype=torch.float32), "b h w -> b 1 h w")
        assert CHANNEL_MODE in ["gray", "rgb"]
        if CHANNEL_MODE == "rgb":
            # boards are black and white, so we repeat across the rgb channels
            self.boards = einops.repeat(self.boards, "b 1 h w -> b 3 h w")

        self.scale_factor = IMAGE_SIZE[0] // self.boards.shape[2]
        # otherwise we'd have to do non-integer scaling to satisfy both the constraint that the aspect ratios are equal
        # and the constraint that the image dimensions are divisible by 14
        assert self.scale_factor % 14 == 0, f"scale factor must be a multiple of 14; got {self.scale_factor}"

        self.noise_scale = noise_scale

    def initializer(self):
        images = upscale_images(self.boards, self.scale_factor)
        # since the internal image data is not scaled, we need to undo the sigmoid that will be applied later
        # it may help training to start out with brights and darks not that saturated by having a largish eps
        eps = 0.01
        return torch.logit(images, eps=eps) + self.noise_scale * torch.randn_like(images)

    def loss_l2(self, optimizable_images):
        """
        loss term to impose the requirement that coarse-graining the images leads to a certain result
        (the base boards from min_res); does so by finding the l2 distance between the coarse-grained images and the base boards
        """
        return torch.mean((downscale_images(optimizable_images.images(), self.scale_factor) - self.boards)**2)
    
    def loss_more_gap(self, optimizable_images):
        """
        loss term to impose the requirement that coarse-graining the images leads to a certain result
        (the base boards from min_res); does so by linearly penalizing any pixel values that are between the thresholds
        """
        threshold_low = 0.5   # black
        threshold_high = 1.0  # white
        currs = downscale_images(optimizable_images.images(), self.scale_factor)

        # this only makes sense if the boards are binary, so verify that
        assert (torch.where(self.boards > 0.5, torch.ones_like(self.boards), torch.zeros_like(self.boards)) == self.boards).all()

        should_ones = torch.where(self.boards > 0.5, currs, torch.ones_like(currs))
        should_zeros = torch.where(self.boards < 0.5, currs, torch.zeros_like(currs))

        return torch.mean(torch.clamp(threshold_high - should_ones, min=0) + torch.clamp(should_zeros - threshold_low, min=0))

    def loss(self, optimizable_images):
        out = self.loss_more_gap(optimizable_images)
        print(f"structure loss  {out}")  # TODO remove
        return out

class DownscaleAndInitialRandom(DownscaleAndInitialUpscale):
    def __init__(self, boards):
        super().__init__(boards, noise_scale=0)

    def initializer(self):
        b, c, _, _ = self.boards.shape
        return torch.randn(b, c, *IMAGE_SIZE)
    
    # loss is as in DownscaleAndInitialUpscale

class NoStructureAndInitialRandom(StructureAndInitializer):
    def __init__(self, n_images):
        self.n_images = n_images

    def initializer(self):
        assert CHANNEL_MODE in ["gray", "rgb"]
        if CHANNEL_MODE == "gray":
            n_channels = 1
        else:
            n_channels = 3
        return torch.randn(self.n_images, n_channels, *IMAGE_SIZE)
    
    def loss(self, optimizable_images):
        return 0
    
def loss_adjacent_change(optimizable_images):
    """
    loss term to penalize large differences between neighboring pixels
    """
    images = optimizable_images.images()
    return torch.mean(torch.abs(images[:, :, :-1, :] - images[:, :, 1:, :])) + torch.mean(torch.abs(images[:, :, :, :-1] - images[:, :, :, 1:]))

def loss_adjacent_change_sublinear(optimizable_images):
    """
    loss term to penalize large differences between neighboring pixels
    """
    images = optimizable_images.images()
    horizontal_loss = torch.abs(images[:, :, :-1, :] - images[:, :, 1:, :])
    vertical_loss = torch.abs(images[:, :, :, :-1] - images[:, :, :, 1:])
    # verify that none of the differences are > 1
    assert torch.all(horizontal_loss <= 1)
    assert torch.all(vertical_loss <= 1)
    # monotonic and sublinear on [0, 1]
    # scale_fn = lambda x: x - x**2/3  # TODO: i had x**2/2 but i'm concerned that the 0 gradient at difference 1 might be a problem
    scale_fn = lambda x: x - x**2/2
    return torch.mean(scale_fn(horizontal_loss)) + torch.mean(scale_fn(vertical_loss))

def loss_high_frequency(optimizable_images):
    """
    loss term to penalize high-frequency noise: applies a high-pass filter and computes the mean absolute response
    """
    kernel = torch.tensor([[1, -2, 1],
                           [-2, 4, -2],
                           [1, -2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
    kernel = kernel / torch.sqrt(torch.sum(kernel**2))
    images = optimizable_images.images()
    filtered_images = F.conv2d(images, kernel, padding=1)
    return torch.mean(torch.abs(filtered_images))

def loss_continuity(optimizable_images):
    """
    loss term to try to make the characters have discernable figures
    """
    out = loss_adjacent_change_sublinear(optimizable_images)
    print(f"continuity loss {out}")  # TODO remove
    return out

def loss_figural(optimizable_images):
    """
    loss term to make the characters have sharper boundaries between black and white;
    penalizes intermediate values between black and white according to x(1-x)
    """
    images = optimizable_images.images()
    out = torch.mean(images * (1 - images))
    print(f"figural loss   {out}")  # TODO remove
    return out

def upscale_images(images: torch.Tensor, scale_factor: int) -> torch.Tensor:
    """for each image in the batch, replace the original pixel values with scale_factor x scale_factor blocks with the same values"""
    return einops.repeat(images, f"b c h w -> b c (h {scale_factor}) (w {scale_factor})")

def downscale_images(tensor: torch.Tensor, scale_factor: int) -> torch.Tensor:
    """pool each image by mean with a factor of scale_factor x scale_factor"""
    return einops.reduce(tensor, f"b c (h {scale_factor}) (w {scale_factor}) -> b c h w", reduction="mean")

def main():
    boards, (n, m) = get_boards()
    # check that the boards have the same aspect ratio as the images
    x, y = IMAGE_SIZE
    assert n / m == x / y, f"boards have aspect ratio {n/m}, images have aspect ratio {x/y}"

    # boards = boards[:10]  # TODO remove: train more quickly for testing

    n_images = len(boards)
    # image_initializer = NoStructureAndInitialRandom(n_images)
    # image_initializer = DownscaleAndInitialUpscale(boards, noise_scale=10)
    image_initializer = DownscaleAndInitialRandom(boards)

    device = "cpu"
    model = torch.hub.load(REPO_ID, MODEL_ID)
    model.eval()
    model.to(device)

    opt_images = OptimizableImages(image_initializer, model, device)

    height = int(n_images**0.5)
    width = n_images // height + 1
    fig, axes = plt.subplots(height, width, figsize=(height/2, width/2))

    for i in range(500):
        if i % 1 == 0:
            # display the current images with matplotlib

            # what the model sees:
            # images = opt_images.model_input_images().detach().cpu()
            # images = (images - images.min()) / (images.max() - images.min())

            # regular 0-1 scale:
            images = opt_images.images().detach().cpu()

            # raw values before scaling to pixel values:
            # images = opt_images.image_data.detach().cpu()
            # images = (images - images.min()) / (images.max() - images.min())
            
            for j, ax in enumerate(axes.flatten()):
                ax.clear()  # Clear previous content from the axis
                ax.axis('off')
                if j >= len(images):
                    continue

                if CHANNEL_MODE == "gray":
                    ax.imshow(images[j, 0], vmin=0, vmax=1, cmap="gray")
                else:
                    ax.imshow(images[j].permute(1, 2, 0), vmin=0, vmax=1)

            # put the step number in the title
            fig.suptitle(f"Step {i}")
            plt.pause(0.1)  # Pause for a short moment to update the plot

        loss = opt_images.loss()
        print(f"loss at step {i}: {loss}")
        loss.backward()
        opt_images.optimizer.step()
        opt_images.optimizer.zero_grad()
        
    plt.show()


if __name__ == "__main__":
    main()
