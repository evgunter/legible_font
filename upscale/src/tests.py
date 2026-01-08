import os
import sys
from PIL import Image
from main import REPO_ID, MODEL_ID, upscale_images, downscale_images, IMAGE_SIZE, OptimizableImages, DownscaleAndInitialUpscale, NoStructureAndInitialRandom
from min_res import get_boards
import torch
import numpy as np
from torchvision import transforms

TEST_IMAGE_DIR = os.path.join(os.path.dirname(__file__), "../../test_data")
TEST_IMAGE_DIR = os.path.abspath(TEST_IMAGE_DIR)

def test_similarities():
    """verify that img1.jpg and img1_modified.jpg are much more similar to each other than to img2.
    requires loading the model, so takes a few seconds"""
    # load images, process them for input into the model, load the model, get the embeddings, and find the distances between them
    original_images = [Image.open(os.path.join(TEST_IMAGE_DIR, filename)) for filename in ["img1.jpg", "img1_modified.jpg", "img2.jpg"]]
    scaled_images = [img.resize((14*20, 14*20)) for img in original_images]
    image_tensors = torch.stack([transforms.ToTensor()(image.convert("RGB")).unsqueeze(0) for image in scaled_images])
    model = torch.hub.load(REPO_ID, MODEL_ID)
    img1_embedding, img1_modified_embedding, img2_embedding = [model(img) for img in image_tensors]
    img1_img1_modified_distance = torch.dist(img1_embedding, img1_modified_embedding)
    img1_img2_distance = torch.dist(img1_embedding, img2_embedding)
    img1_modified_img2_distance = torch.dist(img1_modified_embedding, img2_embedding)

    print(f"img1.jpg and img1_modified.jpg: {img1_img1_modified_distance}")
    print(f"img1.jpg and img2.jpg:          {img1_img2_distance}")
    print(f"img1_modified.jpg and img2.jpg: {img1_modified_img2_distance}")

    # assert that img1.jpg and img1_modified.jpg are much more similar to each other than to img2,
    assert img1_img1_modified_distance < 0.3 * img1_img2_distance
    assert img1_img1_modified_distance < 0.3 * img1_modified_img2_distance

    # ... and that the distances from img1 and img1_modified to img2 are similar
    assert img1_img2_distance < 1.1 * img1_modified_img2_distance
    assert img1_modified_img2_distance < 1.1 * img1_img2_distance

def test_scaling():
    """verify that upscaling and downscaling the images does not change them"""
    images = torch.randn(10, 3, 14, 14)
    assert torch.allclose(images, downscale_images(upscale_images(images, 20), 20))

def test_image_initializer_random():
    """check that the random initializer does not error"""
    image_initializer = NoStructureAndInitialRandom(10)
    OptimizableImages(image_initializer, None, "cpu")

def test_image_initializer_downscale():
    """check that the downscale initializer does not error"""
    image_initializer = DownscaleAndInitialUpscale(torch.randn(10, 3, 3).numpy())
    OptimizableImages(image_initializer, None, "cpu")

def test_image_initializer_round_trip():
    """verify that the initializer inverts sigmoid for upscaled boards (no noise)"""
    boards = (torch.rand(4, 3, 3) > 0.5).float().numpy()
    image_initializer = DownscaleAndInitialUpscale(boards, noise_scale=0)
    init_images = torch.sigmoid(image_initializer.initializer())
    expected = upscale_images(image_initializer.boards, image_initializer.scale_factor)
    assert torch.allclose(init_images, expected, atol=1e-6)

def test_board_aspect_ratio():
    """check that the boards have the same aspect ratio as the images"""
    _, (n, m) = get_boards()
    x, y = IMAGE_SIZE
    assert n / m == x / y, f"boards have aspect ratio {n/m}, images have aspect ratio {x/y}"

def test_model_input_images():
    """verify that .images() returns values between 0 and 1"""
    image_initializer = NoStructureAndInitialRandom(10)

    opt_images = OptimizableImages(image_initializer, None, "cpu")

    # although these checks are not literally guaranteed to pass given the random initialization,
    # if this is a tensor with k values, the probability that all are in (0, 1) is
    # (1/2 (erf(1/sqrt(2) * 1) - erf(1/sqrt(2) * 0)))^k < 0.342^k
    # so to be 99.99999% sure that they do pass, we can take 10^-7 < 0.342^k <= k > 15
    b, c, h, w = opt_images.image_data.shape
    assert b * c * h * w > 15
    assert torch.any(opt_images.image_data < 0)
    assert torch.any(opt_images.image_data > 1)

    images = opt_images.images()

    assert torch.all(images >= 0)
    assert torch.all(images <= 1)


if __name__ == "__main__":
    # check if the --slow flag is passed
    SLOW = "--slow" in sys.argv

    test_scaling()
    test_image_initializer_random()
    test_image_initializer_downscale()
    test_image_initializer_round_trip()
    test_board_aspect_ratio()
    test_model_input_images()

    if SLOW:
        test_similarities()
