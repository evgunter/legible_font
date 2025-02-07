import os
from PIL import Image
from main import REPO_ID, MODEL_ID
import torch
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


if __name__ == "__main__":
    test_similarities()
