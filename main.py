import numpy as np
from PIL import Image

from transfer_func import run_style_transfer


cnt = 0


def save_image_array(im: np.ndarray):
    global cnt
    im = Image.fromarray(im)
    im.save(f"{cnt}.png")
    cnt += 1


if __name__ == "__main__":
    im1 = Image.open("Bangko.jpeg")
    im2 = Image.open("header.jpg")
    im, loss = run_style_transfer(
        content_im=im1,
        style_im=im2,
        num_iterations=300,
        display_interval=100,
        progress_callback=save_image_array,
        model_name="custom",
        # style_weight=1e2,  # needs different style weight for different models. 1e-3 for vgg and 1e2 for mobilenet
        # content_weight=1e1, # 1e3 content weight for vgg and 1e1 content weight for mobilenet
        style_weight=1e5,  # for customized model
        content_weight=1e-1, # for customized model
    )
    Image.fromarray(im).show()