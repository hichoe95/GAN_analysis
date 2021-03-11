from matplotlib import cm
import numpy as np
from PIL as Image


def make_animation(images : list, save_name : str):
    

    for i in range(len(images)):
        image = Image.fromarray((images[i]*255).astype(np.uint8))
        images[i] = image
        images[i] = images[i].resize((128,128))

    images[0].save('./animation.gif',
                   save_all=True, append_images=images[1:], 
                   optimize=False, duration=40, loop=0)