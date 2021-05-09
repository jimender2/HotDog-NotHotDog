from PIL import Image
from keras.models import load_model
import numpy as np
import sys

def preprocess_input(x, v2=True):
    x = x.astype("float32")
    x = x / 255.0
    if v2:
        x = x - 0.5
        x = x * 2.0
    return x

filename = sys.argv[1]
image = Image.open(filename) #.convert('RGB')
im_width, im_height = 100, 100

image = image.resize((im_width, im_height), Image.ANTIALIAS)


image_np = np.array(image.getdata()).reshape(
    (1, im_height, im_width, 3)).astype(np.float32) / 255.


model = load_model('hotdog.h5')

pred = model.predict(image_np)
print(pred)