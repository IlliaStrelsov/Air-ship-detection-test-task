import numpy as np
import os
from skimage.io import imread
from utils import utils
from keras.preprocessing.image import ImageDataGenerator

base_directory = ""
train_directory = base_directory + "/data/train/"

def make_image_gen(in_df, batch_size=48):
    all_batches = list(in_df.groupby('ImageId'))
    out_rgb = []
    out_mask = []
    while True:
        np.random.shuffle(all_batches)
        for c_img_id, c_masks in all_batches:
            rgb_path = os.path.join(train_directory, c_img_id)
            c_img = imread(rgb_path)
            c_mask = np.expand_dims(utils.masks_as_image(c_masks['EncodedPixels'].values), -1)
            if (3, 3) is not None:
                c_img = c_img[::(3, 3)[0], ::(3, 3)[1]]
                c_mask = c_mask[::(3, 3)[0], ::(3, 3)[1]]
            out_rgb += [c_img]
            out_mask += [c_mask]
            if len(out_rgb) >= batch_size:
                yield np.stack(out_rgb, 0) / 255.0, np.stack(out_mask, 0).astype(np.float32)
                out_rgb, out_mask = [], []


def create_aug_gen(in_gen, seed=None):
    np.random.seed(seed if seed is not None else np.random.choice(range(9999)))
    for in_x, in_y in in_gen:
        seed = np.random.choice(range(9999))
        g_x = image_gen.flow(255 * in_x,
                             batch_size=in_x.shape[0],
                             seed=seed,
                             shuffle=True)
        g_y = label_gen.flow(in_y,
                             batch_size=in_x.shape[0],
                             seed=seed,
                             shuffle=True)

        yield next(g_x) / 255.0, next(g_y)


dg_args = dict(featurewise_center=False,
               samplewise_center=False,
               rotation_range=45,
               width_shift_range=0.1,
               height_shift_range=0.1,
               shear_range=0.01,
               zoom_range=[0.9, 1.25],
               horizontal_flip=True,
               vertical_flip=True,
               fill_mode='reflect',
               data_format='channels_last')

image_gen = ImageDataGenerator(**dg_args)
label_gen = ImageDataGenerator(**dg_args)