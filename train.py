import os
import pandas as pd
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
import gc; gc.enable()
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from utils import utils, loss_functions, image_generator
from model.Unet import Unet

def fit():
    seg_model.compile(optimizer=Adam(1e-3, decay=1e-6), loss=loss_functions.FocalLoss, metrics=[loss_functions.dice_coef])

    step_count = min(10, train_df.shape[0] // 48)
    aug_gen = image_generator.create_aug_gen(image_generator.make_image_gen(train_df))
    loss_history = [seg_model.fit(aug_gen,
                                  steps_per_epoch=step_count,
                                  epochs=20,
                                  validation_data=(valid_x, valid_y),
                                  callbacks=callbacks_list,
                                  workers=1
                                  )]
    return loss_history

base_directory = ""
train_directory = os.listdir(base_directory + "/data/train/")
test_directory = os.listdir(base_directory + "/data/test/")
train_ship_seg = pd.read_csv(os.path.join(base_directory, 'train_ship_segmentations_v2.csv'))
train_ship_seg['Ships'] = train_ship_seg['EncodedPixels'].map(lambda c_row: 1 if isinstance(c_row, str) else 0)
unique_img_ids = train_ship_seg.groupby('ImageId').agg({'Ships': 'sum'}).reset_index()
balanced_data = unique_img_ids.groupby('Ships').apply(lambda x: x.sample(4000) if len(x) > 4000 else x)

train_ids, valid_ids = train_test_split(balanced_data,
                                        test_size=0.2,
                                        stratify=balanced_data['Ships'])
train_ids = train_ids.drop('Ships', axis=1)
valid_ids = valid_ids.drop('Ships', axis=1)
train_df = pd.merge(train_ship_seg, train_ids)
valid_df = pd.merge(train_ship_seg, valid_ids)
train_gen = image_generator.make_image_gen(train_df)
train_x, train_y = next(train_gen)
valid_x, valid_y = next(image_generator.make_image_gen(valid_df, 900))
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
cur_gen = image_generator.create_aug_gen(train_gen)
t_x, t_y = next(cur_gen)
gc.collect()
weight_path = "trained_models_v2/{}_weights.best.hdf5".format('seg_model')

checkpoint = ModelCheckpoint(weight_path, monitor='val_dice_coef', verbose=1,
                             mode='max', save_weights_only = True)

reduceLROnPlat = ReduceLROnPlateau(monitor='val_dice_coef', factor=0.2,
                                   patience=3,
                                   verbose=1, mode='max', min_delta=0.0001, cooldown=2, min_lr=1e-6)
early = EarlyStopping(monitor="val_dice_coef",
                      mode="max",
                      patience=15)
callbacks_list = [checkpoint, early, reduceLROnPlat]
seg_model = Unet().build()

loss_history = fit()
fullres_model = seg_model
fullres_model.save('trained_models_v2/trained_model.h5')
