import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib
import os
from os import listdir
from os.path import isfile, join
import pandas as pd

PWD = './data/Bag_to_Depth/src/bag2rgbdepth/scripts/Data'
label_dict = {
    'c': 'onConveyor',
    'h': 'atHome',
    'f': 'inFront',
    'b': 'atBin'
}


def state_image_preprocess():
    csv_pwd = f'{PWD}/state/label.csv'
    label_df = pd.DataFrame(columns=['image', 'onion', 'eef'])
    for index in range(1, 13):
        bag_pwd = f'{PWD}/BagFolder{index}'
        bag_csv = f'{bag_pwd}/label.csv'
        df = pd.read_csv(bag_csv, header=None, names=['image', 'onion', 'eef'])

        start_index = df['image'][0].find('f')
        for name in df['image']:
            try:
                # os.rename(f'{bag_pwd}/rgb_images1/{name}', f'{bag_pwd}/rgb_images1/{name[start_index:]}')
                # os.rename(f'{bag_pwd}/rgb_images1/{name[start_index:]}', f'{bag_pwd}/rgb_images1/{index}_{name[start_index:]}')

                # os.rename(f'{bag_pwd}/depth_images1/{name}', f'{bag_pwd}/depth_images1/{name[start_index:]}')
                os.rename(f'{bag_pwd}/depth_images1/d{name[start_index:]}', f'{bag_pwd}/depth_images1/{index}_d{name[start_index:]}')
            except FileNotFoundError:
                pass

        df['image'][:] = df['image'].apply(lambda x: x[start_index:])
        df['image'][:] = f'{index}_' + df['image']
        label_df = pd.concat([label_df, df], sort=False)

        df.to_csv(f'{bag_pwd}/label_new.csv', index=False, header=None)
    label_df.to_csv(csv_pwd, index=False, header=None)


def state_label():
    for i in range(7, 11):
        csv_pwd = f'{PWD}/rgb_images{i}/{i+12}_label.csv'
        rgb_pwd = f'{PWD}/rgb_images{i}'
        onlyfiles = [f for f in listdir(rgb_pwd) if isfile(join(rgb_pwd, f))]
        onlyfiles.sort()
        for image in onlyfiles:

            start_index = image.find('f')

            os.rename(f'{rgb_pwd}/{image}', f'{rgb_pwd}/{i+12}_{image[start_index:]}')
            with open(csv_pwd, 'a') as f:
                label = f'{i+12}_{image[start_index:]},,\n'
                f.write(label)


# for i in range(1, 13):
#     csv_pwd = f'{PWD}/BagFolder{i}/label.csv'
#     rgb_pwd = f'{PWD}/BagFolder{i}/rgb_images1'
#     onlyfiles = [f for f in listdir(rgb_pwd) if isfile(join(rgb_pwd, f))]
#     for image in range(len(onlyfiles)):
#         img = mpimg.imread(f'{rgb_pwd}/frame{image}.jpg')
#         plt.imshow(img)
#         plt.show(block=False)
#         onion_label = ''
#         while onion_label not in 'chfb' or not onion_label:
#             onion_label = input('Onion:\nc: onConveyor, h:atHome, f:inFront, b:onBin\n')
#
#         eef_label = ''
#         while eef_label not in 'chfb' or not eef_label:
#             eef_label = input('Eef:\nc: onConveyor, h:atHome, f:inFront, b:onBin\n')
#         plt.close()
#         with open(csv_pwd, 'a') as f:
#             label = f'frame{image}.jpg,{label_dict[onion_label]},{label_dict[eef_label]}\n'
#             f.write(label)


if __name__ == '__main__':
    # state_image_preprocess()
    state_label()
