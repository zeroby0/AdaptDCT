import re
from pathlib import Path
import numpy as np
from PIL import Image
import tensorflow as tf
import oklib3 as oklib


def parse_data(file_path):
    ssims = []
    sizes = []

    with open(file_path, "r") as file:
        for line in file:
            # Extract values using regular expressions
            match = re.match(
                r"threshold=\((\d+),\s*(\d+)\),\s*ssim=([\d.]+),\s*orig=(\d+),\s*zlib=(\d+),\s*bz2=(\d+),\s*lzma=(\d+)",
                line,
            )

            if match:
                threshold = tuple(map(int, match.group(1, 2)))

                if threshold == (10, 10) or threshold == (30, 30):
                    continue

                ssim = (float(match.group(3)),)
                # orig =  int(match.group(4)),
                # zlib =  int(match.group(5)),
                bz2 = (float(match.group(6)),)
                # lzma =  int(match.group(7))

                # BZ2 seems to perform the best
                ssims.append(ssim)
                sizes.append(bz2)

    ssims = np.round(np.asarray(ssims, dtype=np.float32).ravel())
    sizes = (
        np.round(np.asarray(sizes, dtype=np.float32).ravel() / 1000) / 1000
    )  # Reduce precision to KB

    # print("SIZE NOT IN MB ANYMORE")

    return ssims, sizes


def make_dataset(indices: list, channel = 'l'):
    dstype_x, dstype_y1, dstype_y2 = [], [], []

    for i in indices:
        image_path = Path(f'../dataset/{i}.png')

        rgb = np.asarray(Image.open(image_path))
        r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]

        oklab = oklib.image_lsrgb_to_oklab((r, g, b))
        oklab = [np.rint(okx).astype(np.uint8) for okx in oklab]

        channels = {
            'l': oklab[0],
            'a': oklab[1],
            'b': oklab[2],
        }

        image_tensor = tf.convert_to_tensor(channels[channel], dtype=tf.float32)

        ssims, sizes = parse_data(f'../dataset/{i}-{channel}.txt')

        ssims = tf.constant(ssims)
        sizes = tf.constant(sizes)

        dstype_x.append(image_tensor)
        dstype_y1.append(ssims)
        dstype_y2.append(sizes)


    dstype_x = tf.stack(dstype_x)
    dstype_y1 = tf.stack(dstype_y1)
    dstype_y2 = tf.stack(dstype_y2)

    return dstype_x, dstype_y1, dstype_y2


# def make_dataset(paths):
#     dstype_x, dstype_y1, dstype_y2 = [], [], []
#     for path_dir in paths:
#         image_path = path_dir / "original.png"

#         image_data = tf.io.read_file(str(image_path))
#         image_tensor = tf.image.decode_image(image_data, channels=1)
#         image_tensor = tf.image.convert_image_dtype(image_tensor, dtype=tf.float32)
#         image_tensor = tf.image.resize(image_tensor, [256, 256])

#         ssims, sizes = parse_data(path_dir / "results.txt")

#         ssims = tf.constant(ssims)
#         sizes = tf.constant(sizes)

#         dstype_x.append(image_tensor)
#         dstype_y1.append(ssims)
#         dstype_y2.append(sizes)

#     dstype_x = tf.stack(dstype_x)
#     dstype_y1 = tf.stack(dstype_y1)
#     dstype_y2 = tf.stack(dstype_y2)

#     return dstype_x, dstype_y1, dstype_y2


if __name__ == '__main__':
    # all_x, all_y1, all_y2 = make_dataset(
    #     [
    #         Path(f"/home/iiitb/varprism-oct24/dataset/ccrop_split_dscale_1024/{i}/l")
    #         for i in range(1, 840)
    #     ]
    # )

    all_x, all_y1, all_y2 = make_dataset(range(1, 200))

    print(tf.reshape(all_y2,[-1]).shape)

    # import matplotlib.pyplot as plt
    # plt.hist(tf.reshape(all_y2,[-1]), bins=50, edgecolor='black')

    # # Add grid
    # plt.grid(True)

    # # Add labels and title
    # plt.xlabel('Value')
    # plt.ylabel('Frequency')
    # plt.title('Histogram with Grid')

    # # Show the plot
    # plt.show()
