import tqdm
import numpy as np
import oklib3 as oklib
from PIL import Image
from scipy.ndimage import zoom
from pathlib import Path
from multiprocessing import Pool


from skimage.metrics import structural_similarity
import zlib
import bz2
import lzma

from Huff import huffman_encode


allowed_tile_sizes = sorted([4, 8, 16])


# fmt: off
thresholds = [
    (00, 00), (00, 10), (00, 20), (00, 30), (00, 40), (00, 50), (00, 60), (00, 70), (00, 80), (00, 90), (00, 100),
              (10, 10), (10, 20), (10, 30), (10, 40), (10, 50), (10, 60), (10, 70), (10, 80), (10, 90), (10, 100),
                        (20, 20), (20, 30), (20, 40), (20, 50), (20, 60), (20, 70), (20, 80), (20, 90), (20, 100),
                                  (30, 30), (30, 40), (30, 50), (30, 60), (30, 70), (30, 80), (30, 90), (30, 100),
                                            (40, 40), (40, 50), (40, 60), (40, 70), (40, 80), (40, 90), (40, 100),
                                                      (50, 50), (50, 60), (50, 70), (50, 80), (50, 90), (50, 100),
                                                                (60, 60), (60, 70), (60, 80), (60, 90), (60, 100),
                                                                          (70, 70), (70, 80), (70, 90), (70, 100),
                                                                                    (80, 80), (80, 90), (80, 100),
                                                                                              (90, 90), (90, 100),
                                                                                                       (100, 100),
]
# fmt: on


def get_variances(okx, maxtilesize):
    variances = np.zeros((okx.shape[0]//maxtilesize, okx.shape[1]//maxtilesize), dtype=np.float32)

    for i in np.r_[: okx.shape[0] : maxtilesize]:
        for j in np.r_[: okx.shape[1] : maxtilesize]:
            variances[i//maxtilesize, j//maxtilesize] = np.var(okx[i : i + maxtilesize, j : j + maxtilesize])
    
    return variances


def tiled_encdec_image(okx, tile_size):

    # Larger DCTs will have a more penalising qTable.
    # Sorry, I couldn't think of a better name
    penalty = 1
    if tile_size ==  4: penalty = 1
    if tile_size ==  8: penalty = 4
    if tile_size == 16: penalty = 16

    q_scaled = zoom(oklib.dqt_90_dct_lum.reshape((8, 8)), tile_size / 8.0, order=3)

    qmat = np.rint(q_scaled * penalty).astype(np.uint16)

    # print(tile_size, qmat)

    okx_dct = oklib.perform_dct(okx, stride=tile_size).astype(np.int16) # NEW: DCT to int16
    okx_dct_q = oklib.quantise_dct(okx_dct, qmat, stride=tile_size).astype(np.int16) # NEW: DCT to int16
    okx_dct_uq = oklib.unquantise_dct(np.rint(okx_dct_q), qmat, stride=tile_size)
    okx_idct = oklib.perform_idct(okx_dct_uq, stride=tile_size)


    return np.rint(okx_idct).astype(np.uint8), okx_dct_q


def process_image(image_path):
    rgb = np.asarray(Image.open(image_path))
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]

    oklab = oklib.image_lsrgb_to_oklab((r, g, b))
    oklab = [np.rint(okx).astype(np.uint8) for okx in oklab]

    channels = {
        'l': oklab[0],
        'a': oklab[1],
        'b': oklab[2],
    }

    for channel in channels:
        okx = channels[channel]

        variances_okx = get_variances(okx, max(allowed_tile_sizes))

        source_tiles_oklab_4, dctq_4   = tiled_encdec_image(okx, 4)
        source_tiles_oklab_8, dctq_8   = tiled_encdec_image(okx, 8)
        source_tiles_oklab_16, dctq_16 = tiled_encdec_image(okx, 16)

        # im_4 = Image.fromarray(source_tiles_oklab_4)
        # im_4.save(image_path.parent / channel / 'dct4.png')

        # im_8 = Image.fromarray(source_tiles_oklab_8)
        # im_8.save(image_path.parent / channel / 'dct8.png')

        # im_16 = Image.fromarray(source_tiles_oklab_16)
        # im_16.save(image_path.parent / channel / 'dct16.png')


        results = ''


        for threshold_8, threshold_4 in thresholds:
            okx_result = np.copy(source_tiles_oklab_16) # Initialise from DCT16

            # Tiles that pass threshold for 8 and 4
            thresholdpass_8  = variances_okx > np.percentile(variances_okx, threshold_8)
            thresholdpass_4  = variances_okx > np.percentile(variances_okx, threshold_4)

            # Actual image pixel indexes that need to be copied
            tiles_copymask_8  = thresholdpass_8.repeat(16, axis=0).repeat(16, axis=1)
            tiles_copymask_4  = thresholdpass_4.repeat(16, axis=0).repeat(16, axis=1)


            okx_result[tiles_copymask_8] = source_tiles_oklab_8[tiles_copymask_8]
            okx_result[tiles_copymask_4] = source_tiles_oklab_4[tiles_copymask_4]

            # im = Image.fromarray(okx_result)
            # im.save(image_path.parent / channel / f'({threshold_8}, {threshold_4}).png')


            # SSIM
            ssim = structural_similarity(okx, okx_result)


            # File size
            index_th_4 = np.where(thresholdpass_4.ravel())[0]
            index_th_8 = np.setdiff1d(np.where(thresholdpass_8.ravel())[0], index_th_4)

            dpixels_th_4  = dctq_4[tiles_copymask_4]
            dpixels_th_8  = dctq_8[tiles_copymask_8 & np.invert(tiles_copymask_4)]
            dpixels_th_16 = dctq_16[np.ones(dctq_16.shape, dtype=bool) & np.invert(tiles_copymask_8)]


            data = index_th_4.tobytes() \
                + index_th_8.tobytes() \
                    + dpixels_th_4.tobytes() \
                    + dpixels_th_8.tobytes() \
                    + dpixels_th_16.tobytes()
            
            len_zlib = len(zlib.compress(data))
            len_bz2 = len(bz2.compress(data))
            len_lzma = len(lzma.compress(data))


            jpeg_data, jpeg_table = huffman_encode(data)
            len_jpeg = len(jpeg_data) + 2*len(jpeg_table)

            results += f'threshold=({threshold_8}, {threshold_4}), ssim={ssim:.6f}, orig={len(data)}, zlib={len_zlib}, bz2={len_bz2}, lzma={len_lzma}, jpeg={len_jpeg}\n'
        
        with open(image_path.parent / f'{image_path.stem}-{channel}.txt', 'w') as resfile:
            resfile.write(results)
        



path_corpus = Path("../dataset/")

images_in_corpus = sorted([x for x in path_corpus.iterdir() if x.suffix != '.txt'])

print(images_in_corpus)

# process_image(images_in_corpus[0])

with Pool(30) as p:
    list(
        tqdm.tqdm(p.imap_unordered(process_image, images_in_corpus), total=len(images_in_corpus))
    )