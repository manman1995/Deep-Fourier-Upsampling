from basicsr.utils.create_lmdb import create_lmdb_for_rain13k
from basicsr.utils import scandir
from basicsr.utils.lmdb_util import make_lmdb_from_imgs
def prepare_keys(folder_path, suffix='png'):
    """Prepare image path list and keys for DIV2K dataset.

    Args:
        folder_path (str): Folder path.

    Returns:
        list[str]: Image path list.
        list[str]: Key list.
    """
    print('Reading image path list ...')
    img_path_list = sorted(
        list(scandir(folder_path, suffix=suffix, recursive=False)))
    keys = [img_path.split('.{}'.format(suffix))[0] for img_path in sorted(img_path_list)]

    return img_path_list, keys

def create_lmdb_for_rainds():
    folder_path = '/home/jieh/Dataset/DERAIN_DATASETS/RAINDS_REAL/train/rainy_image'
    lmdb_path = '/home/jieh/Dataset/DERAIN_DATASETS/RAINDS_REAL/train/rainy_image.lmdb'

    img_path_list, keys = prepare_keys(folder_path, 'png')
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys)

    folder_path = '/home/jieh/Dataset/DERAIN_DATASETS/RAINDS_REAL/train/ground_truth'
    lmdb_path = '/home/jieh/Dataset/DERAIN_DATASETS/RAINDS_REAL/train/ground_truth.lmdb'

    img_path_list, keys = prepare_keys(folder_path, 'png')
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys)

def create_lmdb_for_rain200():
    folder_path = '/home/jieh/Dataset/DERAIN_DATASETS/RAIN200H/train/rainy_image'
    lmdb_path = '/home/jieh/Dataset/DERAIN_DATASETS/RAIN200H/train/rainy_image.lmdb'

    img_path_list, keys = prepare_keys(folder_path, 'png')
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys)

    folder_path = '/home/jieh/Dataset/DERAIN_DATASETS/RAIN200H/train/ground_truth'
    lmdb_path = '/home/jieh/Dataset/DERAIN_DATASETS/RAIN200H/train/ground_truth.lmdb'

    img_path_list, keys = prepare_keys(folder_path, 'png')
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys)

    folder_path = '/home/jieh/Dataset/DERAIN_DATASETS/RAIN200L/train/rainy_image'
    lmdb_path = '/home/jieh/Dataset/DERAIN_DATASETS/RAIN200L/train/rainy_image.lmdb'

    img_path_list, keys = prepare_keys(folder_path, 'png')
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys)

    folder_path = '/home/jieh/Dataset/DERAIN_DATASETS/RAIN200L/train/ground_truth'
    lmdb_path = '/home/jieh/Dataset/DERAIN_DATASETS/RAIN200L/train/ground_truth.lmdb'

    img_path_list, keys = prepare_keys(folder_path, 'png')
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys)


def create_lmdb_for_rain13k():
    folder_path = '/home/jieh/Dataset/RAIN_SYN/RAIN13K/train/rainy_image'
    lmdb_path = '/home/jieh/Dataset/RAIN_SYN/RAIN13K/train/rainy_image.lmdb'

    img_path_list, keys = prepare_keys(folder_path, 'jpg')
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys)

    folder_path = '/home/jieh/Dataset/RAIN_SYN/RAIN13K/train/ground_truth'
    lmdb_path = '/home/jieh/Dataset/RAIN_SYN/RAIN13K/train/ground_truth.lmdb'

    img_path_list, keys = prepare_keys(folder_path, 'jpg')
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys)

def create_lmdb_for_snow100k():
    folder_path = '/home/jieh/Dataset/DESNOW_DATASETS/SNOW100K/train/snow'
    lmdb_path = '/home/jieh/Dataset/DESNOW_DATASETS/SNOW100K/train/snow.lmdb'

    img_path_list, keys = prepare_keys(folder_path, 'jpg')
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys)

    folder_path = '/home/jieh/Dataset/DESNOW_DATASETS/SNOW100K/train/gt'
    lmdb_path = '/home/jieh/Dataset/DESNOW_DATASETS/SNOW100K/train/gt.lmdb'

    img_path_list, keys = prepare_keys(folder_path, 'jpg')
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys)

def create_lmdb_for_KITTI_DATASET():
    folder_path = '/home/jieh/Dataset/DESNOW_DATASETS/KITTI_DATASET/train/snow'
    lmdb_path = '/home/jieh/Dataset/DESNOW_DATASETS/KITTI_DATASET/train/snow.lmdb'

    img_path_list, keys = prepare_keys(folder_path, 'png')
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys)

    folder_path = '/home/jieh/Dataset/DESNOW_DATASETS/KITTI_DATASET/train/gt'
    lmdb_path = '/home/jieh/Dataset/DESNOW_DATASETS/KITTI_DATASET/train/gt.lmdb'

    img_path_list, keys = prepare_keys(folder_path, 'png')
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys)

def create_lmdb_for_CSD():
    folder_path = '/home/jieh/Dataset/DESNOW_DATASETS/CSD/train/snow'
    lmdb_path = '/home/jieh/Dataset/DESNOW_DATASETS/CSD/train/snow.lmdb'

    img_path_list, keys = prepare_keys(folder_path, 'tif')
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys)

    folder_path = '/home/jieh/Dataset/DESNOW_DATASETS/CSD/train/gt'
    lmdb_path = '/home/jieh/Dataset/DESNOW_DATASETS/CSD/train/gt.lmdb'

    img_path_list, keys = prepare_keys(folder_path, 'tif')
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys)
def create_lmdb_for_CITYSCAPE_DATASET():
    folder_path = '/home/jieh/Dataset/DESNOW_DATASETS/CITYSCAPE_DATASET/train/snow'
    lmdb_path = '/home/jieh/Dataset/DESNOW_DATASETS/CITYSCAPE_DATASET/train/snow.lmdb'

    img_path_list, keys = prepare_keys(folder_path, 'png')
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys)

    folder_path = '/home/jieh/Dataset/DESNOW_DATASETS/CITYSCAPE_DATASET/train/gt'
    lmdb_path = '/home/jieh/Dataset/DESNOW_DATASETS/CITYSCAPE_DATASET/train/gt.lmdb'

    img_path_list, keys = prepare_keys(folder_path, 'png')
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys)
# create_lmdb_for_snow100k()
# create_lmdb_for_KITTI_DATASET()
# create_lmdb_for_CSD()
# create_lmdb_for_CITYSCAPE_DATASET()
# create_lmdb_for_rainds()
create_lmdb_for_rain200()