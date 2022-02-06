# Prepare files for submission
import argparse
import os
import glob
import sys
import zipfile

import hdf5storage
from tqdm import tqdm
from matplotlib import pyplot as plt

from Conf import RGB_FILTER_CSV, JPEG_QUALITY, MOSAIC_FILTER_CSV, CROP, SUBMISSION_SIZE_LIMIT
from NTIRE2022Util import load_rgb_filter, createNoisyRGB, save_jpg, loadCube, create_multispectral, load_ms_filter, saveCube


def main(argv=None):
    # Argument parser
    parser = argparse.ArgumentParser(description="NTIRE2022 Spectral Submission Prep Utility")

    parser.add_argument('-i', '--in_dir',    help='Input directory for the evaluated images', required=True)
    parser.add_argument('-o', '--out_dir',    help='Empty output directory for the evaluated images (will be created)', required=True)
    parser.add_argument('-k', '--keep', help="Keep temporary files", action='store_true', default=False)

    args = parser.parse_args(argv)

    out_dir = args.out_dir
    in_dir = args.in_dir
    keep = args.keep

    print(in_dir)
    print(out_dir)

    # Create output directory
    os.makedirs(out_dir, exist_ok=True)

    # Iterate over files
    print("Cropping files from input directory")
    for file in tqdm(glob.glob(f'{in_dir}/*.mat')):
        # Load file
        cube, bands = loadCube(file)

        # Crop center area
        cube = cube[CROP]

        # Save cropped file
        saveCube(os.path.join(out_dir, f'{os.path.basename(file)[:-4]}_crop.mat'), cube, bands=bands)

    # Compress cropped files
    print("Compressing submission")
    outfile = os.path.join(out_dir, 'submission.zip')
    with zipfile.ZipFile(outfile, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zip:
        for file in tqdm(glob.glob(f'{out_dir}/*_crop.mat')):
            zip.write(file, os.path.basename(file))

    # Remove cropped files
    if not keep:
        print("Removing temporary files")
        for file in tqdm(glob.glob(f'{out_dir}/*_crop.mat')):
            os.remove(file)

    # Verify that archive is < 500MB

    if os.path.getsize(outfile) > SUBMISSION_SIZE_LIMIT:
        print("Verifying submission size - ERROR:")
        print("Submission over 500MB and unlikely to be accepted by CodaLab platform")
    else:
        print("Verifying submission size -  SUCCESS!")
        print(f'Submission generated @ {outfile}')


if __name__ == "__main__":
    main(sys.argv)
