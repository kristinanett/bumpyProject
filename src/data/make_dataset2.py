# -*- coding: utf-8 -*-
import logging
from src.data.filter_bags2 import ProcessDay
import argparse
import glob


def main():
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).

        Will always overwrite the previously processed data
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    parser = argparse.ArgumentParser(description="Filter the images and find corresponding imu and mqtt samples for each image. Processed files go in data/processed folder")
    parser.add_argument("--input_folder", help="Input folder containing bag all raw files (with / in the end).", default = "data/raw")
    args = parser.parse_args()

    day_folders = sorted(glob.glob(args.input_folder + "/*/"))
    print(day_folders)

    #filtering out the beginning and end of the bag files
    first_day = day_folders[0] #'data/raw/0405/'
    processor = ProcessDay(first_day)
    processor.processBags()

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
