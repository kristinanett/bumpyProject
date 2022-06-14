# -*- coding: utf-8 -*-
import logging
from src.data.filter_and_process2 import ProcessDay
import argparse
import glob


def main():
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).

        If there is previously processed data the code will append to that. I not it will create new.
    """
    logger = logging.getLogger(__name__)
    logger.info('Making final data set from raw data')

    parser = argparse.ArgumentParser(description="Filter the images and find corresponding imu and mqtt samples for each image. Processed files go in data/processed folder")
    parser.add_argument("input_folder", help="Input folder containing one day of raw files (with / in the end).")
    args = parser.parse_args()

    #Processing the data
    processor = ProcessDay(args.input_folder)
    processor.processBags()

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
