# -*- coding: utf-8 -*-
import logging
from src.data.filter_bags_prev import FilterClass
from src.data.process_data_prev import ProcessorClass
import argparse


def main():
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    parser = argparse.ArgumentParser(description="Filter the images and find corresponding imu and mqtt samples for each image. Processed files go in data/processed folder")
    parser.add_argument("input_folder", help="Input folder containing bag all raw files (with / in the end).", default = "")
    parser.add_argument("--skip_imgs", help="Whether to skip image filtering (image files still needed)", default = "false")
    args = parser.parse_args()

    #filtering out the beginning and end of the bag files
    if args.skip_imgs == "false":
        filter = FilterClass(args.input_folder)
        filter.filterBags()

    #synchronizing the command and imu data
    processor = ProcessorClass(args.input_folder)
    processor.processData()

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
