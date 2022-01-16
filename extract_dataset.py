DIRECTORY = "dataset/"
DATASET_NAME = "dow_jones_index"
FORMAT_END = "_formatted"

import zipfile, os

dir_list = os.listdir(DIRECTORY)
# check if dataset folder have extracted files
if (
    not DATASET_NAME + ".data" in dir_list
    or not DATASET_NAME + ".names" in dir_list
):
    # if not, extract all
    with zipfile.ZipFile(DIRECTORY + DATASET_NAME + ".zip", "r") as zip:
        zip.extractall(DIRECTORY)

# remove all dolar sign from data if it's not done yet
if not DATASET_NAME + FORMAT_END + ".data" in dir_list:
    with open(DIRECTORY + DATASET_NAME + ".data", "r") as in_f, open(
        DIRECTORY + DATASET_NAME + FORMAT_END + ".data", "w+"
    ) as out_f:
        # remove all dolar sign from each line
        lines = map(lambda x: x.replace("$", ""), in_f.readlines())
        out_f.writelines(lines)
