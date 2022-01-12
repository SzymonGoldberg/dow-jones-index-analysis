DIRECTORY = 'dataset/'
DATASET_NAME = 'dow_jones_index'

import zipfile, os

dir_list = os.listdir(DIRECTORY)
# check if dataset folder have expracted files
if not DATASET_NAME + '.data' in dir_list or not DATASET_NAME + '.names' in dir_list:
    # if not, extract all
    with zipfile.ZipFile(DIRECTORY + DATASET_NAME + '.zip', 'r') as zip:
        zip.extractall(DIRECTORY)

# find paragraph related to names
