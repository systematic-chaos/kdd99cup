import os
import urllib.request
import gzip

BASE_URL = "http://kdd.ics.uci.edu/databases/kddcup99/"
DATA = "data/".upper()
TRAIN_FILE = "kddcup.data"
TEST_FILE = "corrected"
GUNZIP = ".gz"

if not(os.path.exists(DATA) and os.path.isdir(DATA)):
    os.mkdir(DATA, 0o770)

# Download data
ftrain = urllib.request.urlretrieve(BASE_URL + TRAIN_FILE + GUNZIP, DATA + TRAIN_FILE + GUNZIP)
ftest = urllib.request.urlretrieve(BASE_URL + TEST_FILE + GUNZIP, DATA + TEST_FILE + GUNZIP)

# Decompress and take the first lines for the training and test datasets
NLINES_TRAIN = 10000
with gzip.open(DATA + TRAIN_FILE + GUNZIP, 'rb') as f_in,\
        gzip.open(DATA + TRAIN_FILE + '.' + str(NLINES_TRAIN) + GUNZIP, 'wb') as f_out:
    f_out.writelines(next(f_in) for n in range(NLINES_TRAIN))

NLINES_TEST = int(NLINES_TRAIN / 10)
with gzip.open(DATA + TEST_FILE + GUNZIP, 'rb') as f_in,\
        gzip.open(DATA + TEST_FILE + '.' + str(NLINES_TEST) + GUNZIP, 'wb') as f_out:
    f_out.writelines(next(f_in) for n in range(NLINES_TEST))
