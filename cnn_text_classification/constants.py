NUM_TESTS         = 2000
NUM_CLASSES       = 6
NUM_EPOCHS        = 10
NUM_MINI_BATCH    = 64
EMBEDDING_SIZE    = 128
NUM_FILTERS       = 128
FILTER_SIZES      = [ 3, 4, 5 ]
L2_LAMBDA         = 0.0001
EVALUATE_EVERY    = 100
CHECKPOINTS_EVERY = 1000

SUMMARY_LOG_DIR = 'summary_log'
CHECKPOINTS_DIR = 'checkpoints'

RAW_FILE        = 'data/raw.txt'
DATA_FILE       = 'data/data.npy'
LABEL_FILE      = 'data/labels.npy'
DICTIONARY_FILE = 'data/dictionaries.npy'
