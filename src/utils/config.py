"""Central configuration defaults for all experiments."""

#Data 
DATA_DIR = "data/raw/iSarcasmEval"
TRAIN_FILE = "train/train.En.csv"
TEST_FILE = "test/task_A_En_test.csv"    
VAL_SPLIT = 0.15                         
RANDOM_SEED = 42

#Baseline 
TFIDF_MAX_FEATURES = 50_000
TFIDF_NGRAM_RANGE = (1, 2)
SVM_C = 1.0

#Transformer 
MAX_LENGTH = 128
BATCH_SIZE = 32
EPOCHS = 3
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1

#Outputs
OUTPUT_DIR = "outputs"
RESULTS_CSV = "outputs/results.csv"
