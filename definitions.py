from os import getcwd, pardir, makedirs, path
from os import path
from os.path import join, abspath, exists

ROOT_DIR = path.dirname(path.abspath(__file__)) # This is your Project Root

DATA_DIR = join(ROOT_DIR, 'data')
TALOS_DIR = join(DATA_DIR, 'talos_logs')
MODELS_DIR = join(ROOT_DIR, 'models')

# if the folders don't exist, create them.
if not exists(DATA_DIR):
    makedirs(DATA_DIR)

if not exists(MODELS_DIR):
    makedirs(MODELS_DIR)