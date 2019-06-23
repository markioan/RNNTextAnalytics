import sys
from os import pardir, getcwd
from os.path import join, abspath

PARENT_DIRECTORY = abspath(join(getcwd(), pardir))
sys.path.insert(0, PARENT_DIRECTORY)