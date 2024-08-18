import pickle
from pathlib import Path
import os

path=os.path.join(Path(os.getcwd()),"046211_modern_optimizers_for_pretrained_models/models/imagenet/trail_one/final_results.pickle")
# print(path)

with open(path, 'rb') as handle:
    final_results = pickle.load(handle)