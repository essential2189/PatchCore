import click
import time
import datetime
from data import LoadDataset
from models import PatchCore
from utils import print_and_export_results
from torch.utils.data import Dataset, DataLoader
from typing import List
import sys
from tqdm import tqdm
import datetime
# seeds
import torch
import torch.nn as nn
import random
import numpy as np
from sklearn.cluster import KMeans

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

import warnings # for some torch warnings regarding depreciation
warnings.filterwarnings("ignore")

ALLOWED_METHODS = ["patchcore"]

# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'



def run_model(method: str, classes: List):
    results = {}

    indices = ((2, 3), (3, 4), (4,))
    name = ['efficientnet_l2', 'wide_resnet50_2', 'resnet50', 'resnet101', 'inception_resnet_v2', 'inception_v4']
    sizes = [224, 256, 299]

    outfile = open('scoreboard_path', 'w')
    for cls in classes:
        for n in name:
            for idx in indices:
                for size in sizes:
                    model = PatchCore(
                        f_coreset=0.1,  # fraction the number of training samples
                        coreset_eps=0.9,  # sparse projection parameter
                        backbone_name=n,
                        out_indices=idx,
                    )
                    data_name = cls.split('/')[-2]

                    print(f"\n█│ Running {method} on {data_name} dataset.")
                    print(  f" ╰{'─'*(len(method)+len(data_name)+23)}\n")
                    train_ds, test_ds = LoadDataset(cls, size=size).get_dataloaders()

                    start = time.time()
                    print("   Training ...")
                    model.fit(train_ds)
                    print("   Testing ...")
                    image_rocauc = model.evaluate(test_ds)
                    end = time.time()

                    print(f"\n   ╭{'─'*(len(data_name)+15)}┬{'─'*20}┬{'─'*20}╮")
                    print(f"   │ Test results {data_name} │ image_rocauc: {image_rocauc:.5f}")
                    print(  f"   ╰{'─'*(len(data_name)+15)}┴{'─'*20}┴{'─'*20}╯")

                    run_time = datetime.timedelta(seconds=end - start)

                    outfile.write('model: {}, idx: {}, rocaur: {}, run time: {}\n'.format(n, idx, float(image_rocauc), run_time))

                    results[n, idx] = [float(image_rocauc)]


    image_results = [v[0] for _, v in results.items()]
    average_image_roc_auc = sum(image_results)/len(image_results)


    total_results = {
        "per_class_results": results,
        "average image rocauc": average_image_roc_auc,
        "model parameters": model.get_parameters(),
    }
    return total_results

@click.command()
@click.argument("method")
@click.option("--dataset", default="all", help="Dataset, defaults to all datasets.")
def cli_interface(method: str, dataset: str): 
    if dataset == "all":
        dataset = ALL_CLASSES
    else:
        dataset = [dataset]

    method = method.lower()
    assert method in ALLOWED_METHODS, f"Select from {ALLOWED_METHODS}."

    total_results = run_model(method, dataset)

    print_and_export_results(total_results)
    
if __name__ == "__main__":
    cli_interface()
