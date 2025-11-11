import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import argparse
from supervoxel import Supervoxel
from data import WindowTransform, GetData, VideoData
from utils import format_time, print_runtime
from torch.utils.data import DataLoader
from time import perf_counter
import os


def compute_avg_videos(video, segs, sv):

    fvid = video.permute(0,2,3,4,1).reshape(-1,3)
    means = sv.scatter_mean_2d(fvid, segs.view(-1))
    sv_vid = means[segs]

    sv_vid = sv_vid.permute(0,4,1,2,3)

    n = sv_vid.flatten().shape[0]

    sse = torch.sum(video - sv_vid)**2

    mse = sse / n
    mean = torch.sum(sv_vid) / n

    return mse, mean, segs.max() + 1

def main(batch_size, space_patch, time_patch, space_lvl, time_lvl, DATASET, sv, filename, device):

    #device = "cuda"


    path = "/cluster/work/projects/ec35/ec-philipth/"

    num_frames = 0
    if DATASET == 1:
        dataset = "UCF_less_f_less_r"
        num_frames = 20
    elif DATASET == 2:
        dataset = "UCF_less_f"
        num_frames = 20
    elif DATASET == 3:
        dataset = "UCF_less_r"
        num_frames = 50
    else:
        raise Exception("No dataset chosen")

    print(f"Path to data: {path}/{dataset}")

    v_data = GetData(path, csv="val")
    X_val, y_val = v_data.get_dataset()
    val_dataset = VideoData(X_val, y_val, os.path.join(path, dataset), v_data.classes, num_frames=num_frames)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=16,
        drop_last=True,
        prefetch_factor=5,
        pin_memory=True
    )

    supervoxel = Supervoxel(
            device=device, time_patch=time_patch, space_patch=space_patch)

    mse_ave = []
    mean_ave = []
    num_sv_ave = []
    edges_ave = []

    for maxlvl in range(1,2):

        print(f"Running for maxlvl {maxlvl}")

        mse_list = []
        mean_list = []
        num_sv_list = []
        num_of_edges = []

        for i, (vid, label) in enumerate(val_loader):

            if i % 100 == 0:
                print(f"batch number: {i}")

            start_time = perf_counter()
            segs, edges, features, seg_indices = supervoxel.process(
                vid = vid.to(device),
                maxlvl=6,
                sv = sv,
                space_patch=space_patch,
                time_patch=time_patch,
                space_lvl = space_lvl,
                time_lvl = time_lvl
            )

            end_time = perf_counter()

            print(f"Time: {end_time - start_time}")

            vid = vid.to("cpu")
            segs = segs.to("cpu")
            mse, mean, num_sv = compute_avg_videos(vid, segs, supervoxel)
            num_edges = edges.shape[1]

            assert isinstance(mse.item(), float), f"type: {type(mse.item())}"
            assert isinstance(mean.item(), float), f"type: {type(mean.item())}"
            assert isinstance(num_sv.item(), (float, int)), f"type: {type(num_sv.item())}"
            assert isinstance(num_edges, (float, int)), f"type: {type(num_edges)}"

            mse_list.append(mse.item())
            mean_list.append(mean.item())
            num_sv_list.append(num_sv.item())
            num_of_edges.append(num_edges)

            del vid
            del segs

        mse_average = np.sum(mse_list)/len(mse_list)
        mean_average = np.sum(mean_list)/len(mean_list)
        num_sv_average = np.sum(num_sv_list)/len(num_sv_list)
        edges_average = np.sum(num_of_edges)/len(num_of_edges)

        print(f"mse: {mse_average}")
        print(f"mean: {mean_average}")
        print(f"num_sv: {num_sv_average}")
        print(f"num_edges: {edges_average}")

        mse_ave.append(mse_average)
        mean_ave.append(mean_average)
        num_sv_ave.append(num_sv_average)
        edges_ave.append(edges_average)

    df = pd.DataFrame({
        "mse_ave": mse_ave,
        "mean_ave": mean_ave,
        "num_sv_ave": num_sv_ave,
        "edges_ave": edges_ave,
    })

    df.to_csv(f"{filename}.csv", index=False)
            


if __name__ == "__main__":
  
    parser = argparse.ArgumentParser(
        description="Statistics of the supervoxel algorithm")

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--space_patch", type=int, default=10)
    parser.add_argument("--time_patch", type=int, default=3)
    parser.add_argument("--space_lvl", type=int, default=4)
    parser.add_argument("--time_lvl", type=int, default=3)
    parser.add_argument("--sv", type=str, default="orig")
    parser.add_argument("--dataset", type=int, default=1)
    parser.add_argument("--filename", type=str)
    parser.add_argument("--device", default="cuda")

    args = parser.parse_args()

    BATCH_SIZE = args.batch_size
    SPACE_PATCH = args.space_patch
    TIME_PATCH = args.time_patch
    SPACE_LVL = args.space_lvl
    TIME_LVL = args.time_lvl
    SV = args.sv
    DATASET = args.dataset
    FILENAME = args.filename
    DEVICE = args.device


    main(BATCH_SIZE, SPACE_PATCH, TIME_PATCH, SPACE_LVL, TIME_LVL, DATASET, SV, FILENAME, DEVICE)


    # main()
