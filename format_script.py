import pandas as pd
import argparse
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument("--input_file")
parser.add_argument("--config", type=int)
parser.add_argument("--id")

args = parser.parse_args()

INPUT_FILE = args.input_file
CONFIG = args.config
ID = args.id

def read_file(filename, start, train):

    line_list = []
    loss = []
    top1 = []
    top5 = []

    with open(filename, "r") as f:
        for line in f.readlines():
            if line.startswith(start):
                line_list.append(line)

    if train:
        for line in line_list:
            l, t1, t5, tt, te = line.split(",")
            temp1, temp2 = l.split(":")
            loss.append(float(temp2.strip()))
            temp1, temp2 = t1.split(":")
            top1.append(float(temp2.strip()))
            temp1, temp2 = t5.split(":")
            top5.append(float(temp2.strip()))

    else:
        for line in line_list:
            l, t1, t5 = line.split(",")
            temp1, temp2 = l.split(":")
            loss.append(float(temp2.strip()))
            temp1, temp2 = t1.split(":")
            top1.append(float(temp2.strip()))
            temp1, temp2 = t5.split(":")
            top5.append(float(temp2.strip()))

    return loss, top1, top5


vloss, vtop1, vtop5 = read_file(INPUT_FILE, "Validation Loss:", False)
tloss, ttop1, ttop5 = read_file(INPUT_FILE, "Epoch Loss:", True)

df = pd.DataFrame({
    "tloss": tloss,
    "ttop1": ttop1,
    "ttop5": ttop5,
    "vloss": vloss,
    "vtop1": vtop1,
    "vtop5": vtop5
})


vtop1 = df[["vtop1"]].to_numpy()
vtop5 = df[["vtop5"]].to_numpy()

ttop1 = df[["ttop1"]].to_numpy()
ttop5 = df[["ttop5"]].to_numpy()

top1 = np.abs(ttop1 - vtop1) * 100
top5 = np.abs(ttop5 - vtop5) * 100

top1[:] = np.trunc(top1 * 1e4) / 1e4
top5[:] = np.trunc(top5 * 1e4) / 1e4



df["top1_diff"] = top1
df["top5_diff"] = top5


df.to_csv(f"csv_stats/config_{CONFIG}_id_{ID}.csv")
