import csv

import numpy as np

datadir = "dataverse_files/"


def preprocess(inputcsv):
    i = 0

    with open(inputcsv, newline='') as csvfile:
        reader = csv.reader(csvfile)

        for row in reader:
            if i == 0:
                data = [[] for t in range(1,len(row))]
                i += 1
            else:
                for t in range(1,len(row)):
                    data[t-1].append(row[t])

    return np.array(data, np.float)

def new_loader(inputcsv):
    i = 0

    with open(inputcsv, newline='') as csvfile:
        reader = csv.reader(csvfile)

        for row in reader:
            if i == 0:
                data = [[] for t in range(len(row))]
                i += 1
            for t in range(len(row)):
                data[t].append(row[t])

    return np.array(data, np.float)
