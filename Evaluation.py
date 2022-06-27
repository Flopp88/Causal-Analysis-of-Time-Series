import os

import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def evaluate(gtfile, predictedfile, nb_var, tau_max):
    columns = [i for i in range(nb_var)]

    """Evaluates the results by comparing it to the ground truth graph, and calculating precision, recall and F1-score"""

    readgt, delays = getdelays(gtfile, columns)
    readpred, preddelays = getdelays(predictedfile, columns)

    gt_matrix = adjacency_matrix(delays, nb_var)
    pred_matrix = adjacency_matrix(preddelays, nb_var)

    frob = Frobenius(gt_matrix, pred_matrix)
    mse = MSE(gt_matrix, pred_matrix)

    FPdirect = 0
    TPdirect = 0
    FN = 0
    FPsdirect = []
    TPsdirect = []
    FNs = []

    for key in readgt:
        for v in readpred[key]:
            if v not in readgt[key]:
                FPdirect += 1
                FPsdirect.append((key, v))
            else:
                TPdirect += 1
                TPsdirect.append((key, v))
        for v in readgt[key]:
            if v not in readpred[key]:
                FN += 1
                FNs.append((key, v))

    print("Total Direct False Positives: ", FPdirect)
    print("Total Direct True Positives: ", TPdirect)
    print("Total Direct False Negatives: ", FN)
    print("TPs direct: ", TPsdirect)
    print("FPs direct: ", FPsdirect)
    print("FNs: ", FNs)

    tpr = TPR(TPdirect, delays)
    fpr = FPR(FPdirect, delays, nb_var)

    print("True Positive Rate: ", tpr)
    print("False Positive Rate: ", fpr)

    precision = recall = 0.
    if float(TPdirect + FPdirect) > 0:
        precision = TPdirect / float(TPdirect + FPdirect)
    print("Precision: ", precision)
    if float(TPdirect + FN) > 0:
        recall = TPdirect / float(TPdirect + FN)
    print("Recall: ", recall)
    if (precision + recall) > 0:
        F1direct = 2 * (precision * recall) / (precision + recall)
    else:
        F1direct = 0.
    print("F1 score: ", F1direct, "(includes only direct causal relationships)")

    percentagecorrect = evaluatedelay(delays, preddelays, TPsdirect, tau_max) * 100
    print("Percentage of delays that are correctly discovered: ", percentagecorrect, "%")

    print("Frobenius Norm: ", frob)
    print("Mean Squarred error: ", mse)

    # plotgraph(delays, columns)
    # plt.figure("Predicted Graph")
    # plotgraph(preddelays, columns)
    # plt.show()

    return [tpr, fpr, precision, recall, F1direct, percentagecorrect, frob, mse]


def getdelays(pddata, columns):
    """Collects the total delay of indirect causal relationships."""

    readgt = dict()
    effects = pddata[1]
    causes = pddata[0]
    delays = pddata[2]
    gtnrrelations = 0
    pairdelays = dict()
    for k in range(len(columns)):
        readgt[k] = []
    for i in range(len(effects)):
        key = effects[i]
        value = causes[i]
        if value not in readgt[key]:
            readgt[key].append(value)
        pairdelays[(key, value)] = delays[i]
        gtnrrelations += 1

    g = nx.DiGraph()
    g.add_nodes_from(readgt.keys())
    for e in readgt:
        cs = readgt[e]
        for c in cs:
            g.add_edge(c, e)

    return readgt, pairdelays


def evaluatedelay(gtdelays, alldelays, TPs, receptivefield):
    """Evaluates the delay discovery of TCDF by comparing the discovered time delays with the ground truth."""
    zeros = 0
    total = 0.
    for i in range(len(TPs)):
        tp = TPs[i]
        discovereddelay = alldelays[tp]
        groundtruth_delays = gtdelays[tp]
        for d in [groundtruth_delays]:
            if d <= receptivefield:
                total += 1.
                error = d - discovereddelay
                if error == 0:
                    zeros += 1

            else:
                next

    if zeros == 0:
        return 0.
    else:
        return zeros / float(total)


def plotgraph(alldelays, columns):
    """Plots a temporal causal graph showing all discovered causal relationships annotated with the time delay between cause and effect."""
    G = nx.DiGraph()
    for c in columns:
        G.add_node(c)
    for pair in alldelays:
        p1, p2 = pair
        nodepair = (columns[p2], columns[p1])

        G.add_edges_from([nodepair], weight=alldelays[pair])

    edge_labels = dict([((u, v,), d['weight'])
                        for u, v, d in G.edges(data=True)])

    pos = nx.circular_layout(G)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    nx.draw(G, pos, node_color='white', edge_color='black', node_size=1000, with_labels=True)
    ax = plt.gca()
    ax.collections[0].set_edgecolor("#000000")

    return G


def adjacency_matrix(dictionary, nb_var):
    adj_matrix = np.zeros((nb_var, nb_var))
    for pair in dictionary:
        if dictionary.get(pair) == 0:
            adj_matrix[pair[1], pair[0]] = -1
        else:
            adj_matrix[pair[1], pair[0]] = dictionary.get(pair)
    return adj_matrix


def Frobenius(true_matrix, predicted_matrix):
    return np.sqrt(np.trace(np.matmul(np.transpose(true_matrix - predicted_matrix), true_matrix - predicted_matrix)))


def MSE(true_matrix, predicted_matrix):
    return np.sum(np.square(true_matrix - predicted_matrix)) / (len(true_matrix[0]))


def FPR(FP, gt_delays, nb_var):
    nb_gt_delays = len(gt_delays)
    return FP / (nb_var * nb_var - nb_gt_delays)


def TPR(TP, gt_delays):
    return TP / len(gt_delays)


def SaveResults():
    return


if __name__ == "__main__":

    architecture = "Fork"
    nb_of_variables = 3

    for file in os.listdir(f"Results/PCMCI+_Results_{architecture}"):
        if file.endswith("groundtruth.csv"):
            gt_file = file

    dir = os.listdir(f"Results/PCMCI+_Results_{architecture}/gpdc")
    tocsv = []
    for file in dir:
        if file.endswith('.csv'):
            print('Evaluation of file : ', file)
            tau_max = int(file.rpartition('=')[2].rpartition('_')[0])
            print('tau_max = ', tau_max)
            gtpddata = pd.read_csv(f"Results/PCMCI+_Results_{architecture}/{gt_file}", header=None)
            predfile = pd.read_csv(f"Results/PCMCI+_Results_{architecture}/gpdc/{file}", header=None)
            save = evaluate(gtpddata,
                            predfile,
                            nb_var=nb_of_variables, tau_max=tau_max)

            tocsv.append(save)
    saveframe = pd.DataFrame(tocsv)
    saveframe=saveframe.transpose()
    saveframe.to_csv("test.csv", index=False, header=False)
