from _csv import reader

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import os
import csv

from tigramite import data_processing as pp
from tigramite import plotting as tp
from tigramite.lpcmci import LPCMCI
from tigramite.pcmci import PCMCI
from tigramite.independence_tests import ParCorr, GPDC, CMIknn

from DataLoader import new_loader
from Evaluation import evaluate, remove_suffix


def data_processing(file, nb_points):
    data = new_loader(file)
    data = np.swapaxes(data, 0, 1)
    data = data[0:nb_points]
    var_names = [r'$X^{%d}$' % j for j in range(len(data[0]))]
    dataframe = pp.DataFrame(data, var_names=var_names)

    return dataframe, var_names


def determine_tau_max(var_names, dataframe, indeptest, method="get_lagged_dependencies"):
    # Search for the optimal max lag value in the time series with a lagged dependencies plot or run autocorrelation (run_bivci, better on large datasets)
    # the tau max value should be the maximal tau value where a dependency graph has its maximum absolute value
    pcmci = PCMCI(dataframe=dataframe,
                  cond_ind_test=indeptest,
                  verbosity=0)

    if method == "get_lagged_dependencies":
        correlations = pcmci.get_lagged_dependencies(tau_max=10, val_only=True)['val_matrix']
    elif method == "run_bivci":
        correlations = pcmci.run_bivci(tau_max=10, val_only=True)['val_matrix']
    else:
        raise ValueError("Unknown method")

    lag_func_matrix = tp.plot_lagfuncs(val_matrix=correlations, setup_args={'var_names': var_names,
                                                                            'x_base': 5, 'y_base': .5})
    plt.show()


def runPCMCI(pcmci, tau_max, tau_min=0, pc_alpha=None):
    # pc_alpha not used: the algorithm selects the parameter from [0.001, 0.005, 0.01, 0.025, 0.05] through an optimisation step

    results = pcmci.run_pcmciplus(tau_min=tau_min, tau_max=tau_max, pc_alpha=pc_alpha)

    '''
    tp.plot_graph(
        graph=results['graph'],
        var_names=var_names,
    )
    plt.show()
    '''

    return results['graph']


def graph2csv(tau_max, graph, file, indeptest):
    list2csv = []
    for i in range(len(graph)):
        for j in range(len(graph[i])):
            for k in range(len(graph[i, j])):
                if graph[i, j, k] == '-->':
                    list2csv.append(f"{i},{j},{k}")
                elif graph[i, j, k] == '<--':
                    list2csv.append(f"{j},{i},{k}")
                elif graph[i, j, k] == 'o-o':
                    list2csv.append(f"{i}o-o{j},{k}")
                elif graph[i, j, k] == 'x-x':
                    list2csv.append(f"{i}x-x{j},{k}")
                elif graph[i, j, k] == 'o->':
                    list2csv.append(f"{i}o->{j},{k}")
                elif graph[i, j, k] == '<->':
                    list2csv.append(f"{i},{j},{k}")
                    list2csv.append(f"{j},{i},{k}")
                else:
                    if graph[i, j, k] != '':
                        print(graph[i, j, k])
                        raise ValueError('Unknown connection')
    print(list2csv)
    try:
        with open(f"Results/LPCMCI_Results_{architecture}/{indeptest}/tau_max={tau_max}_{file}", 'w',
                  newline='') as csvfile:
            writer = csv.writer(csvfile)
            for data in list2csv:
                writer.writerow([data])
    except IOError:
        print("I/O error")

    return list2csv


def latentPCMCI(architecture, dir, indeptest, gt_file, nb_points, pc_alpha=0.01, tau_min=0):
    if indeptest == "parcorr":
        test = ParCorr(significance='analytic')
    elif indeptest == "gpdc":
        test = GPDC(significance='analytic', gp_params=None)
    elif indeptest == "cmi_knn":
        test = CMIknn(significance='shuffle_test', knn=0.1, shuffle_neighbors=5, transform='ranks')
    try:
        os.mkdir(f"Results/LPCMCI_Results_{architecture}")
        os.mkdir(f"Results/LPCMCI_Results_{architecture}/{indeptest}")
    except:
        print("File already existing")

    tocsv = [["TPR", "FPR", "Precision on the edges", "Recall", "(F1)→ score", "Precision on the predicted lags",
              "Frobenius Norm", "MSE", "Structural Hamming distance"]]
    for file in dir:
        print(file)
        if file.endswith('groundtruth.csv') is False:
            dataframe, var_names = data_processing(f"ProcessedData/{architecture}/{file}", nb_points=nb_points)
            lpcmci = LPCMCI(
                dataframe=dataframe,
                cond_ind_test=test,
                verbosity=0)

            determine_tau_max(var_names, dataframe, test)
            print("Input tau_max value for this dataset: ")
            tau_max = int(input())

            graph = lpcmci.run_lpcmci(tau_min=tau_min, tau_max=tau_max, pc_alpha=pc_alpha)['graph']

            tp.plot_time_series_graph(figsize=(6, 7), graph=graph, var_names=var_names)

            plt.savefig(f"Results/LPCMCI_Results_{architecture}/{indeptest}/{remove_suffix(file, '.csv')}.png")
            plt.close()

            converted_graph = graph2csv(tau_max, graph, file, indeptest)

            if any('o-o' in i for i in converted_graph) or any('x-x' in i for i in converted_graph) or any('o->' in i for i in converted_graph):
                tocsv.append([0 for i in range(8)])
            else:
                print
                gtpddata = pd.read_csv(gt_file, header=None)

                pdgraph = pd.DataFrame(list(reader(converted_graph))).astype(int)

                save = evaluate(gtpddata, pdgraph, nb_var=len(var_names), tau_max=tau_max)

                save = [round(x, 2) for x in save]
                tocsv.append(save)

            saveframe = pd.DataFrame(tocsv)
            saveframe = saveframe.transpose()
            saveframe.to_csv(f"Results/LPCMCI_Results_{architecture}/{indeptest}/Evaluation.csv", index=False,
                             header=False)

    return saveframe


def PCMCI_plus(architecture, dir, indeptest, gt_file, nb_points, tau_min=0, pc_alpha=None):
    if indeptest == "parcorr":
        test = ParCorr(significance='analytic')
    elif indeptest == "gpdc":
        test = GPDC(significance='analytic', gp_params=None)
    elif indeptest == "cmi_knn":
        test = CMIknn(significance='shuffle_test', knn=0.1, shuffle_neighbors=5, transform='ranks')

    try:
        os.mkdir(f"Results/PCMCI+_Results_{architecture}")
        os.mkdir(f"Results/PCMCI+_Results_{architecture}/{indeptest}")
    except:
        print("File already existing")

    tocsv = [["TPR", "FPR", "Precision on the edges", "Recall", "F1 score", "Precision on the predicted lags",
              "Frobenius Norm", "MSE", "Structural Hamming distance"]]

    for file in dir:
        print(file)
        if file.endswith('groundtruth.csv') is False:
            dataframe, var_names = data_processing(f"ProcessedData/{architecture}/{file}", nb_points=nb_points)
            pcmci = PCMCI(
                dataframe=dataframe,
                cond_ind_test=test,
                verbosity=0)

            determine_tau_max(var_names, dataframe,test)
            print("Input tau_max value for this dataset: ")
            tau_max = int(input())

            graph = pcmci.run_pcmciplus(tau_min=tau_min, tau_max=tau_max, pc_alpha=pc_alpha)['graph']

            tp.plot_time_series_graph(figsize=(6, 7), graph=graph, var_names=var_names)

            plt.savefig(f"Results/PCMCI+_Results_{architecture}/{indeptest}/{remove_suffix(file, '.csv')}.png")
            plt.close()

            converted_graph = graph2csv(tau_max, graph, file, indeptest)

            if any('o-o' in i for i in converted_graph) or any('x-x' in i for i in converted_graph) or any('o->' in i for i in converted_graph):
                tocsv.append([0 for i in range(8)])
            else:
                print
                gtpddata = pd.read_csv(gt_file, header=None)

                pdgraph = pd.DataFrame(list(reader(converted_graph))).astype(int)

                save = evaluate(gtpddata, pdgraph, nb_var=len(var_names), tau_max=tau_max)

                save = [round(x, 2) for x in save]
                tocsv.append(save)

    saveframe = pd.DataFrame(tocsv)
    saveframe = saveframe.transpose()
    saveframe.to_csv(f"Results/PCMCI+_Results_{architecture}/{indeptest}/Evaluation.csv", index=False, header=False)

    return saveframe


if __name__ == "__main__":
    architecture = "Fork"

    data_dir = os.listdir(f"ProcessedData/{architecture}")
    for file in data_dir:
        if file.endswith("groundtruth.csv"):
            gt_file = f"ProcessedData/{architecture}/{file}"

    PCMCI_plus(architecture, data_dir, "gpdc", gt_file, nb_points=500)
    #latentPCMCI(architecture, data_dir, "gpdc", gt_file, nb_points=500)
