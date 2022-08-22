import os

import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def remove_suffix(input_string, suffix):
    if suffix and input_string.endswith(suffix):
        return input_string[:-len(suffix)]
    return input_string


def evaluate(gtfile, predictedfile, nb_var, tau_max):
    """Evaluates the results by comparing it to the ground truth graph, and calculating precision, recall and F1-score"""

    readgt, delays = getdelays(gtfile, nb_var)
    readpred, preddelays = getdelays(predictedfile, nb_var)

    preddelays = manage_unoriented_edges(readgt, delays, readpred, preddelays)

    gt_matrix = weighted_adjacency_matrix(delays, nb_var)
    pred_matrix = weighted_adjacency_matrix(preddelays, nb_var)

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

    percentagecorrect = evaluatedelay(delays, preddelays, TPsdirect, tau_max)
    print("Percentage of delays that are correctly discovered: ", percentagecorrect)

    print("Frobenius Norm: ", frob)
    print("Mean Squarred error: ", mse)

    # plotgraph(delays, columns)
    # plt.figure("Predicted Graph")
    # plotgraph(preddelays, columns)
    # plt.show()

    return [tpr, fpr, precision, recall, F1direct, percentagecorrect, frob, mse]


def getdelays(pddata, nb_var):
    """Collects the total delay of indirect causal relationships."""
    columns = [i for i in range(nb_var)]

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
        if (key, value) in pairdelays:
            pairdelays[(key, value)] = min(delays[i], pairdelays[(key, value)])
        else:
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


def plotgraph(alldelays, nb_var, edgecolor, selfcauses):
    columns = [i for i in range(nb_var)]
    """Plots a temporal causal graph showing all discovered causal relationships annotated with the time delay between cause and effect."""
    G = nx.DiGraph()
    for c in columns:
        G.add_node(c)
    for pair in alldelays:
        p1, p2 = pair
        if selfcauses:
            nodepair = (columns[p2], columns[p1])
            G.add_edges_from([nodepair], weight=alldelays[pair], length=300)
        else:
            if p1 != p2:
                nodepair = (columns[p2], columns[p1])
                G.add_edges_from([nodepair], weight=alldelays[pair], length=300)

    edge_labels = dict([((u, v,), d['weight'])
                        for u, v, d in G.edges(data=True)])

    pos = nx.circular_layout(G)
    # nx.draw_networkx_edges(G, pos, edge_color=edgecolor, connectionstyle='arc3,rad=0.3')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, label_pos=0.22)

    nx.draw(G, pos, node_color='white', edge_color=edgecolor, node_size=500, with_labels=True)
    ax = plt.gca()
    ax.collections[0].set_edgecolor("#000000")

    return G


def weighted_adjacency_matrix(dictionary, nb_var):
    adj_matrix = np.zeros((nb_var, nb_var))
    for pair in dictionary:
        if dictionary.get(pair) == 0:
            adj_matrix[pair[1], pair[0]] = 0.5
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


def compare_graphs(graph1, graph1name, graph2, graph2name, nb_var, cmp_to_gt=False, selfcauses=True):
    readg2, delaysgraph1 = getdelays(graph1, nb_var)
    readg1, delaysgraph2 = getdelays(graph2, nb_var)

    if cmp_to_gt:
        # When comparing a graph with its gt, g1 is the gt and g2 is the predicted graph
        delaysgraph2 = manage_unoriented_edges(readg1, delaysgraph1, readg2, delaysgraph2)

    commonedges = dict()
    for i in delaysgraph1:
        if i in delaysgraph2:
            if delaysgraph1[i] == delaysgraph2[i]:
                commonedges[i] = delaysgraph1[i]

            else:
                commonedges[i] = f"{delaysgraph1[i]},{delaysgraph2[i]}"

    for i in commonedges:
        del delaysgraph1[i]
        del delaysgraph2[i]

    plotgraph(commonedges, nb_var, 'black', selfcauses)
    plotgraph(delaysgraph1, nb_var, 'red', selfcauses)
    plotgraph(delaysgraph2, nb_var, 'blue', selfcauses)

    commonpatch = mpatches.Patch(color='black', label='Common edges')
    g1patch = mpatches.Patch(color='red', label=graph1name)
    g2patch = mpatches.Patch(color='blue', label=graph2name)
    plt.legend(handles=[commonpatch, g1patch, g2patch])


def manage_unoriented_edges(readgt, gtdelays, readpred, preddelays):
    supress_edges = []
    for i in preddelays:
        if (i[1], i[0]) in preddelays and i[0] != i[1]:
            if (i in gtdelays) and ((i[1], i[0]) not in gtdelays):
                supress_edges.append(i)
            elif ((i[1], i[0]) in gtdelays) and (i not in gtdelays):
                supress_edges.append((i[1], i[0]))

    supress_edges = list(set(supress_edges))
    for i in supress_edges:
        del preddelays[i]
        readpred[i[0]].remove(i[1])

    return preddelays


if __name__ == "__main__":
    #TCDF Evaluation

    folder="tunningts=1000"
    architecture_list = ["Fork", "Mediator", "Vstructure", "Diamond", "7TS", "7ts2h"]
    nb_of_variables_list = [3, 3, 3, 4, 7, 7]
    dir = os.listdir(
        f"C:/Users/flori/Downloads/causal_discovery_for_time_series-master/causal_discovery_for_time_series-master/baselines/scripts_python/python_packages/TCDF-master/TCDF-Master/{folder}")
    tocsv = [
        ["filename", "TPR", "FPR", "Precision on the edges", "Recall", "F1 score",
         "Precision on the predicted lags",
         "Frobenius Norm", "MSE", "SID"]]
    for file in dir:
        if file.endswith('.csv'):
            for arch in architecture_list:
                if remove_suffix(file,f"{remove_suffix(file,'.csv')[-1]}.csv")==arch:
                    architecture=arch
            nb_of_variables = nb_of_variables_list[architecture_list.index(architecture)]
            for gtfile in os.listdir(f"Results/groundtruth"):
                if gtfile.endswith(f"{architecture}_groundtruth.csv"):
                    gt_file = gtfile

            print('Evaluation of file : ', file)
            tau_max = 16
            # print('tau_max = ', tau_max)
            gtpddata = pd.read_csv(f"Results/groundtruth/{gt_file}", header=None)
            try:
                predfile = pd.read_csv(
                    f"C:/Users/flori/Downloads/causal_discovery_for_time_series-master/causal_discovery_for_time_series-master/baselines/scripts_python/python_packages/TCDF-master/TCDF-Master/{folder}/{file}",
                    header=None)
                save = evaluate(gtpddata,
                                predfile,
                                nb_var=nb_of_variables, tau_max=tau_max)
                save = [round(x, 2) for x in save]
                save.insert(0, remove_suffix(file, '.csv'))
                tocsv.append(save)

                compare_graphs(gtpddata, "Ground truth graph", predfile,
                               remove_suffix(file.rpartition('_')[2], '.csv'),
                               nb_of_variables, cmp_to_gt=True)

                #plt.savefig(
                #    f"Results/tau_max=1\PCMCI+_1000points/PCMCI+_Results_{architecture}/gpdc/{remove_suffix(file.rpartition('_')[2], '.csv')}.png")

                #plt.close()


            except:
                print("No predictions or decoding error for file", file)

    saveframe = pd.DataFrame(tocsv)
    saveframe = saveframe.transpose()

    saveframe.to_csv(f"C:/Users/flori/Downloads/causal_discovery_for_time_series-master/causal_discovery_for_time_series-master/baselines/scripts_python/python_packages/TCDF-master/TCDF-Master/{folder}/Evaluation.csv",
                         index=False,
                         header=False)


    '''
    #PC evaluation
    architecture_list = ["Fork", "Mediator", "Vstructure", "Diamond", "7TS", "7TS2H"]
    nb_of_variables_list = [3, 3, 3, 4, 7, 7]

    for i in range(len(architecture_list)):
        architecture = architecture_list[i]
        nb_of_variables = nb_of_variables_list[i]
        for file in os.listdir(f"Results/groundtruth"):
            if file.endswith(f"{architecture}_groundtruth.csv"):
                gt_file = file

        dir = os.listdir(f"Results/tau_max=1/PCMCI+_1000points/PCMCI+_Results_{architecture}/gpdc")

        tocsv = [
            ["filename", "TPR", "FPR", "Precision on the edges", "Recall", "F1 score",
             "Precision on the predicted lags",
             "Frobenius Norm", "MSE", "SID"]]

        for file in dir:
            if file.endswith('.csv'):
                print('Evaluation of file : ', file)
                tau_max = int(file.rpartition('=')[2].rpartition('_')[0])
                # print('tau_max = ', tau_max)
                gtpddata = pd.read_csv(f"Results/groundtruth/{gt_file}", header=None)
                try:
                    predfile = pd.read_csv(f"Results/tau_max=1/PCMCI+_1000points/PCMCI+_Results_{architecture}\gpdc/{file}",
                                           header=None)
                    save = evaluate(gtpddata,
                                    predfile,
                                    nb_var=nb_of_variables, tau_max=tau_max)
                    save = [round(x, 2) for x in save]
                    save.insert(0, remove_suffix(file.rpartition('_')[2], '.csv'))
                    tocsv.append(save)

                    compare_graphs(gtpddata, "Ground truth graph", predfile,
                                   remove_suffix(file.rpartition('_')[2], '.csv'),
                                   nb_of_variables, cmp_to_gt=True)

                    plt.savefig(
                        f"Results/tau_max=1\PCMCI+_1000points/PCMCI+_Results_{architecture}/gpdc/{remove_suffix(file.rpartition('_')[2], '.csv')}.png")

                    plt.close()


                except:
                    print("No predictions or decoding error for file", file)

        saveframe = pd.DataFrame(tocsv)
        saveframe = saveframe.transpose()

        saveframe.to_csv(f"Results/tau_max=1\PCMCI+_1000points/PCMCI+_Results_{architecture}/Evaluation.csv", index=False,
                         header=False)


    #gtpddata = pd.read_csv(f"Results/groundtruth/7TS2H_groundtruth.csv", header=None)
    predfile = pd.read_csv("C:/Users/flori/Desktop/Stage_2A/Code/Results/PCMCI+_Results_cereal_data/gpdc/tau_max=10_cereal_database_linear.csv",
                           header=None)
    nb_var = 112

    #readgt, delays = getdelays(gtpddata, nb_var)
    readpred, preddelays = getdelays(predfile, nb_var)

    plotgraph(preddelays, nb_var, 'black', selfcauses=True)
    # plt.show()
    # plt.figure(figsize=(10,20))
    #compare_graphs(gtpddata, "Ground truth graph", predfile, "7TS2H 3", nb_var,cmp_to_gt=True, selfcauses=False)
    plt.show()

    '''

    '''
    # Experiment with TCDF Filter
    datapd = pd.read_csv("diff_cereal_database_linear.csv")
    predfile = pd.read_csv(
        "C:/Users/flori/Desktop/Stage_2A/Code/Results/LPCMCI_Results_cereal_data/gpdc/TCDF_linear_diff.csv",
        header=None)
    nb_var = 112
    var_names = list(pd.read_csv("diff_cereal_database_linear.csv").columns)

    filter=[]
    for i in predfile.index:
        line = predfile.iloc[i]
        cause = var_names[line[0]]
        consequence = var_names[line[1]]
        lag = line[2]
        print(f"{cause} causes {consequence} with a time lag of {lag}")
        filter.append(cause)
        filter.append(consequence)
        #plt.plot(datapd.iloc[:, line[0]])
        #plt.title(f"Cause {cause}")
        #plt.figure()
        #plt.plot(datapd.iloc[:, line[1]])
        #plt.title(f"Consequence {consequence}")
        #plt.show()
    filter = list(set(filter))
    print(filter)
    print(len(filter))

    datatcdf = pd.read_csv(
        "C:/Users/flori/Desktop/Stage_2A/Code/Results/LPCMCI_Results_cereal_data/gpdc/TCDF_linear_diff.csv",
        header=None)
    for index, row in datatcdf.iterrows():
        print(datapd.columns[row[0]],"causes",datapd.columns[row[1]],"with a time lag of",row[2] )
    a=0

    datapd= datapd[filter]
    print(datapd.columns)



    data=pd.read_csv(
        "C:/Users/flori/Desktop/Stage_2A/Code/Results/LPCMCI_Results_cereal_data/gpdc/tau_max=10_LPCMCI_TCDFfilter.csv",
        header=None)
    print(data.index)
    for index, row in data.iterrows():
        print(datapd.columns[row[0]],"causes",datapd.columns[row[1]],"with a time lag of",row[2] )
    a=0
    '''



    '''
    filter_countries = ["poland", "spain", "finland", "sweden", "lithuania", "latvia", "denmark", "estonia", "austria",
                        "croatia", "slovakia", "bulgaria", "belgium", "luxembourg", "netherlands", "cyprus"]
    list_countries=[]
    list_var=[]
    for i in datapd.columns:
        country = i.split(" ")[1]
        if country not in filter_countries:
            datapd = datapd.drop(i, axis=1)
        else:
            list_countries.append(country)
            list_var.append(i)
    print(set(list_countries))
    print(list_var)
    '''

    #datapd.to_csv("cereal_database_TCDFfilter.csv", index=False)

    # readpred, preddelays = getdelays(predfile, nb_var)
    # plotgraph(preddelays, nb_var, 'black', selfcauses=True)
    # plt.show()
