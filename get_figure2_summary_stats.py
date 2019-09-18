# import DiCE
import warnings
warnings.filterwarnings("ignore")

import os
import sys
import pandas as pd
import numpy as np
import pickle
import argparse
import random
import timeit

import matplotlib.pyplot as pl
import seaborn as sns

def get_figure2_summary_stats(args, postfix_param, lime_discretize_param, lr_dict, cont_radius_dict, summary_metrics, datasets, algorithms, all_total_CFs):

    outdir = 'figures_summary_stats/'
    filename = 'figure2_summary' + '_' + args.postfix +  '_' + args.lime_discretize + '_' + args.cont_radius + '_' + str(int(args.discr_param*100)) + '.data'

    if os.path.exists(os.path.join(outdir, filename)):
        with open(os.path.join(outdir, filename), 'rb') as filehandle:
            results_dict = pickle.load(filehandle)

        print("figure 2 summary already computed..")
        return results_dict

    print('computing figure 2 summary...')
    start_time = timeit.default_timer()

    results_dict = {}
    class_names = {'compass': ['wont_recidivate', 'will_recidivate'], 'adult': ['le50k', 'g50k'],'german': ['good', 'bad'], 'lending': ['Default', 'Paid']}

    for data in datasets:

        # computed in run_DiCE_experiments
        outdir = 'figure1_experiment_results/' + data + '/' + args.model_type + '/'
        filename = data + '_target_cf_classes.data'
        with open(os.path.join(outdir, filename), 'rb') as filehandle:
            target_cf_classes = pickle.load(filehandle)

        # computed in get_figure1_summary_stats.py
        outdir = 'figure1_summary_stats/'
        filename = data + '_nonlinear_valid_unique_list.data'
        with open(os.path.join(outdir, filename), 'rb') as filehandle:
            valid_unique_instances_dict = pickle.load(filehandle)

        outcome_class_dict = {'class0_' + class_names[data][0] : [ix for ix in range(len(target_cf_classes)) if target_cf_classes[ix] == 0.0], 'class1_' + class_names[data][1] : [ix for ix in range(len(target_cf_classes)) if target_cf_classes[ix] == 1.0]}

        for outcome_class in outcome_class_dict:
            results_dict[outcome_class] = {}

            for algorithm in algorithms:

                print(data, '...', algorithm, '...', outcome_class)

                if algorithm == 'NoDiverseCF':
                    diversity_weight = 0.0
                    filealgo = 'DiverseCF'
                    postfix_param1 = 'with_postfix'
                else:
                    diversity_weight = 1.0
                    filealgo = algorithm
                    postfix_param1 = postfix_param

                if algorithm == 'LIME':
                    outdir = 'lime_explanations/' + data + '/figure2_results'
                    filename = 'cont_dist_' + args.cont_radius + '+discrete_perc_' + str(int(args.discr_param*100)) + '.xlsx'

                    df = pd.read_excel(os.path.join(outdir, filename))
                    df = df[df['test_ix'].isin(outcome_class_dict[outcome_class])]

                    for cont_radius in cont_radius_dict[data]:
                        for metric in summary_metrics:
                            metric_series = df[(df['lime_discretize'] == lime_discretize_param) & (df['continuous_radius'] == cont_radius) & (df['discrete_varying_percentage'] == args.discr_param)][metric]

                            metric_avg = metric_series.mean()

                            key = data + algorithm + cont_radius + str(int(args.discr_param*100)) + metric
                            results_dict[outcome_class][key] = [metric_avg] * len(all_total_CFs)

                else:
                    valid_unique_instances = valid_unique_instances_dict[outcome_class][algorithm][postfix_param].copy()

                    cf_config = 'prox_0.5+div_'+str(diversity_weight)+'+algo_' + filealgo + '+yloss_hinge_loss+divloss_dpp_style_inverse_dist+lr_' + str(lr_dict[data + '_' + args.model_type]) + '+postfix_0.1+init_near_x1_False'

                    outdir = 'figure2_experiment_results/' + data + '/' + args.model_type + '/' + cf_config + '/'

                    for total_CFs in all_total_CFs:
                        filename = 'tot_cf_' + str(total_CFs) + '+cont_dist_' + args.cont_radius + '+discrete_perc_' + str(int(args.discr_param*100)) + '.xlsx'
                        df = pd.read_excel(os.path.join(outdir, filename))
                        df = df[df['test_ix'].isin(outcome_class_dict[outcome_class])]

                        print(data, outcome_class, algorithm, filealgo, total_CFs, len(valid_unique_instances[total_CFs]))
                        df = df[df['test_ix'].isin(valid_unique_instances[total_CFs])]

                        for cont_radius in cont_radius_dict[data]:
                            for metric in summary_metrics:
                                metric_series = df[(df['sparsity'] == postfix_param1) & (df['continuous_radius'] == cont_radius) & (df['discrete_varying_percentage'] == args.discr_param)][metric]

                                metric_avg = metric_series.mean()
                                #print(metric, metric_avg)
                                key = data + algorithm + cont_radius + str(int(args.discr_param*100)) + metric
                                if key in results_dict[outcome_class]:
                                    results_dict[outcome_class][key].append(metric_avg)
                                else:
                                    results_dict[outcome_class][key] = [metric_avg]

    outdir = 'figure2_summary_stats/'
    filename = 'figure2_summary' + '_' + args.postfix +  '_' + args.lime_discretize + '_' + args.cont_radius + '_' + str(int(args.discr_param*100)) + '.data'
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    with open(os.path.join(outdir, filename), 'wb') as filehandle:
        pickle.dump(results_dict, filehandle)

    elapsed = timeit.default_timer() - start_time
    m, s = divmod(elapsed, 60)
    print('\n', 'Figure 2 summary done... time taken: ', m, ' mins ', s, ' sec', '\n')

    return results_dict


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # required arguments
    parser.add_argument("--model_type", type=str, help="linear or nonlinear", required=True)
    parser.add_argument("--postfix", type=str, help="with or without", required=True)
    parser.add_argument("--lime_discretize", type=str, help="True or False", required=True)
    parser.add_argument("--cont_radius", type=str, help="mad or user_input", required=True)
    parser.add_argument("--discr_param", type=float, help="0-1", required=True)


    args = parser.parse_args()

    postfix_param = args.postfix + '_postfix'
    lime_discretize_param = 'discretize_' + args.lime_discretize

    lr_dict = {'adult_linear': 0.05, 'adult_nonlinear': 0.05, 'german_linear': 0.05, 'german_nonlinear': 0.05, 'compass_linear': 0.05, 'compass_nonlinear': 0.05, 'lending_linear':0.05, 'lending_nonlinear': 0.05}

    cont_radius_dict = {'adult': ['[5, 2]', '[10, 4]', '[20, 8]'], 'german': ['[3, 544, 1, 1, 4]', '[6, 1088, 2, 2, 7]', '[12, 2176, 3, 3, 14]'], 'compass': ['[1]', '[2]', '[4]'], 'lending': ['[1, 9500, 1, 2]', '[3, 19000, 3, 4]', '[6, 38000, 6, 8]']}

    summary_metrics = ['x1_precision', 'x1_recall', 'x1_f1', 'CF_precision', 'CF_recall', 'CF_f1', 'accuracy']
    datasets = ['adult', 'german', 'compass', 'lending']
    algorithms = ['NoDiverseCF', 'DiverseCF', 'RandomInitCF', 'NoDiverseCF', 'LIME'] #'RandomInitCF',
    all_total_CFs = [1,2,4,6,8,10]

    results_dict = get_figure2_summary_stats(args, postfix_param, lime_discretize_param, lr_dict, cont_radius_dict, summary_metrics, datasets, algorithms, all_total_CFs)
