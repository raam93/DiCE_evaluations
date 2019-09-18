# import DiCE
import warnings
warnings.filterwarnings("ignore")

import dice_ml
from dice_ml.utils import helpers # helper functions

import os
import sys
import pandas as pd
import numpy as np
import pickle
import argparse
import random

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
from collections import Counter

import timeit

def run_experiments(args, total_CFs):

    start_time = timeit.default_timer()

    if args.model_type == 'linear':
        model_name = args.data + '_logR_dice_new.h5'
    elif args.model_type == 'nonlinear':
        model_name = args.data + '_ann_dice_new.h5'

    if args.data == 'adult':
        dataset = helpers.load_adult_income_dataset()
        continuous_features=['age', 'hours_per_week']
        outcome_name='income'
        model_path = 'stored_ml_models/'+model_name

    elif args.data == 'german':
        dataset = pd.read_csv('datasets/german_credit.csv')

        dataset['credits_this_bank']= dataset['credits_this_bank'].astype(str)
        dataset['people_under_maintenance']= dataset['people_under_maintenance'].astype(str)

        german_dtypes = dataset.columns.to_series().groupby(dataset.dtypes).groups
        german_dtypes = {k.name: v.tolist() for k, v in german_dtypes.items()}

        continuous_features = german_dtypes['int64']
        continuous_features = [feat for feat in continuous_features if feat != 'default']
        outcome_name='default'
        model_path='stored_ml_models/'+model_name

    elif args.data == 'compass':
        dataset = pd.read_csv('datasets/compass.csv')
        continuous_features=['priors_count']
        outcome_name='two_year_recid'
        model_path = 'stored_ml_models/'+model_name

    elif args.data == 'lending':
        dataset = pd.read_csv('datasets/lending.csv')
        continuous_features=['emp_length', 'annual_inc', 'open_acc', 'credit_years']
        outcome_name='loan_status'
        model_path = 'stored_ml_models/'+model_name


    d = dice_ml.Data(dataframe=dataset, continuous_features=continuous_features, outcome_name=outcome_name)
    m = dice_ml.Model(model_path = model_path)
    exp = dice_ml.Dice(d, m)

    # dev data creation
    train, test = d.split_data(d.normalize_data(d.one_hot_encoded_data))
    X_train = train.loc[:, train.columns != outcome_name]
    y_train = train.loc[:, train.columns == outcome_name]
    X_test = test.loc[:, test.columns != outcome_name].values
    y_test = test.loc[:, test.columns == outcome_name].values

    test_unique = test.drop_duplicates(subset=d.encoded_feature_names).reset_index(drop=True)

    dev_data = test_unique.iloc[:, test_unique.columns != outcome_name]
    dev_data = d.from_dummies(dev_data)
    dev_data = d.de_normalize_data(dev_data)

    if dev_data.shape[0] < 1000:
        query_instances = dev_data.to_dict(orient='records')
    else:
        dev_data_sampled = dev_data.sample(n=1000, random_state =17)
        dev_data_sampled = dev_data_sampled.head(500)
        query_instances = dev_data_sampled.to_dict(orient='records')

    ## run experiment - Figure 1
    diversity_loss_type = args.diversity_loss_type.replace(':','_')
    cf_config = 'prox_' + str(args.proximity_weight) + '+' + 'div_' + str(args.diversity_weight) + '+' + 'algo_' + args.algorithm + '+' + 'yloss_' + args.yloss_type + '+' + 'divloss_' + diversity_loss_type + '+' + 'lr_' + str(args.learning_rate) + '+' + 'postfix_' + str(args.post_fix_threshold) + '+' + 'init_near_x1_' + str(args.init_near_x1)

    outdir = 'figure1_experiment_results/' + args.data + '/' + args.model_type + '/' + cf_config + '/'
    filename = 'tot_cf_' + str(total_CFs) + '.data'

    print(outdir)
    print(filename)
    if not os.path.exists(os.path.join(outdir, filename)): # run DiCE only if CFs are not already stored
        dice_summary_overall = run_dice(query_instances, exp, args, outdir, filename, total_CFs)
    else: # if CFs are already computed and stored
        with open(os.path.join(outdir, filename), 'rb') as filehandle:
            dice_summary_overall = pickle.load(filehandle)

    elapsed = timeit.default_timer() - start_time
    m, s = divmod(elapsed, 60)
    print('\n', 'Figure 1 part done... time taken: ', m, ' mins ', s, ' sec', '\n')

    # the following computation is required in computing Figue 2 summary stats
    target_cf_classes = []
    for query_instance in query_instances:
        query_instance = exp.data_interface.prepare_query_instance(query_instance=query_instance, encode=True)
        query_instance = np.array([query_instance.iloc[0].values])
        test_pred = exp.predict_fn(query_instance)[0][0]
        target_cf_classes.append(1.0 - round(test_pred))

    outdir = 'figure1_experiment_results/' + args.data + '/' + args.model_type + '/'
    filename = args.data + '_target_cf_classes.data'
    with open(os.path.join(outdir, filename), 'wb') as filehandle:
        pickle.dump(target_cf_classes, filehandle)

    ## Figure 2 experiment starts
    if args.continuous_radius_type is None:
        return

    start_time = timeit.default_timer()

    outdir = 'figure2_experiment_results/' + args.data + '/' + args.model_type + '/' + cf_config + '/'
    filename = 'tot_cf_' + str(total_CFs) + '+cont_dist_' + args.continuous_radius_type + '+discrete_perc_' + str(int(args.discrete_varying_percentage*100)) + '.xlsx'

    if not os.path.exists(os.path.join(outdir, filename)):
        results_per_query = do_figure2_experiment(query_instances, exp, args, dice_summary_overall, total_CFs)

        figure2_results_df = pd.DataFrame(results_per_query, columns = ['test_ix', 'total_CFs_found', 'continuous_radius', 'discrete_varying_percentage', 'sparsity', 'accuracy', 'weighted_f1', 'CF_precision', 'CF_recall', 'CF_f1','CF_support','x1_precision', 'x1_recall', 'x1_f1', 'x1_support', 'tn', 'fp', 'fn', 'tp'])

        if not os.path.exists(outdir):
            os.makedirs(outdir)

        figure2_results_df.to_excel(os.path.join(outdir, filename), index=False)

    else:
        print('file already exists..')

    elapsed = timeit.default_timer() - start_time
    m, s = divmod(elapsed, 60)
    print('\n', 'Figure 2 part done... time taken: ', m, ' mins ', s, ' sec', '\n')

def do_figure2_experiment(query_instances, exp, args, dice_summary_overall, total_CFs):

    # continuous radius parameter
    mads = exp.data_interface.get_mads_from_training_data(normalized=False)
    if args.continuous_radius_type == 'mad':
        mad_radius = [int(i) for i in args.continuous_radius.strip('[]').split(',')]
        continuous_radius = []
        for ix, rad in enumerate(mad_radius):
            rad_sub = []
            for feature in exp.data_interface.continuous_feature_names:
                rad_sub.append(mad_radius[ix]*mads[feature])
            continuous_radius.append(rad_sub)

    elif args.continuous_radius_type == 'user_input':
        temp = args.continuous_radius.replace("[","",2)[:-2].split('],[')
        continuous_radius = [[int(i) for i in j.strip('[]').split(',')] for j in temp]

    if args.do_both_sparsity == 'True':
        sparsity_option = ['without_postfix', 'with_postfix']
    else:
        sparsity_option = ['with_postfix']

    results_per_query = []

    for test_ix, query in enumerate(query_instances):

        if((test_ix % (round(len(query_instances)/10))) == 0):
            print(test_ix , ' done')

        test_pred = blackbox_prediction(exp, [list(query.values())], keys=list(query.keys()))[0]
        target_cf_class = 1.0 - round(test_pred)

        for cont_radius in continuous_radius:

            samples = get_samples(query, exp, cont_radius, args.discrete_varying_percentage, args.sampling_random_seed, args.sampling_size)

            # blackbox predictions
            temp_preds = blackbox_prediction(exp, samples)
            preds_blackbox = convert_probs_preds_to_binary(temp_preds, target_cf_class)

            preds_1nn = {'without_postfix':[], 'with_postfix': []}

            for sample in samples:

                # 1nn predictions
                cfs_distance_isless = False

                distance_from_test_instance = 0
                for feat_ix, feature in enumerate(exp.data_interface.feature_names):
                    if feature in mads:
                        distance_from_test_instance += (abs(query[feature] - sample[feat_ix])/mads[feature])
                    else:
                        if query[feature] != sample[feat_ix]:
                            distance_from_test_instance += 1

                # with and without sparsity
                for sparsity_param in sparsity_option:

                    for cf_ix in range(total_CFs):
                        distance_from_cfs = 0
                        for feat_ix, feature in enumerate(exp.data_interface.feature_names):
                            if feature in mads:
                                distance_from_cfs += (abs(dice_summary_overall[sparsity_param][test_ix][cf_ix][feat_ix] - sample[feat_ix])/mads[feature])
                            else:
                                if dice_summary_overall[sparsity_param][test_ix][cf_ix][feat_ix] != sample[feat_ix]:
                                    distance_from_cfs += 1

                        if distance_from_cfs < distance_from_test_instance:
                            cfs_distance_isless = True
                            break

                    if cfs_distance_isless:
                        preds_1nn[sparsity_param].append('CF')
                    else:
                        preds_1nn[sparsity_param].append('x1')


            # performance metrics
            for sparsity_param in sparsity_option:
                tn, fp, fn, tp, acc, weighted_f1, stats = compute_performance_metrics(preds_blackbox, preds_1nn[sparsity_param])

                total_CFs_found = dice_summary_overall[sparsity_param][test_ix][-1]

                results_per_query.append([test_ix, total_CFs_found, cont_radius, args.discrete_varying_percentage, sparsity_param, acc, weighted_f1, round(stats[0][0],2), round(stats[1][0],2), round(stats[2][0],2), stats[3][0], round(stats[0][1],2), round(stats[1][1],2), round(stats[2][1],2), stats[3][1], tn, fp, fn, tp])

    return results_per_query


def convert_probs_preds_to_binary(preds, target_cf_class):
    preds_blackbox = []
    if target_cf_class == 0:
        for pred_ix in range(len(preds)):
            if preds[pred_ix] < 0.5:
                preds_blackbox.append('CF')
            else:
                preds_blackbox.append('x1')
    else:
        for pred_ix in range(len(preds)):
            if preds[pred_ix] > 0.5:
                preds_blackbox.append('CF')
            else:
                preds_blackbox.append('x1')

    return preds_blackbox

def blackbox_prediction(exp, samples, keys=None):
    preds = []
    for sample in samples:
        if keys is not None:
            sample = dict(zip(keys, sample))
        else:
            sample = dict(zip(exp.data_interface.feature_names, sample))
        encoded = [0.0]*len(exp.data_interface.encoded_feature_names)
        for feature in sample:
            if feature in exp.data_interface.continuous_feature_names:
                feat_ix = exp.data_interface.encoded_feature_names.index(feature)
                val = ((sample[feature]-exp.cont_minx[feat_ix])/(exp.cont_maxx[feat_ix]-exp.cont_minx[feat_ix]))
                encoded_feat_ix = exp.data_interface.encoded_feature_names.index(feature)
                encoded[encoded_feat_ix] = val
            else:
                encoded_feat_ix = exp.data_interface.encoded_feature_names.index(feature+'_'+sample[feature])
                encoded[encoded_feat_ix] = 1.0
        encoded = np.array([encoded])
        preds.append(exp.predict_fn(encoded)[0][0])
    return preds

def compute_performance_metrics(preds_blackbox, preds_1nn):
    preds_blackbox_uniques = list(Counter(preds_blackbox).keys())
    preds_1nn_uniques = list(Counter(preds_1nn).keys())

    if((len(preds_blackbox_uniques)==1) & (len(preds_1nn_uniques)==1) &
       (preds_blackbox_uniques[0]==preds_1nn_uniques[0])): # this condition means there is only one class predicted by both black-box and CF_explain

        if preds_1nn_uniques[0] == 'CF':
            acc = 1.0
            stats = [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [len(preds_1nn), 0]]
            weighted_f1 = 1.0
            tn, fp, fn, tp = [len(preds_1nn), 0, 0, 0]

        elif preds_1nn_uniques[0] == 'x1':
            acc = 1.0
            stats = [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0, len(preds_1nn)]]
            weighted_f1 = 1.0
            tn, fp, fn, tp = [0, 0, 0, len(preds_1nn)]
    else:
        acc = round(accuracy_score(preds_blackbox, preds_1nn), 2)
        stats = precision_recall_fscore_support(preds_blackbox, preds_1nn)
        weighted_f1 = round(((stats[2][0]*stats[3][0])+(stats[2][1]*stats[3][1]))/sum(stats[3]), 2)
        tn, fp, fn, tp = confusion_matrix(preds_blackbox, preds_1nn).ravel()

    return tn, fp, fn, tp, acc, weighted_f1, stats


def get_samples(query, exp, cont_radius, discrete_varying_percentage, sampling_random_seed, sampling_size):

    # first get required parameters
    precisions = exp.data_interface.get_decimal_precisions()[0:len(exp.encoded_continuous_feature_indexes)]

    categorical_features_frequencies = {}
    for feature in exp.data_interface.categorical_feature_names:
        categorical_features_frequencies[feature] = len(exp.data_interface.train_df[feature].value_counts())

    if sampling_random_seed is not None:
        random.seed(sampling_random_seed)
    sampling_categorical_features = random.sample(exp.data_interface.categorical_feature_names,
                                             int(discrete_varying_percentage*len(exp.data_interface.categorical_feature_names)))
    samples = []
    for feature in exp.data_interface.feature_names:
        if feature in exp.data_interface.continuous_feature_names:
            ranges = exp.data_interface.permitted_range[feature]
            feat_ix = exp.data_interface.encoded_feature_names.index(feature)
            low = max(ranges[0], query[feature] - cont_radius[feat_ix])
            high = min(ranges[1], query[feature] + cont_radius[feat_ix])

            if low >= high:
                low = ranges[0]
                high = ranges[1]

            sample = get_continuous_samples(low, high, precisions[feat_ix], size=sampling_size, seed=sampling_random_seed)
            samples.append(sample)

        else:
            if feature in sampling_categorical_features:
                if sampling_random_seed is not None:
                    random.seed(sampling_random_seed)
                sample = random.choices(exp.data_interface.train_df[feature].unique(), k=sampling_size)
                samples.append(sample)
            else:
                samples.append([query[feature]]*sampling_size)

    samples = pd.DataFrame(dict(zip(exp.data_interface.feature_names, samples))).values
    return samples


def get_continuous_samples(low, high, precision, size=1000, seed=None):
    if seed is not None:
        np.random.seed(seed)

    if precision == 0:
        result = np.random.randint(low, high+1, size).tolist()
        result = [float(r) for r in result]
    else:
        result = np.random.uniform(low, high+(10**-precision), size)
        result = [round(r, precision) for r in result]
    return result


def run_dice(query_instances, exp, args, outdir, filename, total_CFs):

    dice_summary_overall = {}
    dice_summary_without_postfix = []
    dice_summary_with_postfix = []

    # get MAD
    mads = exp.data_interface.get_mads(normalized=True)

    # create feature weights
    feature_weights = {}
    for feature in mads:
        feature_weights[feature] = round(1/mads[feature], 2)

    if args.init_near_x1 == "False":
        init_near_x1 = False
    elif args.init_near_x1 == "True":
        init_near_x1 = True

    for test_ix, query_instance in enumerate(query_instances):

        if((test_ix % (round(len(query_instances)/10))) == 0):
            print(test_ix , ' done')

        dice_exp = exp.generate_counterfactuals(query_instance, total_CFs=total_CFs, desired_class="opposite", yloss_type=args.yloss_type, diversity_loss_type=args.diversity_loss_type, feature_weights=feature_weights, max_iter=5000, loss_converge_maxiter=2, proximity_weight=args.proximity_weight, diversity_weight=args.diversity_weight, categorical_penalty=0.1, stopping_threshold=0.5, verbose=False, learning_rate=args.learning_rate, algorithm=args.algorithm,  post_fix_threshold=args.post_fix_threshold, init_near_query_instance=init_near_x1)

        # without postfix
        summary = [cfs for cfs in dice_exp.final_cfs_list]
        summary.extend([exp.converged, exp.max_iterations_run, exp.elapsed, exp.valid_cfs_found, exp.total_CFs_found])
        dice_summary_without_postfix.append(summary)

        # with postfix
        summary = [cfs for cfs in dice_exp.final_cfs_list_postfixed]
        summary.extend([exp.converged, exp.max_iterations_run, exp.elapsed, exp.valid_cfs_found, exp.total_CFs_found])
        dice_summary_with_postfix.append(summary)

    dice_summary_overall['without_postfix'] = dice_summary_without_postfix
    dice_summary_overall['with_postfix'] = dice_summary_with_postfix

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    with open(os.path.join(outdir, filename), 'wb') as filehandle:
        pickle.dump(dice_summary_overall, filehandle)

    return dice_summary_overall


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # required arguments
    parser.add_argument("--data", type=str, help="dataset name: adult/compass/german", required=True)
    parser.add_argument("--learning_rate", type=float, help="learning rate: 0-1", required=True)
    parser.add_argument("--post_fix_threshold", type=float, help="post-fix threshold: 0-1", required=True)
    parser.add_argument("--total_CFs", type=str, help="list of CFs", required=True)
    parser.add_argument("--model_type", type=str, help="linear or nonlinear", required=True)

    # optional arguments
    parser.add_argument("--init_near_x1", type=str, help="init near query instance", required=False, default="True")
    parser.add_argument("--proximity_weight", type=float, help="proximity weight: 0-1", required=False, default=0.5)
    parser.add_argument("--diversity_weight", type=float, help="diversity weight: 0-1", required=False, default=1.0)
    parser.add_argument("--algorithm", type=str, help="CF algorithm: DiverseCF/RandomInitCF/NoDiverseCF", required=False, default='DiverseCF')
    parser.add_argument("--yloss_type", type=str, help="yloss type: l2_loss/log_loss/hinge_loss", required=False, default='hinge_loss')
    parser.add_argument("--diversity_loss_type", type=str, help="diversity loss type: avg_dist/dpp_style:inverse_dist", required=False, default='dpp_style:inverse_dist')

    # figure 2 optional parameters
    parser.add_argument("--continuous_radius_type", type=str, help="continuous radius type: mad/user_input", required=False, default=None)
    parser.add_argument("--continuous_radius", type=str, help="continuous sampling radius: [a,b,c] or [[a,b,c],[d,e,f]]", required=False, default=None)
    parser.add_argument("--discrete_varying_percentage", type=float, help="fraction of discrete features that can be varied while sampling: 0-1", required=False, default=None)
    parser.add_argument("--sampling_random_seed", type=int, help="random seed for getting same samples", required=False, default=17)
    parser.add_argument("--sampling_size", type=int, help="sampling size", required=False, default=1000)
    parser.add_argument("--do_both_sparsity", type=str, help="True or False", required=False, default=True)


    args = parser.parse_args()

    # checking invalid arguments
    if((args.continuous_radius_type == 'mad' and '[[' in args.continuous_radius) &
        (args.continuous_radius_type == 'user_input' and '[[' not in args.continuous_radius)):
        raise ValueError("provide valid continuous radius arguments")

    total_CFs_list = [int(i) for i in args.total_CFs.strip('[]').split(',')]

    for total_CFs in total_CFs_list:
        print('total_CFs: ', total_CFs, '\n')
        run_experiments(args, total_CFs)
