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
import timeit

import matplotlib.pyplot as pl
import seaborn as sns

def get_stats_for_one_data(args):

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

        class_names = ['le50k', 'g50k']

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

        class_names = ['good', 'bad']

    elif args.data == 'compass':
        dataset = pd.read_csv('datasets/compass.csv')
        continuous_features=['priors_count']
        outcome_name='two_year_recid'
        model_path = 'stored_ml_models/'+model_name

        class_names = ['wont_recidivate', 'will_recidivate']

    elif args.data == 'lending':
        dataset = pd.read_csv('datasets/lending.csv')
        continuous_features=['emp_length', 'annual_inc', 'open_acc', 'credit_years']
        outcome_name='loan_status'
        model_path = 'stored_ml_models/'+model_name

        class_names = ['Default', 'Paid']

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
        dev_data_sampled = dev_data
    else:
        dev_data_sampled = dev_data.sample(n=1000, random_state =17)
        dev_data_sampled = dev_data_sampled.head(500)
        query_instances = dev_data_sampled.to_dict(orient='records')

    target_cf_classes = []
    for query_instance in query_instances:
        query_instance = exp.data_interface.prepare_query_instance(query_instance=query_instance, encode=True)
        query_instance = np.array([query_instance.iloc[0].values])
        test_pred = exp.predict_fn(query_instance)[0][0]
        target_cf_classes.append(1.0 - round(test_pred))

    outcome_class_dict = {'class0_' + class_names[0] : [ix for ix in range(len(target_cf_classes)) if target_cf_classes[ix] == 0.0], 'class1_' + class_names[1] : [ix for ix in range(len(target_cf_classes)) if target_cf_classes[ix] == 1.0]}

    cf_algorithms = [i for i in args.algorithms.strip('[]').split(',')]

    valid_unique_instances = {}

    summary_stats_fns = {
        "nonlinear": get_dice_summary_stats_nonlinear,
        "linear": get_dice_summary_stats_linear
    }

    for outcome_class in outcome_class_dict:
        figure1_results = {}
        valid_unique_instances[outcome_class] = {}

        for algorithm in cf_algorithms:

            if algorithm == 'NoDiverseCF':
                diversity_weight = 0.0
                algo = 'DiverseCF'
            else:
                diversity_weight = args.diversity_weight
                algo = algorithm

            figure1_results[algorithm] = {}
            valid_unique_instances[outcome_class][algorithm] = {}

            figure1_results[algorithm]['with_postfix'] = {'valid_unique_CFs': [], 'cont_prox_mad': [], 'cont_prox_count': [], 'cate_prox_count': [], 'cont_div_mad': [], 'cont_div_count': [], 'cate_div_count': []}
            figure1_results[algorithm]['without_postfix'] = {'valid_unique_CFs': [], 'cont_prox_mad': [], 'cont_prox_count': [], 'cate_prox_count': [], 'cont_div_mad': [], 'cont_div_count': [], 'cate_div_count': []}

            diversity_loss_type = args.diversity_loss_type.replace(':','_')
            cf_config = 'prox_' + str(args.proximity_weight) + '+' + 'div_' + str(diversity_weight) + '+' + 'algo_' + algo + '+' + 'yloss_' + args.yloss_type + '+' + 'divloss_' + diversity_loss_type + '+' + 'lr_' + str(args.learning_rate) + '+' + 'postfix_' + str(args.post_fix_threshold) + '+' + 'init_near_x1_' + str(args.init_near_x1)

            valid_unique_instances[outcome_class][algorithm]['with_postfix'] = {}
            valid_unique_instances[outcome_class][algorithm]['without_postfix'] = {}

            for total_CFs in [1,2,4,6,8,10]:
                with open('figure1_experiment_results/'+args.data+'/' + args.model_type +'/' + cf_config + '/tot_cf_' + str(total_CFs) + '.data', 'rb') as filehandle:
                    dice_summary = pickle.load(filehandle)

                for post in ['with_postfix', 'without_postfix']:
                    # print(outcome_class, ' + ', algorithm, ' + ', post, ' + ', total_CFs)
                    results = summary_stats_fns[args.model_type](args.data, total_CFs, dice_summary[post], target_cf_classes, query_instances, dev_data_sampled, exp, outcome_class_dict[outcome_class])

                    for ix, metric in enumerate(figure1_results[algorithm][post]):
                        figure1_results[algorithm][post][metric].append(results[ix])

                    # print(len(results[-1]))
                    valid_unique_instances[outcome_class][algorithm][post][total_CFs] = results[-1].copy()

        outdir = 'figure1_summary_stats/'
        filename = args.data + '_' + args.model_type +  '_' + outcome_class + '.data'
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        with open(os.path.join(outdir, filename), 'wb') as filehandle:
            pickle.dump(figure1_results, filehandle)


    filename = args.data + '_' + args.model_type + '_valid_unique_list.data'
    with open(os.path.join(outdir, filename), 'wb') as filehandle:
        pickle.dump(valid_unique_instances, filehandle)



def get_dice_summary_stats_nonlinear(data, total_CFs, dice_summary, target_cf_classes, query_instances, dev_data_sampled, exp, class_valid_ix):

    avg_valid_cfs = 0
    avg_valid_unique_cfs = 0

    if data == 'german':
        cf_cont_ix = [ix-1 for ix in exp.data_interface.continuous_feature_indexes]
    else:
        cf_cont_ix = [ix for ix in exp.data_interface.continuous_feature_indexes]
    cf_cate_ix = [ix for ix in range(len(exp.data_interface.feature_names)) if ix not in cf_cont_ix]

    org_cont_ix = [ix for ix in range(len(cf_cont_ix))]
    org_cate_ix = [ix for ix in range(len(exp.data_interface.feature_names)) if ix not in org_cont_ix]

    mads = list(exp.data_interface.get_mads().values())
    tot_sample = len([query_instances[ix] for ix in range(len(query_instances)) if ix in class_valid_ix]) #len(query_instances)

    valid_instances_ixs = []
    for ix, summ in enumerate(dice_summary):

        if ix not in class_valid_ix:
            continue

        # stats
        avg_valid_cfs += ((summ[-1]/total_CFs)*100)
        cfs = summ[0:total_CFs]
        res = []
        for cf in cfs:
            if(((target_cf_classes[ix] == 0)&(cf[-1] <= 0.5)) | ((target_cf_classes[ix] == 1)&(cf[-1] >= 0.5))):
            #if round(cf[-1]) == target_cf_classes[ix]:
                res.append(cf)

        unique_valid_cfs = [list(x) for x in set(tuple(x) for x in res)]
        if len(unique_valid_cfs) == total_CFs:
            valid_instances_ixs.append(ix)

        tot = len(unique_valid_cfs)
        avg_valid_unique_cfs += ((tot/total_CFs)*100)

    tot_valid_unique_samples = len(valid_instances_ixs)

    avg_cont_prox = 0
    avg_cate_prox = 0
    avg_cont_div = 0
    avg_cate_div = 0
    avg_cont_prox_count = 0
    avg_cont_div_count = 0

    for ix, summ in enumerate(dice_summary):

        if ix not in class_valid_ix:
            continue

        if ix not in valid_instances_ixs:
            continue

        x1 = list(dev_data_sampled.iloc[ix])

        # proximity
        cont_prox = 0
        cate_prox = 0

        cont_prox_count = 0

        for cf in summ[0:total_CFs]:
            for i,j in zip(cf_cate_ix, org_cate_ix):
                if cf[i] != x1[j]:
                    cate_prox += 1

            for i,j in zip(cf_cont_ix, org_cont_ix):
                if cf[i] != x1[j]:
                    cont_prox_count += 1

                cont_prox += ((abs(cf[i]-x1[j]))/mads[j])

        cate_prox = cate_prox/(total_CFs*len(cf_cate_ix))

        cont_prox = cont_prox/(total_CFs*len(cf_cont_ix))
        cont_prox_count = cont_prox_count/(total_CFs*len(cf_cont_ix))

        # diversity
        cont_div = 0
        cate_div = 0

        cont_div_count = 0

        co = 0
        for m in range(0,total_CFs):
            for n in range(m+1,total_CFs):
                co += 1

                for i in cf_cate_ix:
                    if summ[m][i] != summ[n][i]:
                        cate_div += 1

                for i,j in zip(cf_cont_ix, org_cont_ix):
                    if summ[m][i] != summ[n][i]:
                        cont_div_count += 1

                    cont_div += ((abs(summ[m][i]-summ[n][i]))/mads[j])

        if co != 0:
            cate_div = cate_div/(co*len(cf_cate_ix))

            cont_div = cont_div/(co*len(cf_cont_ix))
            cont_div_count = cont_div_count/(co*len(cf_cont_ix))


        avg_cont_prox += cont_prox
        avg_cate_prox += cate_prox
        avg_cont_div += cont_div
        avg_cate_div += cate_div

        avg_cont_prox_count += cont_prox_count
        avg_cont_div_count += cont_div_count

    if tot_valid_unique_samples == 0:
        tot_valid_unique_samples = 1
        avg_cont_prox = 0
        avg_cont_prox_count = 0
        avg_cate_prox = 0
        avg_cont_div = 0
        avg_cont_div_count = 0
        avg_cate_div = 0

    # print(total_CFs, len(class_valid_ix), tot_valid_unique_samples, len(valid_instances_ixs), avg_cont_prox, avg_cont_prox_count, avg_cate_prox, avg_cont_div, avg_cont_div_count, avg_cate_div, '\n')

    return np.round(avg_valid_unique_cfs/tot_sample, 3), np.round(avg_cont_prox/tot_valid_unique_samples, 3), np.round(avg_cont_prox_count/tot_valid_unique_samples, 3), np.round(avg_cate_prox/tot_valid_unique_samples, 3), np.round(avg_cont_div/tot_valid_unique_samples, 3), np.round(avg_cont_div_count/tot_valid_unique_samples, 3), np.round(avg_cate_div/tot_valid_unique_samples, 3), valid_instances_ixs


def get_dice_summary_stats_linear(data, total_CFs, dice_summary, target_cf_classes, query_instances, dev_data_sampled, exp, class_valid_ix):

    avg_valid_cfs = 0
    avg_valid_unique_cfs = 0

    if data == 'german':
        cf_cont_ix = [ix-1 for ix in exp.data_interface.continuous_feature_indexes]
    else:
        cf_cont_ix = [ix for ix in exp.data_interface.continuous_feature_indexes]
    cf_cate_ix = [ix for ix in range(len(exp.data_interface.feature_names)) if ix not in cf_cont_ix]

    org_cont_ix = [ix for ix in range(len(cf_cont_ix))]
    org_cate_ix = [ix for ix in range(len(exp.data_interface.feature_names)) if ix not in org_cont_ix]

    mads = list(exp.data_interface.get_mads().values())
    tot_sample = len([query_instances[ix] for ix in range(len(query_instances)) if ix in class_valid_ix]) #len(query_instances)

    valid_instances_ixs = []
    for ix, summ in enumerate(dice_summary):

        if ix not in class_valid_ix:
            continue

        # stats
        avg_valid_cfs += ((summ[-1]/total_CFs)*100)
        cfs = summ[0:total_CFs]
        res = []
        for cf in cfs:
            if(((target_cf_classes[ix] == 0)&(cf[-1] <= 0.5)) | ((target_cf_classes[ix] == 1)&(cf[-1] >= 0.5))):
            #if round(cf[-1]) == target_cf_classes[ix]:
                res.append(cf)

        unique_valid_cfs = [list(x) for x in set(tuple(x) for x in res)]
        if len(unique_valid_cfs) == total_CFs:
            valid_instances_ixs.append(ix)

        tot = len(unique_valid_cfs)
        avg_valid_unique_cfs += ((tot/total_CFs)*100)

    tot_valid_unique_samples = len(valid_instances_ixs)

    avg_cont_prox = 0
    avg_cate_prox = 0
    avg_cont_div = 0
    avg_cate_div = 0
    avg_cont_prox_count = 0
    avg_cont_div_count = 0

    for ix, summ in enumerate(dice_summary):

        if ix not in class_valid_ix:
            continue

        if ix not in valid_instances_ixs:
            continue

        x1 = list(dev_data_sampled.iloc[ix])

        # proximity
        cont_prox = 0
        cate_prox = 0

        cont_prox_count = 0

        for cf in summ[0:total_CFs]:
            for i,j in zip(cf_cate_ix, org_cate_ix):
                if cf[i] != x1[j]:
                    cate_prox += 1

            for i,j in zip(cf_cont_ix, org_cont_ix):
                if cf[i] != x1[j]:
                    cont_prox_count += 1

                cont_prox += ((abs(cf[i]-x1[j]))/mads[j])

        cate_prox = cate_prox/(total_CFs*len(cf_cate_ix))

        cont_prox = cont_prox/(total_CFs*len(cf_cont_ix))
        cont_prox_count = cont_prox_count/(total_CFs*len(cf_cont_ix))

        # diversity
        cont_div = 0
        cate_div = 0

        cont_div_count = 0

        co = 0
        for m in range(0,total_CFs):
            for n in range(m+1,total_CFs):
                co += 1

                for i in cf_cate_ix:
                    if summ[m][i] != summ[n][i]:
                        cate_div += 1

                for i,j in zip(cf_cont_ix, org_cont_ix):
                    if summ[m][i] != summ[n][i]:
                        cont_div_count += 1

                    cont_div += ((abs(summ[m][i]-summ[n][i]))/mads[j])

        if co != 0:
            cate_div = cate_div/(co*len(cf_cate_ix))

            cont_div = cont_div/(co*len(cf_cont_ix))
            cont_div_count = cont_div_count/(co*len(cf_cont_ix))


        avg_cont_prox += cont_prox
        avg_cate_prox += cate_prox
        avg_cont_div += cont_div
        avg_cate_div += cate_div

        avg_cont_prox_count += cont_prox_count
        avg_cont_div_count += cont_div_count

    if tot_valid_unique_samples == 0:
        tot_valid_unique_samples = 1
        avg_cont_prox = 0
        avg_cont_prox_count = 0
        avg_cate_prox = 0
        avg_cont_div = 0
        avg_cont_div_count = 0
        avg_cate_div = 0

    # print(total_CFs, len(class_valid_ix), tot_valid_unique_samples, len(valid_instances_ixs), avg_cont_prox, avg_cont_prox_count, avg_cate_prox, avg_cont_div, avg_cont_div_count, avg_cate_div, '\n')

    return np.round(avg_valid_unique_cfs/tot_sample, 3), np.round(avg_cont_prox/tot_valid_unique_samples, 3), np.round(avg_cont_prox_count/tot_valid_unique_samples, 3), np.round(avg_cate_prox/tot_valid_unique_samples, 3), np.round(avg_cont_div/tot_valid_unique_samples, 3), np.round(avg_cont_div_count/tot_valid_unique_samples, 3), np.round(avg_cate_div/tot_valid_unique_samples, 3), valid_instances_ixs

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # required arguments
    parser.add_argument("--data", type=str, help="data: adult/compass/german", required=True)
    parser.add_argument("--learning_rate", type=float, help="learning rate: 0-1", required=True)
    parser.add_argument("--post_fix_threshold", type=float, help="post-fix threshold: 0-1", required=True)
    parser.add_argument("--model_type", type=str, help="linear or nonlinear", required=True)
    parser.add_argument("--algorithms", type=str, help="list of CF algorithms: DiverseCF/RandomInitCF/NoDiverseCF", required=True)

    # optional
    parser.add_argument("--init_near_x1", type=str, help="init near query instance", required=False, default="True")
    parser.add_argument("--proximity_weight", type=float, help="proximity weight: 0-1", required=False, default=0.5)
    parser.add_argument("--diversity_weight", type=float, help="diversity weight: 0-1", required=False, default=1.0)
    parser.add_argument("--yloss_type", type=str, help="yloss type: l2_loss/log_loss/hinge_loss", required=False, default='hinge_loss')
    parser.add_argument("--diversity_loss_type", type=str, help="diversity loss type: avg_dist/dpp_style:inverse_dist", required=False, default='dpp_style:inverse_dist')

    args = parser.parse_args()

    get_stats_for_one_data(args)
