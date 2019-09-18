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

import sklearn
import lime
import lime.lime_tabular
import timeit
import bisect

def run_experiments(args):

    start_time = timeit.default_timer()

    model_name = args.data + '_ann_dice_new.h5'

    if args.data == 'adult':
        dataset = helpers.load_adult_income_dataset()
        continuous_features=['age', 'hours_per_week']
        outcome_name='income'
        model_path = 'stored_ml_models/'+model_name

        lime_class_names = np.array(['<=50k', '>50k'], dtype='object')

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

        cols = list(dataset.columns)
        cols = cols[1:] + [cols[0]]
        dataset = dataset[cols]
        lime_class_names = np.array(['bad', 'good'], dtype='object')

    elif args.data == 'compass':
        dataset = pd.read_csv('datasets/compass.csv')
        continuous_features=['priors_count']
        outcome_name='two_year_recid'
        model_path = 'stored_ml_models/'+model_name

        lime_class_names = np.array(['wont_recid', 'will_recid'], dtype='object')

    elif args.data == 'lending':
        dataset = pd.read_csv('datasets/lending.csv')
        continuous_features=['emp_length', 'annual_inc', 'open_acc', 'credit_years']
        outcome_name='loan_status'
        model_path = 'stored_ml_models/'+model_name

        lime_class_names = np.array(['Default', 'Paid'], dtype='object')

    d = dice_ml.Data(dataframe=dataset, continuous_features=continuous_features, outcome_name=outcome_name)
    m = dice_ml.Model(model_path = model_path)
    exp = dice_ml.Dice(d, m)

    ## Findin LIME explanations
    lime_hp = lime_helper(exp, lime_class_names)

    # find LIME explanations for all test instances
    outdir = 'lime_explanations/' + args.data + '/'
    filename = 'explanations.data'

    if not os.path.exists(os.path.join(outdir, filename)):
        lime_summary = lime_hp.get_lime_explanations(args.sampling_random_seed, outdir, filename)
    else:
        with open(os.path.join(outdir, filename), 'rb') as filehandle:
            lime_summary = pickle.load(filehandle)

    elapsed = timeit.default_timer() - start_time
    m, s = divmod(elapsed, 60)
    print('\n', 'LIME explanations computed... time taken: ', m, ' mins ', s, ' sec', '\n')


    ## Figure 2 experiments starts

    if args.continuous_radius_type is None:
        return

    start_time = timeit.default_timer()

    outdir = 'lime_explanations/' + args.data + '/figure2_results'
    filename = 'cont_dist_' + args.continuous_radius_type + '+discrete_perc_' + str(int(args.discrete_varying_percentage*100)) + '.xlsx'

    if not os.path.exists(os.path.join(outdir, filename)):
        results_per_query = do_figure2_experiment(lime_hp, lime_summary, args)

        figure2_results_df = pd.DataFrame(results_per_query, columns = ['test_ix', 'continuous_radius', 'discrete_varying_percentage', 'lime_discretize', 'accuracy', 'weighted_f1', 'CF_precision', 'CF_recall', 'CF_f1','CF_support','x1_precision', 'x1_recall', 'x1_f1', 'x1_support', 'tn', 'fp', 'fn', 'tp'])

        if not os.path.exists(outdir):
            os.makedirs(outdir)

        figure2_results_df.to_excel(os.path.join(outdir, filename), index=False)

    else:
        print('file already exists..')

    elapsed = timeit.default_timer() - start_time
    m, s = divmod(elapsed, 60)
    print('\n', 'Figure 2 part done... time taken: ', m, ' mins ', s, ' sec', '\n')


def do_figure2_experiment(lime_hp, lime_summary, args):

    # continuous radius parameter
    mads = lime_hp.dice_exp.data_interface.get_mads_from_training_data(normalized=False)
    if args.continuous_radius_type == 'mad':
        mad_radius = [int(i) for i in args.continuous_radius.strip('[]').split(',')]
        continuous_radius = []
        for ix, rad in enumerate(mad_radius):
            rad_sub = []
            for feature in lime_hp.dice_exp.data_interface.continuous_feature_names:
                rad_sub.append(mad_radius[ix]*mads[feature])
            continuous_radius.append(rad_sub)

    elif args.continuous_radius_type == 'user_input':
        temp = args.continuous_radius.replace("[","",2)[:-2].split('],[')
        continuous_radius = [[int(i) for i in j.strip('[]').split(',')] for j in temp]

    results_per_query = []

    for test_ix, query in enumerate(lime_hp.lime_test):

        if((test_ix % (round(len(lime_hp.lime_test)/10))) == 0):
            print(test_ix , ' done')

        test_pred = lime_hp.lime_predict_fn(np.array([query]))[:,1][0]
        target_cf_class = 1.0 - round(test_pred)

        for cont_radius in continuous_radius:
            samples = get_samples(query, lime_hp.dice_exp, cont_radius, args.discrete_varying_percentage, args.sampling_random_seed, args.sampling_size)

            # blackbox prediction
            temp_preds = lime_hp.lime_predict_fn(samples)[:,1]
            preds_blackbox = convert_probs_preds_to_binary(temp_preds, target_cf_class)

            # LIME predictions
            preds_lime = {}#{'discretize_False':[], 'discretize_True': []}

            # discretize_False and True options
            for discretize_param in ['discretize_False', 'discretize_True']:
                weights = lime_summary[discretize_param][test_ix]['weights']
                intercept = lime_summary[discretize_param][test_ix]['intercept']

                temp_preds = lime_hp.get_lime_based_preds(query, samples, weights, intercept, discretize_param)
                preds_lime[discretize_param] = convert_probs_preds_to_binary(temp_preds, target_cf_class)

                # performance metrics
                tn, fp, fn, tp, acc, weighted_f1, stats = compute_performance_metrics(preds_blackbox, preds_lime[discretize_param])

                results_per_query.append([test_ix, cont_radius, args.discrete_varying_percentage, discretize_param, acc, weighted_f1, round(stats[0][0],2), round(stats[1][0],2), round(stats[2][0],2), stats[3][0], round(stats[0][1],2), round(stats[1][1],2), round(stats[2][1],2), stats[3][1], tn, fp, fn, tp])

    return results_per_query

def compute_performance_metrics(preds_blackbox, preds_lime):
    preds_blackbox_uniques = list(Counter(preds_blackbox).keys())
    preds_lime_uniques = list(Counter(preds_lime).keys())

    if((len(preds_blackbox_uniques)==1) & (len(preds_lime_uniques)==1) &
       (preds_blackbox_uniques[0]==preds_lime_uniques[0])): # this condition means there is only one class predicted by both black-box and CF_explain

        if preds_lime_uniques[0] == 'CF':
            acc = 1.0
            stats = [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [len(preds_lime), 0]]
            weighted_f1 = 1.0
            tn, fp, fn, tp = [len(preds_lime), 0, 0, 0]

        elif preds_lime_uniques[0] == 'x1':
            acc = 1.0
            stats = [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0, len(preds_lime)]]
            weighted_f1 = 1.0
            tn, fp, fn, tp = [0, 0, 0, len(preds_lime)]
    else:
        acc = round(accuracy_score(preds_blackbox, preds_lime), 2)
        stats = precision_recall_fscore_support(preds_blackbox, preds_lime)
        weighted_f1 = round(((stats[2][0]*stats[3][0])+(stats[2][1]*stats[3][1]))/sum(stats[3]), 2)
        tn, fp, fn, tp = confusion_matrix(preds_blackbox, preds_lime).ravel()

    return tn, fp, fn, tp, acc, weighted_f1, stats

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
    for lime_feat_ix, feature in enumerate(exp.data_interface.feature_names):
        if feature in exp.data_interface.continuous_feature_names:

            ranges = exp.data_interface.permitted_range[feature]
            feat_ix = exp.data_interface.encoded_feature_names.index(feature)
            low = max(ranges[0], query[lime_feat_ix] - cont_radius[feat_ix])
            high = min(ranges[1], query[lime_feat_ix] + cont_radius[feat_ix])

            if low >= high:
                low = ranges[0]
                high = ranges[1]

            sample = get_continuous_samples(low, high, precisions[feat_ix], size=sampling_size, seed=sampling_random_seed)
            samples.append(sample)
        else:
            if feature in sampling_categorical_features:
                high = categorical_features_frequencies[feature]-1
                sample = get_continuous_samples(0, high, 0, size=sampling_size, seed=sampling_random_seed)
                samples.append(sample)
            else:
                samples.append([query[lime_feat_ix]]*sampling_size)

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


class lime_helper:

    def __init__(self, exp, lime_class_names):

        self.dice_exp = exp
        self.lime_class_names = lime_class_names

        lime_df = self.dice_exp.data_interface.data_df.copy()
        self.lime_data = lime_df.values

        self.lime_labels = self.lime_data[:,-1]
        self.lime_categorical_features = self.dice_exp.data_interface.categorical_feature_indexes.copy()
        self.lime_continuous_features = self.dice_exp.data_interface.continuous_feature_indexes.copy()

        self.lime_categorical_names = {}
        self.lime_le_features = {}
        for feature in self.lime_categorical_features:
            self.lime_le_features[feature] = sklearn.preprocessing.LabelEncoder()
            self.lime_le_features[feature].fit(self.lime_data[:, feature])
            self.lime_data[:, feature] = self.lime_le_features[feature].transform(self.lime_data[:, feature])
            self.lime_categorical_names[feature] = self.lime_le_features[feature].classes_

        self.lime_data = self.lime_data.astype(float)

        self.lime_encoder = sklearn.preprocessing.OneHotEncoder(categorical_features=self.lime_categorical_features)

        self.lime_train, self.lime_test = sklearn.model_selection.train_test_split(self.lime_data, test_size=0.20, random_state=17)

        lime_test_uniques = pd.DataFrame(self.lime_test, columns=self.dice_exp.data_interface.feature_names+[self.dice_exp.data_interface.outcome_name]).drop_duplicates(subset=self.dice_exp.data_interface.feature_names).reset_index(drop=True)

        if lime_test_uniques.shape[0] < 1000:
            self.lime_test = lime_test_uniques.values[:,:-1]
            self.lime_labels_test = lime_test_uniques.values[:,-1]
        else:
            temp = lime_test_uniques.sample(n=1000, random_state =17)
            #temp = temp.head(500)
            self.lime_test = temp.values[:, :-1]
            self.lime_labels_test = temp.values[:,-1]

        self.lime_labels_train = self.lime_train[:,-1]
        self.lime_train = self.lime_train[:,:-1]

        self.lime_data = self.lime_data[:,:-1]

        self.lime_encoder.fit(self.lime_data)
        self.lime_encoded_train = self.lime_encoder.transform(self.lime_train)

        self.lime_le_features_inverse = {}
        for feature in self.lime_categorical_features:
            org_mapping = dict(zip(self.lime_le_features[feature].classes_, self.lime_le_features[feature].transform(self.lime_le_features[feature].classes_)))
            self.lime_le_features_inverse[feature] = {v: k for k, v in org_mapping.items()}

        self.cont_feat_len = len(self.dice_exp.data_interface.continuous_feature_names)

        self.quantiles = []
        for i in self.lime_continuous_features:
            self.quantiles.append(np.percentile(self.lime_train[:,i], [0, 25, 50, 75]).tolist())

        self.train_mean = np.mean(self.lime_train, axis=0)
        self.train_mean[self.lime_categorical_features] = 0
        self.train_std = np.std(self.lime_train, axis=0)
        self.train_std[self.lime_categorical_features] = 1

    def lime_predict_fn(self, xtest):
        encoded_x = self.lime_encoder.transform(xtest)
        encoded_x = encoded_x.toarray()

        # one hot encoder keeps continuous vars to the end in contrast to pd dummies, so swapping them since pd.dummies convention is followed in di_data
        for i in range(len(encoded_x)):
            categors = encoded_x[i][0:(len(encoded_x[i])-self.cont_feat_len)]
            continues = encoded_x[i][(len(encoded_x[i])-self.cont_feat_len):len(encoded_x[i])]
            encoded_x[i] = np.concatenate([continues, categors])

            for cont_idx in range(self.cont_feat_len):
                encoded_x[i,cont_idx] = ((encoded_x[i,cont_idx]-self.dice_exp.cont_minx[cont_idx])/(self.dice_exp.cont_maxx[cont_idx]-self.dice_exp.cont_minx[cont_idx])).astype(float)

        preds = []
        for query_instance in encoded_x:
            query_instance = np.array([query_instance])
            preds.append(self.dice_exp.predict_fn(query_instance)[0][0])

        for i in range(len(preds)):
            preds[i] = [1.0-preds[i], preds[i]]

        return np.array(preds)

    def get_lime_explanations(self, seed, outdir, filename):

        lime_explainer = {}
        lime_explainer['discretize_False'] = lime.lime_tabular.LimeTabularExplainer(self.lime_train , feature_names = self.dice_exp.data_interface.feature_names, class_names = self.lime_class_names, categorical_features = self.lime_categorical_features, categorical_names = self.lime_categorical_names, verbose=0, discretize_continuous = False)

        lime_explainer['discretize_True'] = lime.lime_tabular.LimeTabularExplainer(self.lime_train , feature_names = self.dice_exp.data_interface.feature_names, class_names = self.lime_class_names, categorical_features = self.lime_categorical_features, categorical_names = self.lime_categorical_names, verbose=0, discretize_continuous = True)

        lime_summary = {}
        lime_summary['discretize_False'] = []
        lime_summary['discretize_True'] = []

        for test_ix, query in enumerate(self.lime_test):

            if((test_ix % (round(len(self.lime_test)/10))) == 0):
                print(test_ix , ' done')

            for discretize_param in ['discretize_False', 'discretize_True']:

                np.random.seed(seed)
                lime_exp = lime_explainer[discretize_param].explain_instance(query, self.lime_predict_fn, num_features=len(self.dice_exp.data_interface.feature_names))

                weights = sorted(lime_exp.as_map()[1], key=lambda x: x[0])
                weights = [c[1] for c in weights]
                intercept = lime_exp.intercept[1]

                lime_summary[discretize_param].append({'intercept': intercept, 'weights': weights})

        if not os.path.exists(outdir):
            os.makedirs(outdir)

        with open(os.path.join(outdir, filename), 'wb') as filehandle:
            pickle.dump(lime_summary, filehandle)

        return lime_summary

    def get_lime_based_preds(self, query, samples, weights, intercept, discretize_param):
        lime_preds = []
        for sample in samples:
            pred = intercept
            for i in self.lime_categorical_features:
                if sample[i] == query[i]: pred += weights[i]

            if discretize_param:
                for lime_feat_ix, feature in enumerate(self.lime_continuous_features):
                    x1c = bisect.bisect_left(self.quantiles[lime_feat_ix], query[feature], 0, len(self.quantiles[lime_feat_ix]))-1
                    samc = bisect.bisect_left(self.quantiles[lime_feat_ix], sample[feature], 0, len(self.quantiles[lime_feat_ix]))-1
                    if x1c == samc: pred += weights[feature]
            else:
                for i in self.lime_continuous_features:
                    pred += weights[i]*((sample[i]-self.train_mean[i])/self.train_std[i])

            lime_preds.append(pred)
        return lime_preds


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # required arguments
    parser.add_argument("--data", type=str, help="dataset name: adult/compass/german", required=True)

    # figure 2 optional parameters
    parser.add_argument("--continuous_radius_type", type=str, help="continuous radius type: mad/user_input", required=False, default=None)
    parser.add_argument("--continuous_radius", type=str, help="continuous sampling radius: [a,b,c] or [[a,b,c],[d,e,f]]", required=False, default=None)
    parser.add_argument("--discrete_varying_percentage", type=float, help="fraction of discrete features that can be varied while sampling: 0-1", required=False, default=None)
    parser.add_argument("--sampling_random_seed", type=int, help="random seed for getting same samples", required=False, default=17)
    parser.add_argument("--sampling_size", type=int, help="sampling size", required=False, default=1000)

    args = parser.parse_args()

    # checking invalid arguments
    if((args.continuous_radius_type == 'mad' and '[[' in args.continuous_radius) &
        (args.continuous_radius_type == 'user_input' and '[[' not in args.continuous_radius)):
        raise ValueError("provide valid continuous radius arguments")

    run_experiments(args)
