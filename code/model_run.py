import argparse
import datetime
import time
from copy import deepcopy

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from numpy import interp
from scipy.stats import uniform
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectorMixin
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve
from sklearn.model_selection import RepeatedStratifiedKFold, RandomizedSearchCV, cross_validate, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from xgboost import XGBClassifier


class SelectKBestFromModel(BaseEstimator, SelectorMixin):
    """Feature selection based on k-best features of a fitted model.
    It corresponds to a recursive feature elimination with a single step.
    """

    def __init__(self, estimator, k=3, model_id=None):
        """Initialize the object.
        Parameters
        ----------
        estimator : object
            A supervised learning estimator with a ``fit`` method that provides
            information about feature importance either through a ``coef_``
            attribute or through a ``feature_importances_`` attribute.
        k : int, default=3
            The number of features to select.
        """
        self.estimator = estimator
        self.k = k
        self.model_id = model_id
        self.mask_ = None

    def fit(self, X, y=None):
        """Fit the underlying estimator.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            The training input samples.
        y : array-like, shape = [n_samples]
            The target values.
        """
        self.estimator.fit(X, y)
        return self

    def _get_support_mask(self):
        """Get a mask of the features selected."""
        try:
            scores = self.estimator.coef_[0, :]
        except (AttributeError, KeyError):
            scores = self.estimator.feature_importances_
        mask = np.zeros(len(scores))
        if self.k > len(scores):
            self.k = len(scores)
        mask[np.argpartition(abs(scores), -self.k)[-self.k:]] = 1
        self.mask_ = mask.astype(bool)
        return self.mask_


def get_support(model):
    support = None
    for step in model.best_estimator_.steps:
        if isinstance(step[1], SelectorMixin) or isinstance(step[1], SelectKBestFromModel):
            support = step[1].get_support()
        elif isinstance(step[1], ColumnTransformer):
            support_ct = step[1].transformers_[0][1].steps[-1][1].get_support()
            support_pt = step[1].transformers_[1][1].steps[-1][1].get_support()
            support = np.concatenate([support_ct, support_pt])
    print("Support")
    print(support)
    return support


def get_weights(model):
    try:
        weights = model.best_estimator_.steps[-1][1].coef_[0]
    except AttributeError:
        weights = model.best_estimator_.steps[-1][1].feature_importances_
    print("Weights")
    print(weights)
    return weights


def plot_roc_curves():
    model_id = 0
    auc_tuning = []
    fprs_tuning = []
    tprs_tuning = []
    precs_tuning = []
    recalls_tuning = []
    auc_test = []
    fprs_test = []
    tprs_test = []
    precs_test = []
    recalls_test = []
    for tune_index, test_index in tqdm(cv_out.split(X, y, groups), desc='Outer loop'):
        model = ncv['estimator'][model_id]
        X_tune = X.iloc[tune_index, :]
        y_tune = y.iloc[tune_index]
        X_test = X.iloc[test_index, :]
        y_test = y.iloc[test_index]

        # predict on the test split
        y_tune_pred = model.predict_proba(X_tune)[:, 1]
        # evaluate the performance
        roc_auc = roc_auc_score(y_tune, y_tune_pred)
        auc_tuning.append(roc_auc)
        #  calculate ROC curve
        fpr, tpr, _ = roc_curve(y_tune, y_tune_pred)
        fprs_tuning.append(fpr)
        tprs_tuning.append(tpr)
        # calculate PR curve
        prec, recall, _ = precision_recall_curve(y_tune, y_tune_pred)
        precs_tuning.append(prec)
        recalls_tuning.append(recall)

        # train
        model_temp = deepcopy(model.best_estimator_)
        model_temp.fit(X_tune, y_tune)
        # predict on the test split
        y_test_pred = model.predict_proba(X_test)[:, 1]
        # evaluate the performance
        roc_auc = roc_auc_score(y_test, y_test_pred)
        auc_test.append(roc_auc)
        #  calculate ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_test_pred)
        fprs_test.append(fpr)
        tprs_test.append(tpr)
        # calculate PR curve
        prec, recall, _ = precision_recall_curve(y_test, y_test_pred)
        precs_test.append(prec)
        recalls_test.append(recall)

        # increase model iterator
        model_id += 1

    curves_tuning = {'auc_scores': auc_tuning, 'fprs': fprs_tuning, 'tprs': tprs_tuning,
                     'precs': precs_tuning, 'recalls': recalls_tuning}

    curves_testing = {'auc_scores': auc_test, 'fprs': fprs_test, 'tprs': tprs_test,
                      'precs': precs_test, 'recalls': recalls_test}

    def read_roc_data(curves):
        auc_scores = curves['auc_scores']
        mean_auc = np.mean(auc_scores)
        std_auc = np.std(auc_scores)
        fprs = curves['fprs']
        tprs = curves['tprs']
        mean_fpr = np.linspace(0, 1, 100)
        tprs2 = list()
        for i in range(len(tprs)):
            tprs2.append(interp(mean_fpr, fprs[i], tprs[i]))
            tprs2[-1][0] = 0.0
        mean_tpr = np.mean(tprs2, axis=0)
        mean_tpr[-1] = 1.0
        std_tpr = np.std(tprs2, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        j_index = mean_tpr - mean_fpr
        out = {'mean_fpr': mean_fpr, 'mean_tpr': mean_tpr, 'std_auc': std_auc, 'std_tpr': std_tpr, 'tprs_lower': tprs_lower, 'tprs_upper': tprs_upper, 'j_index': j_index, 'mean_auc': mean_auc}
        return out

    # Display results
    print('Tuning parameters:')
    print(cv_in)
    print(cv_out)
    print(n_iter)

    print(f"Train score: {ncv['train_score'].mean():.3f} +/- {ncv['train_score'].std():.3f}")
    print(f"Test score: {ncv['test_score'].mean():.3f} +/- {ncv['test_score'].std():.3f}")

    aucs = list()
    plt.figure(figsize=(3.75, 3.75))

    # Plot tuning curves
    roc_data = read_roc_data(curves_tuning)
    mean_fpr = roc_data['mean_fpr']
    mean_tpr = roc_data['mean_tpr']
    mean_auc = roc_data['mean_auc']
    std_auc = roc_data['std_auc']
    aucs.append(mean_auc)
    tprs_lower = roc_data['tprs_lower']
    tprs_upper = roc_data['tprs_upper']
    plt.plot(mean_fpr, mean_tpr, color='C0', alpha=0.8,
             label=r'Tuning (AUC = %0.3f $\pm$ %0.3f)' % (mean_auc, std_auc),
             lw=2)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='C0', alpha=.2,
                     label=r'Tuning $\pm$ 1 std. dev.')

    # Plot tuning curves
    roc_data = read_roc_data(curves_testing)
    mean_fpr = roc_data['mean_fpr']
    mean_tpr = roc_data['mean_tpr']
    mean_auc = roc_data['mean_auc']
    std_auc = roc_data['std_auc']
    aucs.append(mean_auc)
    tprs_lower = roc_data['tprs_lower']
    tprs_upper = roc_data['tprs_upper']
    plt.plot(mean_fpr, mean_tpr, color='C8', alpha=0.8,
             label=r'Testing (AUC = %0.3f $\pm$ %0.3f)' % (mean_auc, std_auc),
             lw=2)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='C8', alpha=.2,
                     label=r'Testing $\pm$ 1 std. dev.')

    # The rest
    plt.plot([0, 1], [0, 1], linestyle='--', lw=1, color='k', alpha=.8)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.legend(loc=4, prop={'size': 8})
    plt.tight_layout()
    plt.savefig(f'../results/{modality}_roc.pdf')


def plot_model_weights(hide_zero_weights=False):
    support_this = get_support(model)
    weights = get_weights(model)
    labels = X.columns[support_this]
    xticks = np.array(xticks_codes)[support_this]
    labels_fixed = []
    for label in labels:
        if label[-3:] == '_ct':
            label = label[-2:].upper() + ': ' + label[:-2]
            label = label.replace('_', ' ').rstrip()
        if label[-3:] == '_pt':
            label = label[-2:].upper() + ': ' + label[:-2]
            label = label.replace('_', ' ').rstrip()
        labels_fixed.append(label)

    if hide_zero_weights:
        labels_fixed = np.array(labels_fixed)[weights != 0]
        xticks = xticks[weights != 0]
        weights = weights[weights != 0]

    plt.figure(figsize=(3.75, 3.75))
    plt.bar(range(len(weights)), weights, color="C4", edgecolor='C7', alpha=0.8)
    plt.xticks(range(len(weights)), labels_fixed, rotation=90, fontsize=8)
    plt.ylabel('Weight')
    plt.yticks(fontsize=8)
    plt.tight_layout()
    plt.savefig(f'../results/{modality}_weights2.pdf')

    plt.figure(figsize=(3.75, 3.75))
    plt.bar(range(len(weights)), weights, color="C4", edgecolor='C7', alpha=0.8)
    plt.xticks(range(len(weights)), xticks, fontsize=9)
    plt.ylabel('Weight')
    plt.yticks(fontsize=9)
    plt.tight_layout()
    plt.savefig(f'../results/{modality}_weights3.pdf')


def plot_corrmap(hide_zero_weights=False):
    support_this = get_support(model)
    weights = get_weights(model)
    labels = X.columns[support_this]
    xticks = np.array(xticks_codes)[support_this]
    labels_fixed = []
    for label in labels:
        if label[-3:] == '_ct':
            label = label[-2:].upper() + ': ' + label[:-2]
            label = label.replace('_', ' ').rstrip()
        if label[-3:] == '_pt':
            label = label[-2:].upper() + ': ' + label[:-2]
            label = label.replace('_', ' ').rstrip()
        labels_fixed.append(label)

    if hide_zero_weights:
        labels = labels[weights != 0]
        xticks = np.array(xticks_codes)[support_this][weights != 0]
        labels_fixed = np.array(labels_fixed)[weights != 0]

    df = X[labels].corr(method='kendall')
    plt.figure(figsize=(5, 5))
    ax = sns.heatmap(df, annot=True, linewidths=.5, vmin=-1, vmax=1, center=0, annot_kws={"size": 8})
    ax.set_xticklabels(labels_fixed, fontsize=8, rotation=90)
    ax.set_yticklabels(labels_fixed, fontsize=8)
    plt.tight_layout()
    plt.savefig(f'../results/{modality}_cm2.pdf')

    plt.figure(figsize=(3.75, 3.75))
    ax = sns.heatmap(df, annot=True, linewidths=.5, vmin=-1, vmax=1, center=0, annot_kws={"size": 9})
    ax.set_xticklabels(xticks, fontsize=9)
    ax.set_yticklabels(xticks, fontsize=9)
    plt.tight_layout()
    plt.savefig(f'../results/{modality}_cm3.pdf')


if __name__ == "__main__":
    # parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--modality",
        type=str
    )
    args = parser.parse_args()
    modality = args.modality

    # data
    data_red = pd.read_csv('../data/data_red.csv')
    X_ct = pd.read_csv('../data/radiomics_ct_red.csv')
    ct_codes = [f"CT_{e}" for e in range(1, X_ct.columns.shape[0] + 1)]
    X_pt = pd.read_csv('../data/radiomics_pt_red.csv')
    pt_codes = [f"PET_{e}" for e in range(1, X_pt.columns.shape[0] + 1)]
    ptct_codes = ct_codes + pt_codes
    y = data_red['tgr4_hpd1']
    groups = data_red['id.radiomics']

    if modality == 'ct':
        X = X_ct
        xticks_codes = ct_codes
    elif modality == 'pet':
        X = X_pt
        xticks_codes = pt_codes
    elif modality == 'petct':
        X = pd.concat([X_ct, X_pt], axis=1)
        xticks_codes = ptct_codes
    else:
        print("Incorrect modality.")
        exit()

    # parameters
    random_state = 42
    cv_in = RepeatedStratifiedKFold(n_splits=5, n_repeats=20, random_state=random_state)
    cv_out = StratifiedGroupKFold(n_splits=5)
    n_iter = 100
    max_dim = 6

    # model architecture
    if modality == 'ct' or modality == 'pet':
        pipe = Pipeline([('scaler', StandardScaler()),
                         ('fs', SelectKBestFromModel(XGBClassifier(eval_metric='logloss', use_label_encoder=False, random_state=random_state))),
                         ('clf', LogisticRegression(solver='saga', random_state=random_state))
                         ])
        param_dist = {'fs__k': range(1, max_dim + 1),
                      'fs__estimator__n_estimators': [100, 200],
                      'fs__estimator__max_depth': range(1, 11),
                      'fs__estimator__learning_rate': [1e-3, 1e-2, 1e-1, 0.5, 1.],
                      'fs__estimator__subsample': np.arange(0.05, 1.01, 0.05),
                      'fs__estimator__min_child_weight': range(1, 21),
                      'fs__estimator__n_jobs': [1],
                      'clf__C': np.logspace(-5, 10, 100, base=2),
                      'clf__penalty': ['elasticnet'],
                      'clf__l1_ratio': uniform(),
                      'clf__class_weight': [None, 'balanced']}
    elif modality == 'petct':
        ct_pipe = Pipeline([('scaler', StandardScaler()),
                            ('fs', SelectKBestFromModel(XGBClassifier(eval_metric='logloss', use_label_encoder=False, random_state=random_state))),
                            ])
        pt_pipe = Pipeline([('scaler', StandardScaler()),
                            ('fs', SelectKBestFromModel(XGBClassifier(eval_metric='logloss', use_label_encoder=False, random_state=random_state))),
                            ])
        column_transformer = ColumnTransformer([
            ('ct_pipe', ct_pipe, X_ct.columns),
            ('pt_pipe', pt_pipe, X_pt.columns)
        ])
        pipe = Pipeline([('column_transformer', column_transformer),
                         ('clf', LogisticRegression(solver='saga', random_state=random_state))
                         ])
        param_dist = {'column_transformer__ct_pipe__fs__k': range(1, max_dim + 1),
                      'column_transformer__ct_pipe__fs__estimator__n_estimators': [100, 200],
                      'column_transformer__ct_pipe__fs__estimator__max_depth': range(1, 11),
                      'column_transformer__ct_pipe__fs__estimator__learning_rate': [1e-3, 1e-2, 1e-1, 0.5, 1.],
                      'column_transformer__ct_pipe__fs__estimator__subsample': np.arange(0.05, 1.01, 0.05),
                      'column_transformer__ct_pipe__fs__estimator__min_child_weight': range(1, 21),
                      'column_transformer__ct_pipe__fs__estimator__n_jobs': [1],
                      'column_transformer__pt_pipe__fs__k': range(1, max_dim + 1),
                      'column_transformer__pt_pipe__fs__estimator__n_estimators': [100, 200],
                      'column_transformer__pt_pipe__fs__estimator__max_depth': range(1, 11),
                      'column_transformer__pt_pipe__fs__estimator__learning_rate': [1e-3, 1e-2, 1e-1, 0.5, 1.],
                      'column_transformer__pt_pipe__fs__estimator__subsample': np.arange(0.05, 1.01, 0.05),
                      'column_transformer__pt_pipe__fs__estimator__min_child_weight': range(1, 21),
                      'column_transformer__pt_pipe__fs__estimator__n_jobs': [1],
                      'clf__C': np.logspace(-5, 10, 100, base=2),
                      'clf__penalty': ['elasticnet'],
                      'clf__l1_ratio': uniform(),
                      'clf__class_weight': [None, 'balanced']}
    else:
        print("Incorrect modality.")
        exit()

    # run model
    time_start = time.time()
    rs = RandomizedSearchCV(estimator=pipe, param_distributions=param_dist, n_iter=n_iter, scoring='roc_auc', n_jobs=-2,
                            cv=cv_in, random_state=random_state)
    ncv = cross_validate(rs, X, y, groups=groups, scoring='roc_auc', cv=cv_out, n_jobs=1, return_train_score=True,
                         return_estimator=True)
    model = rs.fit(X, y)
    y_proba = rs.predict_proba(X)
    y_proba = pd.DataFrame(y_proba, columns=['class 0', 'class 1'])
    y_proba = pd.merge(left= data_red[['id.radiomics', 'id.lesion', 'lesion.location', 'tgr4_hpd1']], right=y_proba, left_index=True, right_index=True)
    y_proba.to_csv(f'../results/{modality}_ypred.csv', index=None)
    run_time = round(time.time() - time_start)
    print(f'\n\nRuntime = {datetime.timedelta(seconds=run_time)}')

    # visualization
    plot_roc_curves()
    plot_model_weights()
    plot_corrmap()
