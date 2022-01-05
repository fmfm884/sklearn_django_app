import numpy as np
from django import forms
from betterforms.multiform import MultiForm
from django.core.exceptions import ValidationError
import ast


toy_dataset_choice = [
    ('boston', 'ボストンの住宅価格'),
    ('iris', 'アイリス（アヤメ）の種類'),
    ('diabetes', '糖尿病の進行状況'),
    ('digits', '手書き文字（数字）'),
    ('linnerud', '生理学的（physiological）測定結果と運動（exercise）測定結果'),
    ('wine', 'ワインの種類'),
    ('breast_cancer', 'がんの診断結果'),
]

real_world_dataset_choice = [
    ('olivetti_faces', '同一人物の様々な状態の顔画像'),
    ('20newsgroups', 'トピック別のニュース記事'),
    ('20newsgroups_vectorized', 'トピック別のニュース記事の特徴抽出済みバージョン'),
    ('lfw_people', '有名人の顔写真'),
    ('lfw_pairs', '有名人の顔のペア写真'),
    ('covtype', '森林の木の種類'),
    ('rcv1', 'カテゴリ別のニュース（ベクトル化済み）'),
    ('kddcup99', 'ネットワークの侵入検知'),
    ('california_housing', 'カリフォルニアの住宅価格'),
]

img_dataset_choice = [
    ('olivetti_faces', '同一人物の様々な状態の顔画像'),
    ('lfw_people', '有名人の顔写真'),
    ('lfw_pairs', '有名人の顔のペア写真'),
]

dataset_choice = toy_dataset_choice + real_world_dataset_choice

toy_dataset_classification_choice = [
    ('iris', 'アイリス（アヤメ）の種類'),
    ('digits', '手書き文字（数字）'),
    ('wine', 'ワインの種類'),
    ('breast_cancer', 'がんの診断結果'),
]

toy_dataset_regression_choice = [
    ('boston', 'ボストンの住宅価格'),
    ('diabetes', '糖尿病の進行状況'),
    ('linnerud', '生理学的（physiological）測定結果と運動（exercise）測定結果'),
]

dataset_classification_choice = [
    ('iris', 'アイリス（アヤメ）の種類'),
    ('digits', '手書き文字（数字）'),
    ('wine', 'ワインの種類'),
    ('breast_cancer', 'がんの診断結果'),
    ('olivetti_faces', '同一人物の様々な状態の顔画像'),
    ('20newsgroups', 'トピック別のニュース記事'),
    ('20newsgroups_vectorized', 'トピック別のニュース記事の特徴抽出済みバージョン'),
    ('lfw_people', '有名人の顔写真'),
    ('lfw_pairs', '有名人の顔のペア写真'),
    ('covtype', '森林の木の種類'),
    ('rcv1', 'カテゴリ別のニュース（ベクトル化済み）'),
    ('kddcup99', 'ネットワークの侵入検知'),
    ]

dataset_regression_choice = [
    ('boston', 'ボストンの住宅価格'),
    ('diabetes', '糖尿病の進行状況'),
    ('linnerud', '生理学的（physiological）測定結果と運動（exercise）測定結果'),
    ('california_housing', 'カリフォルニアの住宅価格'),
    ]


class DatasetForm(forms.Form):
    dataset = forms.ChoiceField(label='データセット:', choices=dataset_choice)


class DatasetClassificationForm(forms.Form):
    dataset = forms.ChoiceField(label='分類データセット:', choices=dataset_classification_choice)


class ToyDatasetClassificationForm(forms.Form):
    dataset = forms.ChoiceField(label='Toy分類データセット:', choices=toy_dataset_classification_choice)


class DatasetRegressionForm(forms.Form):
    dataset = forms.ChoiceField(label='回帰データセット:', choices=dataset_regression_choice)


class ToyDatasetRegressionForm(forms.Form):
    dataset = forms.ChoiceField(label='Toy回帰データセット:', choices=toy_dataset_regression_choice)


def char_to_int(param):
    try:
        return int(param)
    except ValueError:
        raise ValidationError('Int型を指定してください。')

def char_to_float(param):
    try:
        return float(param)
    except ValueError:
        raise ValidationError('Float型を指定してください。')

def char_to_int_float(param):
    if '.' in param:
        try:
            return float(param)
        except ValueError:
            raise ValidationError('Int型またはFloat型を指定してください。')
    else:
        try:
            return int(param)
        except ValueError:
            raise ValidationError('Int型またはFloat型を指定してください。')


class LogisticForm(forms.Form):
    penalty_choice = [('l2', 'L2'), ('l1', 'L1')]
    class_weight_choice = [('None', None), ('balanced', 'balanced')]
    solver_choice = [
        ('lbfgs', 'lbfgs'), ('newton-cg', 'newton-cg'),
        ('liblinear', 'liblinear'), ('sag', 'sag'),
        ('saga', 'saga')
    ]
    penalty = forms.ChoiceField(label='penalty:', choices=penalty_choice, help_text='LogisticRegression パラメータ')
    tol = forms.FloatField(initial=1e-4, label='tol:')
    C = forms.FloatField(initial=1.0, label='C:')
    class_weight = forms.ChoiceField(label="class_weight:", choices=class_weight_choice)
    solver = forms.ChoiceField(initial='lbfgs', label='solver:', choices=solver_choice)
    random_state = forms.IntegerField(initial=0, label='random_state:')

    def clean(self):
        params = super().clean()
        if params['class_weight'] == 'None':
            params['class_weight'] = None
        return params


class LogisticClassificationMultiForm(MultiForm):
    form_classes = {
        'dataset': ToyDatasetClassificationForm,
        'logistic': LogisticForm,
    }


class KNeighborsForm(forms.Form):
    weights_choice = [('uniform', 'uniform'),('distance', 'distance'),]
    algorithm_choice = [
        ('auto', 'auto'), ('ball_tree', 'ball_tree'),
        ('kd_tree', 'kd_tree'), ('brute', 'brute'),
    ]

    n_neighbors = forms.IntegerField(initial=5, label='n_neighbors:', help_text='KNeighbors パラメータ')
    weights = forms.ChoiceField(label='weights:', choices=weights_choice)
    algorithm = forms.ChoiceField(label='algorithm:', choices=algorithm_choice)
    leaf_size = forms.IntegerField(initial=30, label='leaf_size:')
    p = forms.IntegerField(initial=2, label='p:', min_value=1)
    metric = forms.CharField(initial='minkowski', label='metric:')
    #metric_params = forms.CharField(initial='None', label='metric_params:')
    n_jobs = forms.IntegerField(initial=1, label='n_jogs:')


class KNeighborsClassificationMultiForm(MultiForm):
    form_classes = {
        'dataset': ToyDatasetClassificationForm,
        'k_neighbors': KNeighborsForm,
    }


class SGDClassificationForm(forms.Form):
    loss_choice =[
        ('hinge', 'hinge'), ('log', 'log'),
        ('modified_huber', 'modified_huber'),
        ('squared_hinge', 'squared_hinge'), ('perceptron', 'perceptron'),
        ('squared_epsilon_insensitive', 'squared_epsilon_insensitive')
    ]
    penalty_choice = [('l1', 'L1'), ('l2', 'L2'), ('elasticnet', 'elasticnet')]
    learning_rate_choice = [
        ('constant', 'constant'), ('optimal', 'optimal'),
        ('invscaling', 'invscaling'), ('adaptive', 'adaptive')
    ]
    class_weight_choice = [('None', 'None'), ('balanced', 'balanced')]

    loss = forms.ChoiceField(initial='hinge', label='loss:', choices=loss_choice, help_text='SGD パラメータ')
    penalty = forms.ChoiceField(initial='l2', label='penalty:', choices=penalty_choice)
    alpha = forms.FloatField(initial=0.0001, label='alpha:')
    l1_ratio = forms.FloatField(initial=0.15, label='l1_ratio:')
    fit_intercept = forms.BooleanField(initial=True, label='fit_intercept:', required=False)
    max_iter = forms.IntegerField(initial=1000, label='max_iter:')
    tol = forms.FloatField(initial=1e-3, label='tol:')
    shuffle = forms.BooleanField(initial=True, label='shuffle:', required=False)
    verbose = forms.IntegerField(initial=0, label='verbose:')
    epsilon = forms.FloatField(initial=0.1, label='epsilon:')
    n_jobs = forms.IntegerField(initial=1, label='n_jobs:')
    random_state = forms.IntegerField(initial=0, label='random_state:')
    learning_rate = forms.ChoiceField(initial='optimal', label='learning_rate:', choices=learning_rate_choice)
    eta0 = forms.FloatField(initial=0.0, label='eta0:')
    power_t = forms.FloatField(initial=0.5, label='power_t:')
    early_stopping = forms.BooleanField(initial=False, label='early_stopping:', required=False)
    validation_fraction = forms.FloatField(initial=0.1, label='validation_fraction:')
    n_iter_no_change = forms.IntegerField(initial=5, label='n_iter_no_change:')
    class_weight = forms.ChoiceField(label='class_weight:', choices=class_weight_choice)
    warm_start = forms.BooleanField(initial=False, label='warm_start:', required=False)
    average = forms.BooleanField(initial=False, label='average:', required=False)

    def clean(self):
        params = super().clean()
        if params['class_weight'] == 'None':
            params['class_weight'] = None
        return params


class SGDClassificationMultiForm(MultiForm):
    form_classes = {
        'dataset': ToyDatasetClassificationForm,
        'sgd': SGDClassificationForm,
    }


class SVCForm(forms.Form):
    kernel_choice = [
        ('linear', 'linear'), ('poly', 'poly'),
        ('rbf', 'rbf'), ('sigmoid', 'sigmoid'),
        ('precomputed', 'precomputed'),
    ]
    gamma_choice = [('scale', 'scale'),('auto', 'auto'),]
    class_weight_choice = [('None', 'None'),('balanced', 'balanced')]
    decision_function_shape_choice = [('ovo', 'ovo'),('ovr', 'ovr')]

    C = forms.FloatField(initial=1.0, label='C:', help_text='SVC パラメータ')
    kernel = forms.ChoiceField(initial='rbf', label='kernel:', choices=kernel_choice)
    degree = forms.IntegerField(initial=3, label='degree:')
    gamma = forms.ChoiceField(initial='scale', label='gamma:', choices=gamma_choice)
    coef0 = forms.FloatField(initial=0.0, label='coef0:')
    shrinking = forms.BooleanField(initial=True, label='shrinking:', required=False)
    probability = forms.BooleanField(initial=False, label='probability:', required=False)
    tol = forms.FloatField(initial=1e-3, label='tol:')
    cache_size = forms.FloatField(initial=200, label='cache_size:')
    class_weight = forms.ChoiceField(label='class_weight:', choices=class_weight_choice)
    verbose = forms.BooleanField(initial=False, label='verbose:', required=False)
    max_iter = forms.IntegerField(initial=-1, label='max_iter:', min_value=-1)
    decision_function_shape = forms.ChoiceField(initial='ovr', label='decision_function_shape:', choices=decision_function_shape_choice)
    break_ties = forms.BooleanField(initial=False, label='break_ties:', required=False)
    random_state = forms.IntegerField(initial=0, label='random_state:')

    def clean(self):
        params = super().clean()
        if params['class_weight'] == 'None':
            params['class_weight'] = None
        return params


class SVCMultiForm(MultiForm):
    form_classes = {
        'dataset': ToyDatasetClassificationForm,
        'svc': SVCForm,
    }


class RandomForestClassifierForm(forms.Form):
    criterion_choice = [('gini', 'gini'), ('entropy', 'entropy')]
    max_features_choice = [('auto', 'auto'), ('sqrt', 'sqrt'), ('log2', 'log2')]
    class_weight_choice = [('None', 'None'), ('balanced', 'balanced'), ('balanced_subsample', 'balanced_subsample')]

    n_estimators = forms.IntegerField(initial=100, label='n_estimators:', help_text='RandomFrestClassifier パラメータ')
    criterion = forms.ChoiceField(initial='gini', label='criterion:', choices=criterion_choice)
    max_depth = forms.CharField(initial='None', label='max_depth:', required=False)
    min_samples_split = forms.CharField(initial=2, label='min_samples_split:')
    min_samples_leaf = forms.CharField(initial=1, label='min_samples_leaf:')
    min_weight_fraction_leaf = forms.FloatField(initial=0.0, label='min_weight_fraction_leaf:')
    max_features = forms.ChoiceField(initial='auto', label='max_features:', choices=max_features_choice)
    max_leaf_nodes = forms.CharField(initial='None', label='max_leaf_nodes:')
    min_impurity_decrease = forms.FloatField(initial=0.0, label='min_impurity_decrease:')
    bootstrap = forms.BooleanField(initial=True, label='bootstrap:', required=False)
    oob_score = forms.BooleanField(initial=False, label='oob_score:', required=False)
    n_jobs = forms.CharField(initial='None', label='n_jobs:')
    random_state = forms.IntegerField(initial=0, label='random_state:')
    verbose = forms.IntegerField(initial=0, label='verbose:')
    warm_start = forms.BooleanField(initial=False, label='warm_start:', required=False)
    class_weight = forms.ChoiceField(label='class_weight:', choices=class_weight_choice, required=False)
    ccp_alpha = forms.FloatField(initial=0.0, label='ccp_alpha:', min_value=0.0)
    max_samples = forms.CharField(initial='None', label='max_samples:')

    def clean_max_depth(self):
        max_depth = self.cleaned_data.get('max_depth')
        if max_depth == 'None':
            max_depth = None
        else:
            max_depth = char_to_int(max_depth)
        return max_depth

    def clean_min_samples_split(self):
        min_samples_split = self.cleaned_data.get('min_samples_split')
        return char_to_int_float(min_samples_split)

    def clean_min_samples_leaf(self):
        min_samples_leaf = self.cleaned_data.get('min_samples_leaf')
        return char_to_int_float(min_samples_leaf)

    def clean_max_leaf_nodes(self):
        max_leaf_nodes = self.cleaned_data.get('max_leaf_nodes')
        if max_leaf_nodes == 'None':
            max_leaf_nodes = None
        else:
            max_leaf_nodes = char_to_int(max_leaf_nodes)
        return max_leaf_nodes

    def clean_n_jobs(self):
        n_jobs = self.cleaned_data.get('n_jobs')
        if n_jobs == 'None':
            n_jobs = None
        else:
            n_jobs = char_to_int(n_jobs)
        return n_jobs

    def clean_class_weight(self):
        class_weight = self.cleaned_data.get('class_weight')
        if class_weight == 'None':
            class_weight = None
        return class_weight
    
    def clean_max_samples(self):
        max_samples = self.cleaned_data.get('max_samples')
        if max_samples == 'None':
            max_samples = None
        else:
            max_samples = char_to_int_float(max_samples)
        return max_samples


class RandomForestClassifierMultiForm(MultiForm):
    form_classes = {
        'dataset': ToyDatasetClassificationForm,
        'rfc': RandomForestClassifierForm,
    }


class SGDRegressionForm(forms.Form):
    loss_choice =[
        ('squared_error', 'squared_error'), 
        ('huber','huber'), 
        ('epsilon_insensitive', 'epsilon_insensitive'), 
        ('squared_epsilon_insensitive', 'squared_epsilon_insensitive'),
    ]
    penalty_choice = [('l1', 'L1'), ('l2', 'L2'), ('elasticnet', 'elasticnet')]
    learning_rate_choice = [
        ('constant', 'constant'), ('optimal', 'optimal'),
        ('invscaling', 'invscaling'), ('adaptive', 'adaptive')
    ]
    class_weight_choice = [('None', 'None'), ('balanced', 'balanced')]

    loss = forms.ChoiceField(initial='squared_error', label='loss:', choices=loss_choice, help_text='SGD パラメータ')
    penalty = forms.ChoiceField(initial='l2', label='penalty:', choices=penalty_choice)
    alpha = forms.FloatField(initial=0.0001, label='alpha:')
    l1_ratio = forms.FloatField(initial=0.15, label='l1_ratio:')
    fit_intercept = forms.BooleanField(initial=True, label='fit_intercept:', required=False)
    max_iter = forms.IntegerField(initial=1000, label='max_iter:')
    tol = forms.FloatField(initial=1e-3, label='tol:')
    shuffle = forms.BooleanField(initial=True, label='shuffle:', required=False)
    verbose = forms.IntegerField(initial=0, label='verbose:')
    epsilon = forms.FloatField(initial=0.1, label='epsilon:')
    random_state = forms.IntegerField(initial=0, label='random_state:')
    learning_rate = forms.ChoiceField(initial='invscaling', label='learning_rate:', choices=learning_rate_choice)
    eta0 = forms.FloatField(initial=0.01, label='eta0:')
    power_t = forms.FloatField(initial=0.25, label='power_t:')
    early_stopping = forms.BooleanField(initial=False, label='early_stopping:', required=False)
    validation_fraction = forms.FloatField(initial=0.1, label='validation_fraction:')
    n_iter_no_change = forms.IntegerField(initial=5, label='n_iter_no_change:')
    warm_start = forms.BooleanField(initial=False, label='warm_start:', required=False)
    average = forms.CharField(initial='False', label='average:')

    def clean_average(self):
        average = self.cleaned_data.get('average')
        if average == 'False':
            average = False
        elif average == 'True':
            average = True
        else:
            average = char_to_int(average)
        return average


class SGDRegressionMultiForm(MultiForm):
    form_classes = {
        'dataset': ToyDatasetRegressionForm,
        'sgd': SGDRegressionForm,
    }


class ElasticNetForm(forms.Form):
    selection_choice = [('cyclic', 'cyclic'), ('random', 'random')]

    alpha = forms.FloatField(initial=1.0, label='alpha', help_text='ElasticNet パラメータ')
    l1_ratio = forms.FloatField(initial=0.5, label='l1_ratio')
    fit_intercept = forms.BooleanField(initial=True, label='fit_intercept', required=False)
    normalize = forms.BooleanField(initial=False, label='normalize', required=False)
    precompute = forms.BooleanField(initial=False, label='precompute', required=False)
    max_iter = forms.IntegerField(initial=1000, label='max_iter')
    copy_X = forms.BooleanField(initial=True, label='copy_X', required=False)
    tol = forms.FloatField(initial=1e-4, label='tol')
    warm_start = forms.BooleanField(initial=False, label='warm_start', required=False)
    positive = forms.BooleanField(initial=False, label='positive', required=False)
    random_state = forms.IntegerField(initial=0, label='random_state')
    selection = forms.ChoiceField(initial='cyclic', label='selection', choices=selection_choice)


class ElasticNetMultiForm(MultiForm):
    form_classes = {
        'dataset': ToyDatasetRegressionForm,
        'elasticnet': ElasticNetForm,
    }


class LassoForm(forms.Form):
    selection_choice = [('cyclic', 'cyclic'), ('random', 'random')]

    alpha = forms.FloatField(initial=1.0, label='alpha:', help_text='Lasso パラメータ')
    fit_intercept = forms.BooleanField(initial=True, label='fit_intercept:', required=False)
    normalize = forms.BooleanField(initial=False, label='normalize:', required=False)
    precompute = forms.BooleanField(initial=False, label='precompute:', required=False)
    copy_X = forms.BooleanField(initial=True, label='copy_X:', required=False)
    max_iter = forms.IntegerField(initial=1000, label='max_iter:')
    tol = forms.FloatField(initial=1e-4, label='tol:')
    warm_start = forms.BooleanField(initial=False, label='warm_start:', required=False)
    positive = forms.BooleanField(initial=False, label='positive:', required=False)
    random_state = forms.IntegerField(initial=0, label='random_state:')
    selection = forms.ChoiceField(initial='cyclic', label='selection:', choices=selection_choice)


class LassoMultiForm(MultiForm):
    form_classes = {
        'dataset': ToyDatasetRegressionForm,
        'lasso': LassoForm,
    }


class RidgeForm(forms.Form):
    solver_choice = [
        ('auto', 'auto'), ('svd', 'svd'),
        ('cholesky', 'cholesky'), ('lsqr', 'lsqr'),
        ('sparse_cg', 'sparse_cg'), ('sag', 'sag'),
        ('saga', 'saga'), ('lbfgs', 'lbfgs'),
    ]

    alpha = forms.FloatField(initial=1.0, label='alpha:', help_text='Ridge パラメータ')
    fit_intercept = forms.BooleanField(initial=True, label='fit_intercept:')
    normalize = forms.BooleanField(initial=False, label='normalize:', required=False)
    copy_X = forms.BooleanField(initial=True, label='copy_X:')
    max_iter = forms.CharField(initial='None', label='max_iter:') # initial がダメ
    tol = forms.FloatField(initial=1e-3, label='tol:')
    solver = forms.ChoiceField(initial='auto', label='solver:', choices=solver_choice)
    positive = forms.BooleanField(initial=False, label='positive:', required=False)
    random_state = forms.IntegerField(initial=0, label='random_state:')

    def clean_max_iter(self):
        max_iter = self.cleaned_data.get('max_iter')
        if max_iter == 'None':
            max_iter = None
        else:
            max_iter = char_to_int(max_iter)
        return max_iter


class RidgeMultiForm(MultiForm):
    form_classes = {
        'dataset': ToyDatasetRegressionForm,
        'ridge': RidgeForm,
    }


class SVRForm(forms.Form):
    kernel_choice = [
        ('kernel', 'kernel'), ('poly', 'poly'),
        ('rbf', 'rbf'), ('sigmoid', 'sigmoid'),
        ('precomputed', 'precomputed'),
    ]
    gamma_choice = [('scale', 'scale'), ('auto', 'auto')]

    kernel = forms.ChoiceField(initial='rbf', label='kernel:', choices=kernel_choice, help_text='SVR パラメータ')
    degree = forms.IntegerField(initial=3, label='degree:')
    gamma = forms.ChoiceField(initial='scale', label='gamma:', choices=gamma_choice)
    coef0 = forms.FloatField(initial=0.0, label='coef0:')
    tol = forms.FloatField(initial=1e-3, label='tol:')
    C = forms.FloatField(initial=1.0, label='C:')
    epsilon = forms.FloatField(initial=0.1, label='epsilon:')
    shrinking = forms.BooleanField(initial=True, label='shrinking:', required=False)
    cache_size = forms.FloatField(initial=200, label='cache_size')
    verbose = forms.BooleanField(initial=False, label='verbose', required=False)
    max_iter = forms.IntegerField(initial=-1, label='max_iter:')


class SVRMultiForm(MultiForm):
    form_classes = {
        'dataset': ToyDatasetRegressionForm,
        'svr': SVRForm,
    }


class RandomForestRegressorForm(forms.Form):
    criterion_choice = [
        ('squared_error', 'squared_error'), ('absolute_error', 'absolute_error'),
        ('poisson', 'poisson'),
        ]
    max_features_choice = [('auto', 'auto'), ('sqrt', 'sqrt'), ('log2', 'log2')]

    n_estimators = forms.IntegerField(initial=100, label='n_estimators:', help_text='RandomForestRegressor パラメータ')
    criterion = forms.ChoiceField(initial='squared_error', label='criterion:', choices=criterion_choice)
    max_depth = forms.CharField(initial='None', label='max_depth:')
    min_samples_split = forms.CharField(initial=2, label='min_samples_split:')
    min_samples_leaf = forms.CharField(initial=1, label='min_samples_leaf:')
    min_weight_fraction_leaf = forms.FloatField(initial=0.0, label='min_weight_fraction_leaf:')
    max_features = forms.ChoiceField(initial='auto', label='max_features:', choices=max_features_choice)
    max_leaf_nodes = forms.CharField(initial='None', label='max_leaf_nodes:')
    min_impurity_decrease = forms.FloatField(initial=0.0, label='min_impurity_decrease:')
    bootstrap = forms.BooleanField(initial=True, label='bootstrap:', required=False)
    oob_score = forms.BooleanField(initial=False, label='oob_score:', required=False)
    n_jobs = forms.CharField(initial='None', label='n_jobs:')
    random_state = forms.IntegerField(initial=0, label='random_state:')
    verbose = forms.IntegerField(initial=0, label='verbose:')
    warm_start = forms.BooleanField(initial=False, label='warm_start:', required=False)
    ccp_alpha = forms.FloatField(initial=0.0, label='ccp_alpha:')
    max_samples = forms.CharField(initial='None', label='max_samples:')

    def clean_max_depth(self):
        max_depth = self.cleaned_data.get('max_depth')
        if max_depth == 'None':
            max_depth = None
        else:
            max_depth = char_to_int(max_depth)
        return max_depth

    def clean_min_samples_split(self):
        min_samples_split = self.cleaned_data.get('min_samples_split')
        return char_to_int_float(min_samples_split)

    def clean_min_samples_leaf(self):
        min_samples_leaf = self.cleaned_data.get('min_samples_leaf')
        return char_to_int_float(min_samples_leaf)

    def clean_max_leaf_nodes(self):
        max_leaf_nodes = self.cleaned_data.get('max_leaf_nodes')
        if max_leaf_nodes == 'None':
            max_leaf_nodes = None
        else:
            max_leaf_nodes = char_to_int(max_leaf_nodes)
        return max_leaf_nodes

    def clean_n_jobs(self):
        n_jobs = self.cleaned_data.get('n_jobs')
        if n_jobs == 'None':
            n_jobs = None
        else:
            n_jobs = char_to_int(n_jobs)
        return n_jobs

    def clean_max_samples(self):
        max_samples = self.cleaned_data.get('max_samples')
        if max_samples == 'None':
            max_samples = None
        else:
            max_samples = char_to_int_float(max_samples)
        return max_samples


class RandomForestRegressorMultiForm(MultiForm):
    form_classes = {
        'dataset': ToyDatasetRegressionForm,
        'rfr': RandomForestRegressorForm,
    }


class PCAForm(forms.Form):
    svd_solver_choice = [
        ('auto', 'auto'), ('full', 'full'),
        ('arpack', 'arpack'), ('randomized', 'randomized'),
    ]

    n_components=forms.CharField(initial='None', label='n_components:', help_text='PCA パラメータ')
    copy = forms.BooleanField(initial=True, label='copy', required=False)
    whiten = forms.BooleanField(initial=False, label='whiten', required=False)
    svd_solver = forms.ChoiceField(initial='auto', label='svd_solver', choices=svd_solver_choice)
    tol = forms.FloatField(initial=0.0, label='tol')
    iterated_power = forms.CharField(initial='auto', label='iterated_power')
    random_state = forms.IntegerField(initial=0, label='random_state')

    def clean_n_components(self):
        n_components = self.cleaned_data.get('n_components')
        if n_components == 'None':
            n_components = None
        elif n_components == 'mle':
            pass
        else:
            n_components = char_to_int_float(n_components)
        return n_components

    def clean_iterated_power(self):
        iterated_power = self.cleaned_data.get('iterated_power')
        if iterated_power == 'auto':
            pass
        else:
            iterated_power = char_to_int(iterated_power)
        return iterated_power


class PCAMultiForm(MultiForm):
    form_classes = {
        'dataset': ToyDatasetClassificationForm,
        'pca': PCAForm,
    }


class KernelPCAForm(forms.Form):
    kernel_choice = [
        ('linear', 'linear'), ('poly', 'poly'), ('rbf', 'rbf'), ('sigmoid', 'sigmoid'), 
        ('cosine', 'cosine'), ('precomputed', 'precomputed')
    ]
    eigen_solver_choice = [
        ('auto', 'auto'), ('dense', 'dense'), ('arpack', 'arpack'), ('randomized', 'randomized'),
    ]

    n_components = forms.CharField(initial='None', label='n_components:', required=False, help_text='Kernel PCA パラメータ')
    kernel = forms.ChoiceField(initial='linear', label='kernel:', choices=kernel_choice)
    gamma = forms.CharField(initial='None', label='gamma:')
    degree = forms.IntegerField(initial=3, label='degree:')
    coef0 = forms.FloatField(initial=1, label='coef0')
    #kernel_params = forms.
    alpha = forms.FloatField(initial=1.0, label='alpha:')
    fit_inverse_transform = forms.BooleanField(initial=False, label='fit_inverse_transform:', required=False)
    eigen_solver = forms.ChoiceField(initial='auto', label='eigen_solver:', choices=eigen_solver_choice)
    tol = forms.FloatField(initial=0, label='tol:')
    max_iter = forms.CharField(initial='None', label='max_iter:')
    iterated_power = forms.CharField(initial='auto', label='iterated_power:')
    remove_zero_eig = forms.BooleanField(initial=False, label='remove_zero_eig:', required=False)
    random_state = forms.IntegerField(initial=0, label='random_state:')
    copy_X = forms.BooleanField(initial=True, label='copy_X:', required=False)
    n_jobs = forms.CharField(initial='None', label='n_jobs:')

    def clean_n_components(self):
        n_components = self.cleaned_data.get('n_components')
        if n_components == 'None':
            n_components = None
        else:
            n_components = char_to_int(n_components)
        return n_components

    def clean_gamma(self):
        gamma = self.cleaned_data.get('gamma')
        if gamma == 'None':
            gamma = None
        else:
            gamma = char_to_float(gamma)
        return gamma

    def clean_max_iter(self):
        max_iter = self.cleaned_data.get('max_iter')
        if max_iter == 'None':
            max_iter = None
        else:
            max_iter = char_to_int(max_iter)
        return max_iter

    def clean_iterated_power(self):
        iterated_power = self.cleaned_data.get('iterated_power')
        if iterated_power == 'auto':
            pass
        elif '-' in iterated_power:
            raise ValidationError('0以上の値を指定してください。')
        else:
            char_to_int(iterated_power)
        return iterated_power

    def clean_n_jobs(self):
        n_jobs = self.cleaned_data.get('n_jobs')
        if n_jobs == 'None':
            n_jobs = None
        else:
            n_jobs = char_to_int(n_jobs)
        return n_jobs


class KernelPCAMultiForm(MultiForm):
    form_classes = {
        'dataset': ToyDatasetClassificationForm,
        'kpca': KernelPCAForm,
    }



class TruncatedSVDForm(forms.Form):
    algorithm_choice = [('arpack', 'arpack'), ('randomized', 'randomized')]

    n_components = forms.IntegerField(initial=2, label='n_components:', help_text='TruncatedSVD パラメータ')
    algorithm = forms.ChoiceField(initial='randomized', label='algorithm:', choices=algorithm_choice)
    n_iter = forms.IntegerField(initial=5, label='n_iter:')
    random_state = forms.IntegerField(initial=0, label='random_state:')
    tol = forms.FloatField(initial=0.0, label='tol:')


class TruncatedSVDMultiForm(MultiForm):
    form_classes = {
        'dataset': ToyDatasetClassificationForm,
        'truncatedsvd': TruncatedSVDForm,
    }


class KMeansForm(forms.Form):
    init_choice = [('k-means++', 'k-means++'), ('random', 'random')]
    algorithm_choice = [('auto', 'auto'), ('full', 'full'), ('elkan', 'elkan')]

    n_clusters = forms.IntegerField(initial=8, label='n_clusters:', help_text='KMeans パラメータ')
    init = forms.ChoiceField(initial='k-means++', label='init:', choices=init_choice)
    n_init = forms.IntegerField(initial=10, label='n_init:')
    max_iter = forms.IntegerField(initial=300, label='max_iter:')
    tol = forms.FloatField(initial=1e-4, label='tol:')
    verbose = forms.IntegerField(initial=0, label='verbose:')
    random_state = forms.IntegerField(initial=0, label='random_state:')
    copy_x = forms.BooleanField(initial=True, label='copy_x', required=False)
    algorithm = forms.ChoiceField(initial='auto', label='algorithm:', choices=algorithm_choice)


class MeanShiftForm(forms.Form):
    bandwidth = forms.CharField(initial='None', label='bandwidth:', help_text='MeanShift パラメータ')
    seeds = forms.CharField(initial='None', label='seeds:')
    bin_seeding = forms.BooleanField(initial=False, label='bin_seeding:', required=False)
    min_bin_freq = forms.IntegerField(initial=1, label='min_bin_freq:')
    cluster_all = forms.BooleanField(initial=True, label='cluster_all:', required=False)
    n_jobs = forms.CharField(initial='None', label='n_jobs:')
    max_iter = forms.IntegerField(initial=300, label='max_iter:')

    def clean_bandwidth(self):
        bandwidth = self.cleaned_data.get('bandwidth')
        if bandwidth == 'None':
            bandwidth = None
        else:
            bandwidth = char_to_float(bandwidth)
        return bandwidth

    def clean_seeds(self):
        seeds = self.cleaned_data.get('seeds')
        if seeds == 'None':
            seeds = None
        return seeds

    def clean_n_jobs(self):
        n_jobs = self.cleaned_data.get('n_jobs')
        if n_jobs == 'None':
            n_jobs = None
        else:
            n_jobs = char_to_int(n_jobs)
        return n_jobs


class VBGMMForm(forms.Form):
    covariance_type_choice = [
        ('spherical', 'spherical'), ('tied', 'tied'),
        ('diag', 'diag'), ('full', 'full'),
    ]
    init_params_choice = [('kmeans', 'kmeans'), ('random', 'random')]
    weight_concentration_prior_type_choice = [('dirichlet_process', 'dirichlet_process'), ('dirichlet_distribution', 'dirichlet_distribution')]

    n_components = forms.IntegerField(initial=1, label='n_components:', help_text='VBGMM パラメータ')
    covariance_type = forms.ChoiceField(initial='diag', label='covariance:', choices=covariance_type_choice)
    tol = forms.FloatField(initial=1e-3, label='tol:')
    reg_covar = forms.FloatField(initial=1e-6, label='reg_covar:')
    max_iter = forms.IntegerField(initial=100, label='max_iter:')
    n_init = forms.IntegerField(initial=1, label='n_init:')
    init_params = forms.ChoiceField(initial='kmeans', label='init_params:', choices=init_params_choice)
    weight_concentration_prior_type = forms.ChoiceField(initial='dirichlet_process', label='weight_concentration_prior_type:', choices=weight_concentration_prior_type_choice)
    weight_concentration_prior = forms.CharField(initial='None', label='weight_concentration_prior:')
    mean_precision_prior = forms.CharField(initial='None', label='mean_precision_prior:')
    mean_prior = forms.CharField(initial='None', label='mean_prior:')
    degrees_of_freedom_prior = forms.CharField(initial='None', label='covariance_prior:')
    covariance_prior = forms.CharField(initial='None', label='covariance_prior:')
    random_state = forms.IntegerField(initial=0, label='random_state:')
    warm_start = forms.BooleanField(initial=False, label='warm_start:', required=False)
    verbose = forms.IntegerField(initial=0, label='verbose:')
    verbose_interval = forms.IntegerField(initial=10, label='verbose_interval:')

    def clean_weight_concentration_prior(self):
        weight_concentration_prior = self.cleaned_data.get('weight_concentration_prior')
        if weight_concentration_prior == 'None':
            weight_concentration_prior = None
        else:
            weight_concentration_prior = char_to_float(weight_concentration_prior)
        return weight_concentration_prior

    def clean_mean_precision_prior(self):
        mean_precision_prior = self.cleaned_data.get('mean_precision_prior')
        if mean_precision_prior == 'None':
            mean_precision_prior = None
        else:
            mean_precision_prior = char_to_float(mean_precision_prior)
        return mean_precision_prior

    def clean_mean_prior(self):
        mean_prior = self.cleaned_data.get('mean_prior')
        if mean_prior == 'None':
            mean_prior = None
        else:
            mean_prior = char_to_float(mean_prior)
        return mean_prior

    def clean_degrees_of_freedom_prior(self):
        degrees_of_freedom_prior = self.cleaned_data.get('degrees_of_freedom_prior')
        if degrees_of_freedom_prior == 'None':
            degrees_of_freedom_prior = None
        else:
            degrees_of_freedom_prior = char_to_float(degrees_of_freedom_prior)
        return degrees_of_freedom_prior

    def clean_covariance_prior(self):
        covariance_prior = self.cleaned_data.get('covariance_prior')
        if covariance_prior == 'None':
            covariance_prior = None
        else:
            covariance_prior = char_to_float(covariance_prior)
        return covariance_prior


class SpectralClusteringForm(forms.Form):
    eigen_solver_choice = [
        ('None', 'None'), ('arpack', 'arpack'), ('lobpcg', 'lobpcg'), ('amg', 'amg'),
    ]
    affinity_choice = [
        ('nearest_neighbors', 'nearest_neighbors'), ('rbf', 'rbf'),
        ('precomputed', 'precomputed'), ('precomputed_nearest_neighbors', 'precomputed_nearest_neighbors'),
    ]
    assign_labels_choice = [('kmeans', 'kmeans'), ('discretize', 'discretize')]

    n_clusters = forms.IntegerField(initial=8, label='n_clusters:', help_text='SpectralClustering パラメータ')
    eigen_solver = forms.ChoiceField(initial='None', label='eigen_solver:', choices=eigen_solver_choice)
    n_components = forms.IntegerField(initial=n_clusters.initial, label='n_components:')
    random_state = forms.IntegerField(initial=0, label='random_state:')
    n_init = forms.IntegerField(initial=10, label='n_init:')
    gamma = forms.FloatField(initial=1.0, label='gamma:')
    affinity = forms.ChoiceField(initial='rbf', label='affinity:', choices=affinity_choice)
    n_neighbors = forms.IntegerField(initial=10, label='n_neighbors:')
    eigen_tol = forms.FloatField(initial=0.0, label='eigen_tol:')
    assign_labels = forms.ChoiceField(initial='kmeans', label='assign_labels:', choices=assign_labels_choice)
    degree = forms.FloatField(initial=3, label='degree:')
    coef0 = forms.FloatField(initial=1, label='coef0:')
    kernel_params = forms.CharField(initial='None', label='kernel_params:')
    n_jobs = forms.CharField(initial='None', label='n_jobs:')
    verbose = forms.BooleanField(initial=False, label='verbose:', required=False)

    def clean_eigen_solver(self):
        eigen_solver = self.cleaned_data.get('eigen_solver')
        if eigen_solver == 'None':
            eigen_solver = None

    def clean_kernel_params(self):
        kernel_params = self.cleaned_data.get('kernel_params')
        if kernel_params == 'None':
            kernel_params = None

    def clean_n_jobs(self):
        n_jobs = self.cleaned_data.get('n_jobs')
        if n_jobs == 'None':
            n_jobs = None
        else:
            n_jobs = char_to_int(n_jobs)
        return n_jobs


class GaussianMixtureForm(forms.Form):
    covariance_type_choice = [
        ('full', 'full'), ('tied', 'tied'),
        ('diag', 'diag'), ('spherical', 'spherical'),
    ]
    init_params_choice = [('kmeans', 'kmeans'), ('random', 'random'),]
    precisions_init_choice = [
        ('spherical', 'spherical'), ('tied', 'tied'),
        ('diag', 'diag'), ('full','full'),
    ]

    n_components = forms.IntegerField(initial=1, label='n_components:', help_text='GaussianMixture パラメータ')
    covariance_type = forms.ChoiceField(initial='full', label='covariance_type:', choices=covariance_type_choice)
    tol = forms.FloatField(initial=1e-3, label='tol:')
    reg_covar = forms.FloatField(initial=1e-6, label='reg_covar:')
    max_iter = forms.IntegerField(initial=100, label='max_iter:')
    n_init = forms.IntegerField(initial=1, label='n_init:')
    init_params = forms.ChoiceField(initial='kmeans', label='init_params:', choices=init_params_choice)
    weights_init = forms.CharField(initial='None', label='weight_init:')
    means_init = forms.CharField(initial='None', label='means_init:')
    precisions_init = forms.ChoiceField(initial='None', label='precisions_init:', choices=precisions_init_choice)
    random_state = forms.IntegerField(initial=0, label='random_state:')
    warm_start = forms.BooleanField(initial=False, label='warm_start:', required=False)
    verbose = forms.IntegerField(initial=0, label='verbose:')
    verbose_interval = forms.IntegerField(initial=10, label='verbose_interval:')

    def clean_weights_init(self):
        weights_init = self.cleaned_data.get('weights_init')
        if weights_init == 'None':
            weights_init = None
        else:
            weights_init = char_to_int(weights_init)
        return weights_init

    def clean_means_init(self):
        means_init = self.cleaned_data.get('means_init')
        if means_init == 'None':
            means_init = None
        else:
            means_init = char_to_int(means_init)
        return means_init

    def clean_precisions_init(self):
        precisions_init = self.cleaned_data.get('precisions_init')
        if precisions_init == 'None':
            precisions_init = None


class KMeansPCAMultiForm(MultiForm):
    form_classes = {
        'dataset': ToyDatasetClassificationForm,
        'kmeans': KMeansForm,
        'pca': PCAForm,
    }


class MeanShiftPCAMultiForm(MultiForm):
    form_classes = {
        'dataset': ToyDatasetClassificationForm,
        'meanshift': MeanShiftForm,
        'pca': PCAForm,
    }


class VBGMMPCAMultiForm(MultiForm):
    form_classes = {
        'dataset': ToyDatasetClassificationForm,
        'vbgmm': VBGMMForm,
        'pca': PCAForm,
    }


class SpectralClusteringPCAMultiForm(MultiForm):
    form_classes = {
        'dataset': ToyDatasetClassificationForm,
        'spectralclustering': SpectralClusteringForm,
        'pca': PCAForm,
    }


class GaussianMixturePCAMultiForm(MultiForm):
    form_classes = {
        'dataset': ToyDatasetClassificationForm,
        'gmm': GaussianMixtureForm,
        'pca': PCAForm,
    }


class GridSearchCVForm(forms.Form):
    verbose_choice = [('1', 1), ('2', 2), ('3', 3)]

    scoring = forms.CharField(initial='None', label='scoring:', help_text='GridSearchCV パラメータ')
    n_jobs = forms.CharField(initial=1, label='n_jobs:')
    refit = forms.BooleanField(initial=True, label='refit:', required=False)
    cv = forms.CharField(initial='None', label='cv:')
    verbose = forms.ChoiceField(initial='1', label='verbose:', choices=verbose_choice)
    pre_dispatch = forms.IntegerField(initial=2*n_jobs.initial, label='pre_dispatch:')
    error_score = forms.CharField(initial=np.nan, label='error_score:')
    return_train_score = forms.BooleanField(initial=False, label='return_train_score:', required=False)
    estimator = forms.CharField(initial='SVC', label='estimator:', disabled=True)
    param_grid = forms.CharField(initial='{}', label='param_grid:', widget=forms.Textarea, help_text='msg:')

    def clean_scoring(self):
        scoring = self.cleaned_data.get('scoring')
        if scoring == 'None':
            scoring = None

    def clean_n_jobs(self):
        n_jobs = self.cleaned_data.get('n_jobs')
        if n_jobs == 'None':
            n_jobs = None
        else:
            n_jobs = char_to_int(n_jobs)
        return n_jobs

    def clean_cv(self):
        cv = self.cleaned_data.get('cv')
        if cv == 'None':
            cv = None
        else:
            cv = char_to_int(cv)
        return cv

    def clean_verbose(self):
        verbose = self.cleaned_data.get('verbose')
        verbose = char_to_int(verbose)
        return verbose

    def clean_error_score(self):
        error_score = self.cleaned_data.get('error_score')
        if error_score == 'nan':
            error_score = np.nan
        elif error_score == 'raise':
            pass
        else:
            error_score = char_to_int(error_score)
        return error_score

    def clean_param_grid(self):
        param_grid = self.cleaned_data.get('param_grid')
        try:
            param_grid = ast.literal_eval(param_grid)
        except:
            raise ValidationError('入力が正しくありません。')
        return param_grid


class GridSearchCVMultiForm(MultiForm):
    form_classes = {
        'dataset': ToyDatasetClassificationForm,
        'gridsearchcv': GridSearchCVForm,
    }


class RandomizedSearchCVForm(forms.Form):
    verbose_choice = [('1', 1), ('2', 2), ('3', 3)]

    n_iter = forms.IntegerField(initial=10, label='n_iter:', help_text='RandomizedSearchCV パラメータ')
    scoring = forms.CharField(initial='None', label='scoring:')
    n_jobs = forms.CharField(initial='None', label='n_jobs:')
    refit = forms.BooleanField(initial=True, label='refit:', required=False)
    cv = forms.IntegerField(initial=5, label='cv:')
    verbose = forms.ChoiceField(initial='1', label='verbose:', choices=verbose_choice)
    pre_dispatch = forms.IntegerField(initial=2*1, label='pre_dispatch:')
    random_state = forms.IntegerField(initial=0, label='random_state:')
    error_score = forms.CharField(initial=np.nan, label='error_score:')
    return_train_score = forms.BooleanField(initial=False, label='return_train_score:', required=False)
    estimator = forms.CharField(initial='SVC', label='estimator:', disabled=True)
    param_distributions = forms.CharField(initial='{}', label='param_distributions:', widget=forms.Textarea, help_text='msg:')

    def clean_scoring(self):
        scoring = self.cleaned_data.get('scoring')
        if scoring == 'None':
            scoring = None

    def clean_n_jobs(self):
        n_jobs = self.cleaned_data.get('n_jobs')
        if n_jobs == 'None':
            n_jobs = None
        else:
            n_jobs = char_to_int(n_jobs)
        return n_jobs

    def clean_cv(self):
        cv = self.cleaned_data.get('cv')
        if cv == 'None':
            cv = None
        else:
            cv = char_to_int(cv)
        return cv

    def clean_verbose(self):
        verbose = self.cleaned_data.get('verbose')
        verbose = char_to_int(verbose)
        return verbose

    def clean_error_score(self):
        error_score = self.cleaned_data.get('error_score')
        if error_score == 'nan':
            error_score = np.nan
        elif error_score == 'raise':
            pass
        else:
            error_score = char_to_int(error_score)
        return error_score

    def clean_param_distributions(self):
        param_distributions = self.cleaned_data.get('param_distributions')
        try:
            param_distributions = ast.literal_eval(param_distributions)
        except:
            raise ValidationError('入力が正しくありません。')
        return param_distributions


class RandomizedSearchCVMultiForm(MultiForm):
    form_classes = {
        'dataset': ToyDatasetClassificationForm,
        'randomizedsearchcv': RandomizedSearchCVForm,
    }


class EstimatorForm(forms.Form):
    estimator_choice = [
        ('Logistic Regression', 'Logistic Regression'), ('k-neighbors', 'k-neighbors'),
        ('SGD Classifier', 'SGD Classifier'), ('SVC', 'SVC'), ('RandomForestClassifier', 'RandomForestClassifier'),
        ('SGD Regressor', 'SGD Regressor'), ('ElasticNet', 'ElasticNet'), ('Lasso', 'Lasso'),
        ('Ridge', 'Ridge'), ('SVR', 'SVR'), ('RandomForestRegressor', 'RandomForestRegressor'),
    ]

    estimator = forms.ChoiceField(label='estimator:', choices=estimator_choice)


class EstimatorMultiForm(MultiForm):
    form_classes = {
        'dataset': ToyDatasetClassificationForm,
        'estimator': EstimatorForm, 
    }


