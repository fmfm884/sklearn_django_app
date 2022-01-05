from types import prepare_class
from sklearn.linear_model import LogisticRegression, SGDClassifier, SGDRegressor, ElasticNet, Lasso, Ridge
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, SVR
from sklearn.cluster import KMeans, MeanShift, SpectralClustering
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture
from sklearn.decomposition import PCA, KernelPCA, TruncatedSVD
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


class LogisticRegression(LogisticRegression):
    def __init__(self, params):
        super().__init__(
            penalty=params['penalty'],
            tol=params['tol'],
            C=params['C'], 
            class_weight=params['class_weight'],
            solver=params['solver'], 
            random_state=params['random_state']
        )


class KNeighborsClassifier(KNeighborsClassifier):
    def __init__(self, params):
        super().__init__(
            n_neighbors=params['n_neighbors'], weights=params['weights'],
            algorithm=params['algorithm'], leaf_size=params['leaf_size'],
            p=params['p'], metric=params['metric'],
            #metric_params=params['metric_params'], 
            n_jobs=params['n_jobs']
        )


class SGDClassifier(SGDClassifier):
    def __init__(self, params):
        super().__init__(
            loss=params['loss'],
            penalty=params['penalty'], 
            alpha=params['alpha'],
            l1_ratio=params['l1_ratio'],
            fit_intercept=params['fit_intercept'],
            max_iter=params['max_iter'],
            tol=params['tol'],
            shuffle=params['shuffle'],
            verbose=params['verbose'],
            epsilon=params['epsilon'],
            n_jobs=params['n_jobs'],
            random_state=params['random_state'],
            learning_rate=params['learning_rate'],
            eta0=params['eta0'],
            power_t=params['power_t'],
            early_stopping=params['early_stopping'],
            validation_fraction=params['validation_fraction'],
            n_iter_no_change=params['n_iter_no_change'],
            class_weight=params['class_weight'],
            warm_start=params['warm_start'],
            average=params['average']
        )


class SVC(SVC):
    def __init__(self, params):
        super().__init__(
            C=params['C'], 
            kernel=params['kernel'],
            degree=params['degree'], 
            gamma=params['gamma'],
            coef0=params['coef0'], 
            shrinking=params['shrinking'],
            probability=params['probability'], 
            tol=params['tol'],
            cache_size=params['cache_size'], 
            class_weight=params['class_weight'],
            verbose=params['verbose'], 
            max_iter=params['max_iter'],
            decision_function_shape=params['decision_function_shape'],
            break_ties=params['break_ties'],
            random_state=params['random_state']
        )


class RandomForestClassifier(RandomForestClassifier):
    def __init__(self, params):
        super().__init__(
            n_estimators=params['n_estimators'],
            criterion=params['criterion'],
            max_depth=params['max_depth'],
            min_samples_split=params['min_samples_split'],
            min_samples_leaf=params['min_samples_leaf'],
            min_weight_fraction_leaf=params['min_weight_fraction_leaf'],
            max_features=params['max_features'],
            max_leaf_nodes=params['max_leaf_nodes'],
            min_impurity_decrease=params['min_impurity_decrease'],
            bootstrap=params['bootstrap'],
            oob_score=params['oob_score'],
            n_jobs=params['n_jobs'],
            random_state=params['random_state'],
            verbose=params['verbose'],
            warm_start=params['warm_start'],
            class_weight=params['class_weight'],
            ccp_alpha=params['ccp_alpha'],
            max_samples=params['max_samples']
        )


class SGDRegressor(SGDRegressor):
    def __init__(self, params):
        super().__init__(
            loss=params['loss'], 
            penalty=params['penalty'],
            alpha=params['alpha'], 
            l1_ratio=params['l1_ratio'],
            fit_intercept=params['fit_intercept'], 
            max_iter=params['max_iter'],
            tol=params['tol'], 
            shuffle=params['shuffle'],
            verbose=params['verbose'], 
            epsilon=params['epsilon'],
            random_state=params['random_state'], 
            learning_rate=params['learning_rate'],
            eta0=params['eta0'], 
            power_t=params['power_t'],
            early_stopping=params['early_stopping'], 
            validation_fraction=params['validation_fraction'],
            n_iter_no_change=params['n_iter_no_change'], 
            warm_start=params['warm_start'],
            average=params['average']
        )


class ElasticNet(ElasticNet):
    def __init__(self, params):
        super().__init__(
            alpha=params['alpha'], l1_ratio=params['l1_ratio'],
            fit_intercept=params['fit_intercept'], normalize=params['normalize'],
            precompute=params['precompute'], max_iter=params['max_iter'],
            copy_X=params['copy_X'], tol=params['tol'],
            warm_start=params['warm_start'], positive=params['positive'],
            random_state=params['random_state'], selection=params['selection'],
        )

class Lasso(Lasso):
    def __init__(self, params):
        super().__init__(
            alpha=params['alpha'], fit_intercept=params['fit_intercept'],
            normalize=params['normalize'], precompute=params['precompute'],
            copy_X=params['copy_X'], max_iter=params['max_iter'],
            tol=params['tol'], warm_start=params['warm_start'],
            positive=params['positive'], random_state=params['random_state'],
            selection=params['selection']
        )


class Ridge(Ridge):
    def __init__(self, params):
        super().__init__(
            alpha=params['alpha'],
            fit_intercept=params['fit_intercept'],
            normalize=params['normalize'],
            copy_X=params['copy_X'],
            max_iter=params['max_iter'],
            tol=params['tol'],
            solver=params['solver'],
            positive=params['positive'],
            random_state=params['random_state'],
        )



class SVR(SVR):
    def __init__(self, params):
        super().__init__(
            kernel=params['kernel'],
            degree=params['degree'],
            gamma=params['gamma'],
            coef0=params['coef0'],
            tol=params['tol'],
            C=params['C'],
            epsilon=params['epsilon'],
            shrinking=params['shrinking'],
            cache_size=params['cache_size'],
            verbose=params['verbose'],
            max_iter=params['max_iter'],
        )


class RandomForestRegressor(RandomForestRegressor):
    def __init__(self, params):
        super().__init__(
            n_estimators=params['n_estimators'],
            criterion=params['criterion'],
            max_depth=params['max_depth'],
            min_samples_split=params['min_samples_split'],
            min_samples_leaf=params['min_samples_leaf'],
            min_weight_fraction_leaf=params['min_weight_fraction_leaf'],
            max_features=params['max_features'],
            max_leaf_nodes=params['max_leaf_nodes'],
            min_impurity_decrease=params['min_impurity_decrease'],
            bootstrap=params['bootstrap'],
            oob_score=params['oob_score'],
            n_jobs=params['n_jobs'],
            random_state=params['random_state'],
            verbose=params['verbose'],
            warm_start=params['warm_start'],
            ccp_alpha=params['ccp_alpha'],
            max_samples=params['max_samples']
        )


class PCA(PCA):
    def __init__(self, params):
        super().__init__(
            n_components=params['n_components'],
            copy=params['copy'],
            whiten=params['whiten'],
            svd_solver=params['svd_solver'],
            tol=params['tol'],
            iterated_power=params['iterated_power'],
            random_state=params['random_state']
        )


class KernelPCA(KernelPCA):
    def __init__(self, params):
        super().__init__(
            n_components=params['n_components'],
            kernel=params['kernel'],
            gamma=params['gamma'],
            degree=params['degree'],
            coef0=params['coef0'],
            #kernel_params=params['kernel_params'],
            alpha=params['alpha'],
            fit_inverse_transform=params['fit_inverse_transform'],
            eigen_solver=params['eigen_solver'],
            tol=params['tol'],
            max_iter=params['max_iter'],
            iterated_power=params['iterated_power'],
            remove_zero_eig=params['remove_zero_eig'],
            random_state=params['random_state'],
            copy_X=params['copy_X'],
            n_jobs=params['n_jobs']
        )



class TruncatedSVD(TruncatedSVD):
    def __init__(self, params):
        super().__init__(
            n_components=params['n_components'],
            algorithm=params['algorithm'],
            n_iter=params['n_iter'],
            random_state=params['random_state'],
            tol=params['tol']
        )


class KMeans(KMeans):
    def __init__(self, params):
        super().__init__(
            n_clusters=params['n_clusters'],
            init=params['init'],
            n_init=params['n_init'],
            max_iter=params['max_iter'],
            tol=params['tol'],
            verbose=params['verbose'],
            random_state=params['random_state'],
            copy_x=params['copy_x'],
            algorithm=params['algorithm']
        )


class MeanShift(MeanShift):
    def __init__(self, params):
        super().__init__(
            bandwidth=params['bandwidth'],
            seeds=params['seeds'],
            bin_seeding=params['bin_seeding'],
            min_bin_freq=params['min_bin_freq'],
            cluster_all=params['cluster_all'],
            n_jobs=params['n_jobs'],
            max_iter=params['max_iter']
        )


class VBGMM(BayesianGaussianMixture):
    def __init__(self, params):
        super().__init__(
            n_components=params['n_components'],
            covariance_type=params['covariance_type'],
            tol=params['tol'],
            reg_covar=params['reg_covar'],
            max_iter=params['max_iter'],
            n_init=params['n_init'],
            init_params=params['init_params'],
            weight_concentration_prior_type=params['weight_concentration_prior_type'],
            weight_concentration_prior=params['weight_concentration_prior'],
            mean_precision_prior=params['mean_precision_prior'],
            mean_prior=params['mean_prior'],
            degrees_of_freedom_prior=params['degrees_of_freedom_prior'],
            covariance_prior=params['covariance_prior'],
            random_state=params['random_state'],
            warm_start=params['warm_start'],
            verbose=params['verbose'],
            verbose_interval=params['verbose_interval']
        )


class SpectralClustering(SpectralClustering):
    def __init__(self, params):
        super().__init__(
            n_clusters=params['n_clusters'],
            eigen_solver=params['eigen_solver'],
            n_components=params['n_components'],
            random_state=params['random_state'],
            n_init=params['n_init'],
            gamma=params['gamma'],
            affinity=params['affinity'],
            n_neighbors=params['n_neighbors'],
            eigen_tol=params['eigen_tol'],
            assign_labels=params['assign_labels'],
            degree=params['degree'],
            coef0=params['coef0'],
            kernel_params=params['kernel_params'],
            n_jobs=params['n_jobs'],
            verbose=params['verbose']
        )


class GaussianMixture(GaussianMixture):
    def __init__(self, params):
        super().__init__(
            n_components=params['n_components'],
            covariance_type=params['covariance_type'],
            tol=params['tol'],
            reg_covar=params['reg_covar'],
            max_iter=params['max_iter'],
            n_init=params['n_init'],
            init_params=params['init_params'],
            weights_init=params['weights_init'],
            means_init=params['means_init'],
            precisions_init=params['precisions_init'],
            random_state=params['random_state'],
            warm_start=params['warm_start'],
            verbose=params['verbose'],
            verbose_interval=params['verbose_interval']
        )


class GridSearchCV(GridSearchCV):
    def __init__(self, params):
        super().__init__(
            estimator=params['estimator'],
            param_grid=params['param_grid'],
            scoring=params['scoring'],
            n_jobs=params['n_jobs'],
            refit=params['refit'],
            cv=params['cv'],
            verbose=params['verbose'],
            pre_dispatch=params['pre_dispatch'],
            #error_score=params['error_score'],
            return_train_score=params['return_train_score']
        )


class RandomizedSearchCV(RandomizedSearchCV):
    def __init__(self, params):
        super().__init__(
            estimator=params['estimator'],
            param_distributions=params['param_distributions'],
            n_iter=params['n_iter'],
            scoring=params['scoring'],
            n_jobs=params['n_jobs'],
            refit=params['refit'],
            cv=params['cv'],
            verbose=params['verbose'],
            pre_dispatch=params['pre_dispatch'],
            random_state=params['random_state'],
            error_score=params['error_score'],
            return_train_score=params['return_train_score'],
        )