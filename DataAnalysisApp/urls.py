from os import name
from django.urls import path
from . import views 


urlpatterns = [
    path('home/', views.HomeView.as_view(), name='home'),
    path('category/dataset/', views.DatasetView.as_view(), name='dataset'),
    path('category/classification/introduction/', views.ClfIntroductionView.as_view(), name='clf_introduction'),
    path('category/classification/logistic/', views.LogisticClassifierView.as_view(), name='clf_logistic'),
    path('category/classification/k-neighbor/', views.KNeighborsClassifierView.as_view(), name='clf_k_neighbors'),
    path('category/classification/svc/', views.SVCView.as_view(), name='clf_svc'),
    path('category/classification/sgd/', views.SGDClassifierView.as_view(), name='clf_sgd'),
    path('category/classification/rfc', views.RandomForestClassifierView.as_view(), name='clf_rfc'),
    path('category/regression/introduction/', views.RegIntroductionView.as_view(), name='reg_introduction'),
    path('category/regression/sgd/', views.SGDRegressorView.as_view(), name='reg_sgd'),
    path('category/regression/elasticnet/', views.ElasticNetView.as_view(), name='reg_elasticnet'),
    path('category/regression/lasso/', views.LassoView.as_view(), name='reg_lasso'),
    path('category/regression/ridge/', views.RidgeView.as_view(), name='reg_ridge'),
    path('category/regression/svr/', views.SVRView.as_view(), name='reg_svr'),
    path('category/regression/rfr', views.RandomForestRegressorView.as_view(), name='reg_rfr'),
    path('category/dimensionality_reduction/introduction/', views.DrIntroductionView.as_view(), name='dr_introduction'),
    path('category/dimensionality_reduction/pca/', views.PCAView.as_view(), name='dr_pca'),
    path('category/dimensionality_reduction/kpca/', views.KernelPCAView.as_view(), name='dr_kpca'),
    path('category/dimensionality_reduction/truncatedsvd/', views.TruncatedSVDView.as_view(), name='dr_truncatedsvd'),
    path('category/clustering/introduction/', views.ClstIntroductionView.as_view(), name='clst_introduction'),
    path('category/clustering/k-means/', views.KMeansView.as_view(), name='clst_kmeans'),
    path('category/clustering/meanshift/', views.MeanShiftView.as_view(), name='clst_meanshift'),
    path('category/clustering/vbgmm/', views.VBGMMView.as_view(), name='clst_vbgmm'),
    path('category/clustering/spectralclustering/', views.SpectralClusteringView.as_view(), name='clst_spectralclustering'),
    path('category/clustering/gaussianmixture', views.GaussianMixtureView.as_view(), name='clst_gaussianmixture'),
    path('category/tuning/introduction/', views.TuningIntroductionView.as_view(), name='tuning_introduction'),
    path('category/tuning/gridsearchcv/', views.GridSearchCVView.as_view(), name='tuning_gridsearchcv'),
    path('category/tuning/randomizedsearchcv/', views.RandomizedSearchCVView.as_view(), name='tuning_randomizedsearchcv'),
]