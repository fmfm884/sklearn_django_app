import os
import shutil
import pandas as pd
#from django.shortcuts import render, render_to_response
from django.views.generic import TemplateView, FormView
from django import forms
from . import my_forms
from .data_analysis.data import PrepareDataset
from .data_analysis import algorithm
from .data_analysis.evaluate import ClfEval, RegEval, ClstEval
import traceback
from sklearn.svm import SVC, SVR

MEDIA_BASE_JPG_PATH = os.path.join('static', 'DataAnalysisAPP', 'media')


class HomeView(TemplateView):
    template_name = 'DataAnalysisApp/home.html'
    def get(self, request, *args, **kwargs):
        shutil.rmtree(MEDIA_BASE_JPG_PATH)
        os.makedirs(MEDIA_BASE_JPG_PATH)
        return super().get(request, *args, **kwargs)

class DatasetView(FormView):
    template_name = 'DataAnalysisApp/category/dataset/dataset.html'
    form_class = my_forms.DatasetForm

    def form_valid(self, form):
        params = form.cleaned_data
        my_data = PrepareDataset(params['dataset'])
        df_dataset = my_data.create_dataframe()
        ctxt = self.get_context_data(data=params, form=form)
        if 1 < df_dataset.shape[1] and df_dataset.shape[1] <= 30:
            corr_heat_jpg_path = my_data.correlation(save_jpg_base_path=MEDIA_BASE_JPG_PATH).split(os.sep, maxsplit=1)[-1]
            ctxt.update({'corr_heat_jpg_path': corr_heat_jpg_path})
        img_dataset_keys = [key[0] for key in my_forms.img_dataset_choice]
        if params['dataset'] in img_dataset_keys:
            if params['dataset'] == 'olivetti_faces':
                img_data_jpg_path = my_data.draw_img(save_jpg_base_path=MEDIA_BASE_JPG_PATH, h=64, w=64).split(os.sep, maxsplit=1)[-1]
            elif params['dataset'] == 'lfw_people':
                img_data_jpg_path = my_data.draw_img(save_jpg_base_path=MEDIA_BASE_JPG_PATH, h=62, w=47).split(os.sep, maxsplit=1)[-1]
            elif params['dataset'] == 'lfw_pairs':
                img_data_jpg_path = my_data.draw_img(save_jpg_base_path=MEDIA_BASE_JPG_PATH, h=124, w=47).split(os.sep, maxsplit=1)[-1]
            ctxt.update({'img_data_jpg_path': img_data_jpg_path})
        ctxt.update({'dataframe': df_dataset.iloc[:10,:].to_html(classes='table', index=False, justify='left'),})
        return self.render_to_response(ctxt)


class ClfIntroductionView(TemplateView):
    template_name = 'DataAnalysisApp/category/classification/introduction.html'

    def get(self, request, **kwargs):
        ctxt = {'form': False}
        return self.render_to_response(ctxt)


class LogisticClassifierView(FormView):
    template_name = 'DataAnalysisApp/category/classification/logistic.html'
    form_class = my_forms.LogisticClassificationMultiForm

    def form_valid(self, form):
        params = form.cleaned_data
        my_data = PrepareDataset(params['dataset']['dataset'])
        dataset = my_data.load_dataset()
        df_dataset = my_data.create_dataframe()
        _, _, y_train, y_test = my_data.data_train_test_split(random_state=0)
        x_train_scaled, x_test_scaled = my_data.normalize()

        ctxt = self.get_context_data(data=params, form=form)

        try:
            log_reg = algorithm.LogisticRegression(params=params['logistic']).fit(x_train_scaled, y_train)
        except ValueError as e:
            err_msg = traceback.format_exception_only(type(e), e)[0].rstrip('\n')
            ctxt.update({'err_msg': err_msg})
            return self.render_to_response(ctxt)
        prediction = log_reg.predict(x_test_scaled)

        eval = ClfEval(y_test, prediction)

        ctxt.update({
            'dataset_msg': '上5行を抜粋したデータセットを確認',
            'df_dataset': df_dataset[:5].to_html(classes='table', index=False, justify='left'),
            'confusion_jpg_path':  eval.confusion_matrix(target_names=dataset.target_names, save_jpg_base_path=MEDIA_BASE_JPG_PATH).split(os.sep, maxsplit=1)[-1],
            'clf_report': eval.clf_report(target_names=dataset.target_names).to_html(classes='table', justify='left'),
        })
        return self.render_to_response(ctxt)


class SGDClassifierView(FormView):
    template_name = 'DataAnalysisApp/category/classification/sgd.html'
    form_class = my_forms.SGDClassificationMultiForm

    def form_valid(self, form):
        params = form.cleaned_data
        my_data = PrepareDataset(params['dataset']['dataset'])
        dataset = my_data.load_dataset()
        df_dataset = my_data.create_dataframe()
        _, _, y_train, y_test = my_data.data_train_test_split(random_state=0)
        x_train_scaled, x_test_scaled = my_data.normalize()

        ctxt = self.get_context_data(data=params, form=form)

        try:
            sgd = algorithm.SGDClassifier(params=params['sgd']).fit(x_train_scaled, y_train)
        except ValueError as e:
            err_msg = traceback.format_exception_only(type(e), e)[0].rstrip('\n')
            ctxt.update({'err_msg': err_msg})
            return self.render_to_response(ctxt)
        prediction = sgd.predict(x_test_scaled)

        eval = ClfEval(y_test, prediction)

        ctxt.update({
            'dataset_msg': '上5行を抜粋したデータセットを確認',
            'df_dataset': df_dataset[:5].to_html(classes='table', index=False, justify='left'),
            'confusion_jpg_path': eval.confusion_matrix(target_names=dataset.target_names, save_jpg_base_path=MEDIA_BASE_JPG_PATH).split(os.sep, maxsplit=1)[-1],
            'clf_report': eval.clf_report(target_names=dataset.target_names).to_html(classes='table', justify='left'),
        })
        return self.render_to_response(ctxt)


class KNeighborsClassifierView(FormView):
    template_name = 'DataAnalysisApp/category/classification/k-neighbor.html'
    form_class = my_forms.KNeighborsClassificationMultiForm

    def form_valid(self, form):
        params = form.cleaned_data
        my_data = PrepareDataset(params['dataset']['dataset'])
        dataset = my_data.load_dataset()
        df_dataset = my_data.create_dataframe()
        _, _, y_train, y_test = my_data.data_train_test_split(random_state=0)
        x_train_scaled, x_test_scaled = my_data.normalize()

        ctxt = self.get_context_data(data=params, form=form)

        try:
            kn = algorithm.KNeighborsClassifier(params=params['k_neighbors']).fit(x_train_scaled, y_train)
        except ValueError as e:
            err_msg = traceback.format_exception_only(type(e), e)[0].rstrip('\n')
            ctxt.update({'err_msg': err_msg})
            return self.render_to_response(ctxt)
        prediction = kn.predict(x_test_scaled)

        eval = ClfEval(y_test, prediction)

        ctxt.update({
            'dataset_msg': '上5行を抜粋したデータセットを確認',
            'df_dataset': df_dataset[:5].to_html(classes='table', index=False, justify='left'),
            'confusion_jpg_path':  eval.confusion_matrix(target_names=dataset.target_names, save_jpg_base_path=MEDIA_BASE_JPG_PATH).split(os.sep, maxsplit=1)[-1],
            'clf_report': eval.clf_report(target_names=dataset.target_names).to_html(classes='table', justify='left'),
        })
        return self.render_to_response(ctxt)


class SVCView(FormView):
    template_name = 'DataAnalysisApp/category/classification/svc.html'
    form_class = my_forms.SVCMultiForm

    def form_valid(self, form):
        params = form.cleaned_data
        my_data = PrepareDataset(params['dataset']['dataset'])
        dataset = my_data.load_dataset()
        df_dataset = my_data.create_dataframe()
        _, _, y_train, y_test = my_data.data_train_test_split(random_state=0)
        x_train_scaled, x_test_scaled = my_data.normalize()

        ctxt = self.get_context_data(data=params, form=form)

        try:
            svc = algorithm.SVC(params=params['svc']).fit(x_train_scaled, y_train)
        except ValueError as e:
            err_msg = traceback.format_exception_only(type(e), e)[0].rstrip('\n')
            ctxt.update({'err_msg': err_msg})
            return self.render_to_response(ctxt)
        prediction = svc.predict(x_test_scaled)

        eval = ClfEval(y_test, prediction)

        ctxt.update({
            'dataset_msg': '上5行を抜粋したデータセットを確認',
            'df_dataset': df_dataset[:5].to_html(classes='table', index=False, justify='left'),
            'confusion_jpg_path':  eval.confusion_matrix(target_names=dataset.target_names, save_jpg_base_path=MEDIA_BASE_JPG_PATH).split(os.sep, maxsplit=1)[-1],
            'clf_report': eval.clf_report(target_names=dataset.target_names).to_html(classes='table', justify='left'),
        })
        return self.render_to_response(ctxt)


class RandomForestClassifierView(FormView):
    template_name = 'DataAnalysisApp/category/classification/rfc.html'
    form_class = my_forms.RandomForestClassifierMultiForm

    def form_valid(self, form):
        params = form.cleaned_data
        my_data = PrepareDataset(params['dataset']['dataset'])
        dataset = my_data.load_dataset()
        df_dataset = my_data.create_dataframe()
        x_train, x_test, y_train, y_test = my_data.data_train_test_split(random_state=0)

        ctxt = self.get_context_data(data=params, form=form)

        try:
            rfc = algorithm.RandomForestClassifier(params=params['rfc']).fit(x_train, y_train)
        except ValueError as e:
            err_msg = traceback.format_exception_only(type(e), e)[0].rstrip('\n')
            ctxt.update({'err_msg': err_msg})
            return self.render_to_response(ctxt)
        prediction = rfc.predict(x_test)

        eval = ClfEval(y_test, prediction)

        ctxt.update({
            'dataset_msg': '上5行を抜粋したデータセットを確認',
            'df_dataset': df_dataset[:5].to_html(classes='table', index=False, justify='left'),
            'confusion_jpg_path':  eval.confusion_matrix(target_names=dataset.target_names, save_jpg_base_path=MEDIA_BASE_JPG_PATH).split(os.sep, maxsplit=1)[-1],
            'clf_report': eval.clf_report(target_names=dataset.target_names).to_html(classes='table', justify='left'),
        })
        return self.render_to_response(ctxt)


class RegIntroductionView(TemplateView):
    template_name = 'DataAnalysisApp/category/regression/introduction.html'

    def get(self, request, **kwargs):
        ctxt = {'form': False}
        return self.render_to_response(ctxt)


class SGDRegressorView(FormView):
    template_name = 'DataAnalysisApp/category/regression/sgd.html'
    form_class = my_forms.SGDRegressionMultiForm

    def form_valid(self, form):
        params = form.cleaned_data
        my_data = PrepareDataset(params['dataset']['dataset'])
        dataset = my_data.load_dataset()
        df_dataset = my_data.create_dataframe()
        _, _, y_train, y_test = my_data.data_train_test_split(random_state=0)
        x_train_scaled, x_test_scaled = my_data.normalize()

        ctxt = self.get_context_data(data=params, form=form)

        prediction = []
        try:
            # 目的変数が複数の場合
            if len(dataset.target.shape) > 1:
                # 各目的変数ごとにモデルを学習する
                for idx_obj_var in range(dataset.target.shape[1]):
                    y_train_single = y_train[:, idx_obj_var]
                    sgd = algorithm.SGDRegressor(params=params['sgd']).fit(x_train_scaled, y_train_single)
                    prediction.append(sgd.predict(x_test_scaled))
                obj_var = dataset.target_names
            # 目的変数が１つの場合
            else:
                sgd = algorithm.SGDRegressor(params=params['sgd']).fit(x_train_scaled, y_train)
                prediction.append(sgd.predict(x_test_scaled))
                obj_var = ['ObjectiveVariable_0']
        except ValueError as e:
            err_msg = traceback.format_exception_only(type(e), e)[0].rstrip('\n')
            ctxt.update({'err_msg': err_msg})
            return self.render_to_response(ctxt)

        eval = RegEval(y_test, prediction)

        ctxt.update({
            'dataset_msg': '上5行を抜粋したデータセットを確認',
            'df_dataset': df_dataset[:5].to_html(classes='table', index=False, justify='left'),
            'reg_report': eval.reg_report(obj_var).T.to_html(classes='table', justify='left'),
            'true_pred_jpg_path': eval.plot_true_pred(MEDIA_BASE_JPG_PATH).split(os.sep, maxsplit=1)[-1],
            'residual_error_jpg_path': eval.plot_residual_error(MEDIA_BASE_JPG_PATH).split(os.sep, maxsplit=1)[-1],
        })
        return self.render_to_response(ctxt)


class ElasticNetView(FormView):
    template_name = 'DataAnalysisApp/category/regression/elasticnet.html'
    form_class = my_forms.ElasticNetMultiForm

    def form_valid(self, form):
        params = form.cleaned_data
        my_data = PrepareDataset(params['dataset']['dataset'])
        dataset = my_data.load_dataset()
        df_dataset = my_data.create_dataframe()
        _, _, y_train, y_test = my_data.data_train_test_split(random_state=0)
        x_train_scaled, x_test_scaled = my_data.normalize()

        ctxt = self.get_context_data(data=params, form=form)

        prediction = []
        try:
            # 目的変数が複数の場合
            if len(dataset.target.shape) > 1:
                # 各目的変数ごとにモデルを学習する
                for idx_obj_var in range(dataset.target.shape[1]):
                    y_train_single = y_train[:, idx_obj_var]
                    elastic = algorithm.ElasticNet(params=params['elasticnet']).fit(x_train_scaled, y_train_single)
                    prediction.append(elastic.predict(x_test_scaled))
                obj_var = dataset.target_names
            # 目的変数が１つの場合
            else:
                elastic = algorithm.ElasticNet(params=params['elasticnet']).fit(x_train_scaled, y_train)
                prediction.append(elastic.predict(x_test_scaled))
                obj_var = ['ObjectiveVariable_0']
        except ValueError as e:
            err_msg = traceback.format_exception_only(type(e), e)[0].rstrip('\n')
            ctxt.update({'err_msg': err_msg})
            return self.render_to_response(ctxt)

        eval = RegEval(y_test, prediction)

        ctxt.update({
            'dataset_msg': '上5行を抜粋したデータセットを確認',
            'df_dataset': df_dataset[:5].to_html(classes='table', index=False, justify='left'),
            'reg_report': eval.reg_report(obj_var).T.to_html(classes='table', justify='left'),
            'true_pred_jpg_path': eval.plot_true_pred(MEDIA_BASE_JPG_PATH).split(os.sep, maxsplit=1)[-1],
            'residual_error_jpg_path': eval.plot_residual_error(MEDIA_BASE_JPG_PATH).split(os.sep, maxsplit=1)[-1],
        })
        return self.render_to_response(ctxt)


class LassoView(FormView):
    template_name = 'DataAnalysisApp/category/regression/lasso.html'
    form_class = my_forms.LassoMultiForm

    def form_valid(self, form):
        params = form.cleaned_data
        my_data = PrepareDataset(params['dataset']['dataset'])
        dataset = my_data.load_dataset()
        df_dataset = my_data.create_dataframe()
        _, _, y_train, y_test = my_data.data_train_test_split(random_state=0)
        x_train_scaled, x_test_scaled = my_data.normalize()

        ctxt = self.get_context_data(data=params, form=form)

        prediction = []
        try:
            # 目的変数が複数の場合
            if len(dataset.target.shape) > 1:
                # 各目的変数ごとにモデルを学習する
                for idx_obj_var in range(dataset.target.shape[1]):
                    y_train_single = y_train[:, idx_obj_var]
                    lasso = algorithm.Lasso(params=params['lasso']).fit(x_train_scaled, y_train_single)
                    prediction.append(lasso.predict(x_test_scaled))
                obj_var = dataset.target_names
            # 目的変数が１つの場合
            else:
                lasso = algorithm.Lasso(params=params['lasso']).fit(x_train_scaled, y_train)
                prediction.append(lasso.predict(x_test_scaled))
                obj_var = ['ObjectiveVariable_0']
        except ValueError as e:
            err_msg = traceback.format_exception_only(type(e), e)[0].rstrip('\n')
            ctxt.update({'err_msg': err_msg})
            return self.render_to_response(ctxt)

        eval = RegEval(y_test, prediction)

        ctxt.update({
            'dataset_msg': '上5行を抜粋したデータセットを確認',
            'df_dataset': df_dataset[:5].to_html(classes='table', index=False, justify='left'),
            'reg_report': eval.reg_report(obj_var).T.to_html(classes='table', justify='left'),
            'true_pred_jpg_path': eval.plot_true_pred(MEDIA_BASE_JPG_PATH).split(os.sep, maxsplit=1)[-1],
            'residual_error_jpg_path': eval.plot_residual_error(MEDIA_BASE_JPG_PATH).split(os.sep, maxsplit=1)[-1],
        })
        return self.render_to_response(ctxt)


class RidgeView(FormView):
    template_name = 'DataAnalysisApp/category/regression/ridge.html'
    form_class = my_forms.RidgeMultiForm

    def form_valid(self, form):
        params = form.cleaned_data
        my_data = PrepareDataset(params['dataset']['dataset'])
        dataset = my_data.load_dataset()
        df_dataset = my_data.create_dataframe()
        _, _, y_train, y_test = my_data.data_train_test_split(random_state=0)
        x_train_scaled, x_test_scaled = my_data.normalize()

        ctxt = self.get_context_data(data=params, form=form)

        prediction = []
        try:
            # 目的変数が複数の場合
            if len(dataset.target.shape) > 1:
                # 各目的変数ごとにモデルを学習する
                for idx_obj_var in range(dataset.target.shape[1]):
                    y_train_single = y_train[:, idx_obj_var]
                    ridge = algorithm.Ridge(params=params['ridge']).fit(x_train_scaled, y_train_single)
                    prediction.append(ridge.predict(x_test_scaled))
                obj_var = dataset.target_names
            # 目的変数が１つの場合
            else:
                ridge = algorithm.Ridge(params=params['ridge']).fit(x_train_scaled, y_train)
                prediction.append(ridge.predict(x_test_scaled))
                obj_var = ['ObjectiveVariable_0']
        except ValueError as e:
            err_msg = traceback.format_exception_only(type(e), e)[0].rstrip('\n')
            ctxt.update({'err_msg': err_msg})
            return self.render_to_response(ctxt)

        eval = RegEval(y_test, prediction)

        ctxt.update({
            'dataset_msg': '上5行を抜粋したデータセットを確認',
            'df_dataset': df_dataset[:5].to_html(classes='table', index=False, justify='left'),
            'reg_report': eval.reg_report(obj_var).T.to_html(classes='table', justify='left'),
            'true_pred_jpg_path': eval.plot_true_pred(MEDIA_BASE_JPG_PATH).split(os.sep, maxsplit=1)[-1],
            'residual_error_jpg_path': eval.plot_residual_error(MEDIA_BASE_JPG_PATH).split(os.sep, maxsplit=1)[-1],
        })
        return self.render_to_response(ctxt)


class SVRView(FormView):
    template_name = 'DataAnalysisApp/category/regression/svr.html'
    form_class = my_forms.SVRMultiForm

    def form_valid(self, form):
        params = form.cleaned_data
        my_data = PrepareDataset(params['dataset']['dataset'])
        dataset = my_data.load_dataset()
        df_dataset = my_data.create_dataframe()
        _, _, y_train, y_test = my_data.data_train_test_split(random_state=0)
        x_train_scaled, x_test_scaled = my_data.normalize()

        ctxt = self.get_context_data(data=params, form=form)

        prediction = []
        try:
            # 目的変数が複数の場合
            if len(dataset.target.shape) > 1:
                # 各目的変数ごとにモデルを学習する
                for idx_obj_var in range(dataset.target.shape[1]):
                    y_train_single = y_train[:, idx_obj_var]
                    svr = algorithm.SVR(params=params['svr']).fit(x_train_scaled, y_train_single)
                    prediction.append(svr.predict(x_test_scaled))
                obj_var = dataset.target_names
            # 目的変数が１つの場合
            else:
                svr = algorithm.SVR(params=params['svr']).fit(x_train_scaled, y_train)
                prediction.append(svr.predict(x_test_scaled))
                obj_var = ['ObjectiveVariable_0']
        except ValueError as e:
            err_msg = traceback.format_exception_only(type(e), e)[0].rstrip('\n')
            ctxt.update({'err_msg': err_msg})
            return self.render_to_response(ctxt)

        eval = RegEval(y_test, prediction)

        ctxt.update({
            'dataset_msg': '上5行を抜粋したデータセットを確認',
            'df_dataset': df_dataset[:5].to_html(classes='table', index=False, justify='left'),
            'reg_report': eval.reg_report(obj_var).T.to_html(classes='table', justify='left'),
            'true_pred_jpg_path': eval.plot_true_pred(MEDIA_BASE_JPG_PATH).split(os.sep, maxsplit=1)[-1],
            'residual_error_jpg_path': eval.plot_residual_error(MEDIA_BASE_JPG_PATH).split(os.sep, maxsplit=1)[-1],
        })
        return self.render_to_response(ctxt)


class RandomForestRegressorView(FormView):
    template_name = 'DataAnalysisApp/category/regression/rfr.html'
    form_class = my_forms.RandomForestRegressorMultiForm

    def form_valid(self, form):
        params = form.cleaned_data
        my_data = PrepareDataset(params['dataset']['dataset'])
        dataset = my_data.load_dataset()
        df_dataset = my_data.create_dataframe()
        x_train, x_test, y_train, y_test = my_data.data_train_test_split(random_state=0)

        ctxt = self.get_context_data(data=params, form=form)

        prediction = []
        try:
            # 目的変数が複数の場合
            if len(dataset.target.shape) > 1:
                # 各目的変数ごとにモデルを学習する
                for idx_obj_var in range(dataset.target.shape[1]):
                    y_train_single = y_train[:, idx_obj_var]
                    rfr = algorithm.RandomForestRegressor(params=params['rfr']).fit(x_train, y_train_single)
                    prediction.append(rfr.predict(x_test))
                obj_var = dataset.target_names
            # 目的変数が１つの場合
            else:
                rfr = algorithm.RandomForestRegressor(params=params['rfr']).fit(x_train, y_train)
                prediction.append(rfr.predict(x_test))
                obj_var = ['ObjectiveVariable_0']
        except ValueError as e:
            err_msg = traceback.format_exception_only(type(e), e)[0].rstrip('\n')
            ctxt.update({'err_msg': err_msg})
            return self.render_to_response(ctxt)

        eval = RegEval(y_test, prediction)

        ctxt.update({
            'dataset_msg': '上5行を抜粋したデータセットを確認',
            'df_dataset': df_dataset[:5].to_html(classes='table', index=False, justify='left'),
            'reg_report': eval.reg_report(obj_var).T.to_html(classes='table', justify='left'),
            'true_pred_jpg_path': eval.plot_true_pred(MEDIA_BASE_JPG_PATH).split(os.sep, maxsplit=1)[-1],
            'residual_error_jpg_path': eval.plot_residual_error(MEDIA_BASE_JPG_PATH).split(os.sep, maxsplit=1)[-1],
        })
        return self.render_to_response(ctxt)


class DrIntroductionView(TemplateView):
    template_name = 'DataAnalysisApp/category/dimensionality_reduction/introduction.html'

    def get(self, request, **kwargs):
        ctxt = {'form': False}
        return self.render_to_response(ctxt)


class PCAView(FormView):
    template_name = 'DataAnalysisApp/category/dimensionality_reduction/pca.html'
    form_class = my_forms.PCAMultiForm

    def form_valid(self, form):
        params = form.cleaned_data
        my_data = PrepareDataset(params['dataset']['dataset'])
        dataset = my_data.load_dataset()
        df_dataset = my_data.create_dataframe()

        ctxt = self.get_context_data(data=params, form=form)

        try:
            pca = algorithm.PCA(params=params['pca']).fit_transform(df_dataset)
        except ValueError as e:
            err_msg = traceback.format_exception_only(type(e), e)[0].rstrip('\n')
            ctxt.update({'err_msg': err_msg})
            return self.render_to_response(ctxt)

        df_pca = pd.DataFrame(pca)
        df_true = pd.Series(dataset.target, name='true')
        df_data = pd.concat([df_pca, df_true], axis=1)

        ctxt.update({
            'dataset_msg': '上5行を抜粋したデータセットを確認',
            'df_dataset': df_dataset[:5].to_html(classes='table', index=False, justify='left'),
            'data_jpg_path': ClstEval.plot_data(df_data=df_data, save_jpg_base_path=MEDIA_BASE_JPG_PATH, labels=dataset.target_names).split(os.sep, maxsplit=1)[-1],
        })
        return self.render_to_response(ctxt)



class KernelPCAView(FormView):
    template_name = 'DataAnalysisApp/category/dimensionality_reduction/kernel_pca.html'
    form_class = my_forms.KernelPCAMultiForm

    def form_valid(self, form):
        params = form.cleaned_data
        my_data = PrepareDataset(params['dataset']['dataset'])
        dataset = my_data.load_dataset()
        df_dataset = my_data.create_dataframe()

        ctxt = self.get_context_data(data=params, form=form)

        try:
            kpca = algorithm.KernelPCA(params=params['kpca']).fit_transform(df_dataset)
        except ValueError as e:
            err_msg = traceback.format_exception_only(type(e), e)[0].rstrip('\n')
            ctxt.update({'err_msg': err_msg})
            return self.render_to_response(ctxt)

        df_kpca = pd.DataFrame(kpca)
        df_true = pd.Series(dataset.target, name='true')
        df_data = pd.concat([df_kpca, df_true], axis=1)

        ctxt.update({
            'dataset_msg': '上5行を抜粋したデータセットを確認',
            'df_dataset': df_dataset[:5].to_html(classes='table', index=False, justify='left'),
            'data_jpg_path': ClstEval.plot_data(df_data=df_data, save_jpg_base_path=MEDIA_BASE_JPG_PATH, labels=dataset.target_names).split(os.sep, maxsplit=1)[-1],
        })
        return self.render_to_response(ctxt)


class TruncatedSVDView(FormView):
    template_name = 'DataAnalysisApp/category/dimensionality_reduction/truncatedsvd.html'
    form_class = my_forms.TruncatedSVDMultiForm

    def form_valid(self, form):
        params = form.cleaned_data
        my_data = PrepareDataset(params['dataset']['dataset'])
        dataset = my_data.load_dataset()
        df_dataset = my_data.create_dataframe()

        ctxt = self.get_context_data(data=params, form=form)

        try:
            truncatedsvd = algorithm.TruncatedSVD(params=params['truncatedsvd']).fit_transform(df_dataset)
        except ValueError as e:
            err_msg = traceback.format_exception_only(type(e), e)[0].rstrip('\n')
            ctxt.update({'err_msg': err_msg})
            return self.render_to_response(ctxt)

        df_truncatedsvd = pd.DataFrame(truncatedsvd)
        df_true = pd.Series(dataset.target, name='true')
        df_data = pd.concat([df_truncatedsvd, df_true], axis=1)

        ctxt.update({
            'dataset_msg': '上5行を抜粋したデータセットを確認',
            'df_dataset': df_dataset[:5].to_html(classes='table', index=False, justify='left'),
            'data_jpg_path': ClstEval.plot_data(df_data=df_data, save_jpg_base_path=MEDIA_BASE_JPG_PATH, labels=dataset.target_names).split(os.sep, maxsplit=1)[-1],
        })
        return self.render_to_response(ctxt)


class ClstIntroductionView(TemplateView):
    template_name = 'DataAnalysisApp/category/clustering/introduction.html'

    def get(self, request, **kwargs):
        ctxt = {'form': False}
        return self.render_to_response(ctxt)


class KMeansView(FormView):
    template_name = 'DataAnalysisApp/category/clustering/kmeans.html'
    form_class = my_forms.KMeansPCAMultiForm

    def form_valid(self, form):
        params = form.cleaned_data
        my_data = PrepareDataset(params['dataset']['dataset'])
        dataset = my_data.load_dataset()
        df_dataset = my_data.create_dataframe()
        data_scaled = my_data.normalize(split=False)

        ctxt = self.get_context_data(data=params, form=form)

        try:
            kmeans = algorithm.KMeans(params=params['kmeans']).fit(data_scaled)
            pca = algorithm.PCA(params=params['pca']).fit_transform(df_dataset)
        except ValueError as e:
            err_msg = traceback.format_exception_only(type(e), e)[0].rstrip('\n')
            ctxt.update({'err_msg': err_msg})
            return self.render_to_response(ctxt)

        df_pred = pd.Series(kmeans.labels_, name='pred')
        df_pca = pd.DataFrame(pca)
        df_true = pd.Series(dataset.target, name='true')
        df_data = pd.concat([df_pca, df_pred, df_true], axis=1)

        ctxt.update({
            'dataset_msg': '上5行を抜粋したデータセットを確認',
            'df_dataset': df_dataset[:5].to_html(classes='table', index=False, justify='left'),
            'data_jpg_path': ClstEval.multi_plot_data(df_data=df_data, save_jpg_base_path=MEDIA_BASE_JPG_PATH, labels=dataset.target_names).split(os.sep, maxsplit=1)[-1],
        })
        return self.render_to_response(ctxt)


class MeanShiftView(FormView):
    template_name = 'DataAnalysisApp/category/clustering/meanshift.html'
    form_class = my_forms.MeanShiftPCAMultiForm

    def form_valid(self, form):
        params = form.cleaned_data
        my_data = PrepareDataset(params['dataset']['dataset'])
        dataset = my_data.load_dataset()
        df_dataset = my_data.create_dataframe()
        data_scaled = my_data.normalize(split=False)

        ctxt = self.get_context_data(data=params, form=form)

        try:
            meanshift = algorithm.MeanShift(params=params['meanshift']).fit(data_scaled)
            pca = algorithm.PCA(params=params['pca']).fit_transform(df_dataset)
        except ValueError as e:
            err_msg = traceback.format_exception_only(type(e), e)[0].rstrip('\n')
            ctxt.update({'err_msg': err_msg})
            return self.render_to_response(ctxt)

        df_pred = pd.Series(meanshift.labels_, name='pred')
        df_pca = pd.DataFrame(pca)
        df_true = pd.Series(dataset.target, name='true')
        df_data = pd.concat([df_pca, df_pred, df_true], axis=1)

        ctxt.update({
            'dataset_msg': '上5行を抜粋したデータセットを確認',
            'df_dataset': df_dataset[:5].to_html(classes='table', index=False, justify='left'),
            'data_jpg_path': ClstEval.multi_plot_data(df_data=df_data, save_jpg_base_path=MEDIA_BASE_JPG_PATH, labels=dataset.target_names).split(os.sep, maxsplit=1)[-1],
        })
        return self.render_to_response(ctxt)


class VBGMMView(FormView):
    template_name = 'DataAnalysisApp/category/clustering/vbgmm.html'
    form_class = my_forms.VBGMMPCAMultiForm

    def form_valid(self, form):
        params = form.cleaned_data
        my_data = PrepareDataset(params['dataset']['dataset'])
        dataset = my_data.load_dataset()
        df_dataset = my_data.create_dataframe()
        data_scaled = my_data.normalize(split=False)

        ctxt = self.get_context_data(data=params, form=form)

        try:
            vbgmm = algorithm.VBGMM(params=params['vbgmm']).fit(data_scaled)
            pca = algorithm.PCA(params=params['pca']).fit_transform(df_dataset)
        except ValueError as e:
            err_msg = traceback.format_exception_only(type(e), e)[0].rstrip('\n')
            ctxt.update({'err_msg': err_msg})
            return self.render_to_response(ctxt)
        prediction = vbgmm.predict(data_scaled)

        df_pred = pd.Series(prediction, name='pred')
        df_pca = pd.DataFrame(pca)
        df_true = pd.Series(dataset.target, name='true')
        df_data = pd.concat([df_pca, df_pred, df_true], axis=1)

        ctxt.update({
            'dataset_msg': '上5行を抜粋したデータセットを確認',
            'df_dataset': df_dataset[:5].to_html(classes='table', index=False, justify='left'),
            'data_jpg_path': ClstEval.multi_plot_data(df_data=df_data, save_jpg_base_path=MEDIA_BASE_JPG_PATH, labels=dataset.target_names).split(os.sep, maxsplit=1)[-1],
        })
        return self.render_to_response(ctxt)


class SpectralClusteringView(FormView):
    template_name = 'DataAnalysisApp/category/clustering/spectralclustering.html'
    form_class = my_forms.SpectralClusteringPCAMultiForm

    def form_valid(self, form):
        params = form.cleaned_data
        my_data = PrepareDataset(params['dataset']['dataset'])
        dataset = my_data.load_dataset()
        df_dataset = my_data.create_dataframe()
        data_scaled = my_data.normalize(split=False)

        ctxt = self.get_context_data(data=params, form=form)

        try:
            spectralclustering = algorithm.SpectralClustering(params=params['spectralclustering']).fit(data_scaled)
            pca = algorithm.PCA(params=params['pca']).fit_transform(df_dataset)
        except ValueError as e:
            err_msg = traceback.format_exception_only(type(e), e)[0].rstrip('\n')
            ctxt.update({'err_msg': err_msg})
            return self.render_to_response(ctxt)

        df_pred = pd.Series(spectralclustering.labels_, name='pred')
        df_pca = pd.DataFrame(pca)
        df_true = pd.Series(dataset.target, name='true')
        df_data = pd.concat([df_pca, df_pred, df_true], axis=1)

        ctxt.update({
            'dataset_msg': '上5行を抜粋したデータセットを確認',
            'df_dataset': df_dataset[:5].to_html(classes='table', index=False, justify='left'),
            'data_jpg_path': ClstEval.multi_plot_data(df_data=df_data, save_jpg_base_path=MEDIA_BASE_JPG_PATH, labels=dataset.target_names).split(os.sep, maxsplit=1)[-1],
        })
        return self.render_to_response(ctxt)


class GaussianMixtureView(FormView):
    template_name = 'DataAnalysisApp/category/clustering/gmm.html'
    form_class = my_forms.GaussianMixturePCAMultiForm

    def form_valid(self, form):
        params = form.cleaned_data
        my_data = PrepareDataset(params['dataset']['dataset'])
        dataset = my_data.load_dataset()
        df_dataset = my_data.create_dataframe()
        data_scaled = my_data.normalize(split=False)

        ctxt = self.get_context_data(data=params, form=form)

        try:
            gmm = algorithm.GaussianMixture(params=params['gmm']).fit(data_scaled)
            pca = algorithm.PCA(params=params['pca']).fit_transform(df_dataset)
        except ValueError as e:
            err_msg = traceback.format_exception_only(type(e), e)[0].rstrip('\n')
            ctxt.update({'err_msg': err_msg})
            return self.render_to_response(ctxt)
        prediction = gmm.predict(data_scaled)

        df_pred = pd.Series(prediction, name='pred')
        df_pca = pd.DataFrame(pca)
        df_true = pd.Series(dataset.target, name='true')
        df_data = pd.concat([df_pca, df_pred, df_true], axis=1)

        ctxt.update({
            'dataset_msg': '上5行を抜粋したデータセットを確認',
            'df_dataset': df_dataset[:5].to_html(classes='table', index=False, justify='left'),
            'data_jpg_path': ClstEval.multi_plot_data(df_data=df_data, save_jpg_base_path=MEDIA_BASE_JPG_PATH, labels=dataset.target_names).split(os.sep, maxsplit=1)[-1],
        })
        return self.render_to_response(ctxt)


class TuningIntroductionView(TemplateView):
    template_name = 'DataAnalysisApp/category/tuning/introduction.html'

    def get(self, request, **kwargs):
        ctxt = {'form': False}
        return self.render_to_response(ctxt)


class GridSearchCVView(FormView):
    template_name = 'DataAnalysisApp/category/tuning/gridsearchcv.html'
    form_class = my_forms.GridSearchCVMultiForm

    def form_valid(self, form):
        params = form.cleaned_data
        params['gridsearchcv']['estimator'] = SVC(random_state=0)
        my_data = PrepareDataset(params['dataset']['dataset'])
        dataset = my_data.load_dataset()
        df_dataset = my_data.create_dataframe()
        x_train_val, x_test, y_train_val, y_test = my_data.data_train_test_split()
        x_train_val_scaled, x_test_scaled = my_data.normalize()

        ctxt = self.get_context_data(data=params, form=form)

        try:
            tuned_model = algorithm.GridSearchCV(params=params['gridsearchcv']).fit(x_train_val_scaled, y_train_val)
        except ValueError as e:
            err_msg = traceback.format_exception_only(type(e), e)[0].rstrip('\n')
            ctxt.update({'err_msg': err_msg})
            return self.render_to_response(ctxt)
        best_model = tuned_model.best_estimator_
        prediction = best_model.predict(x_test_scaled)

        eval = ClfEval(y_test, prediction)

        ctxt.update({
            'dataset_msg': '上5行を抜粋したデータセットを確認',
            'df_dataset': df_dataset[:5].to_html(classes='table', index=False, justify='left'),
            'df_tuning': pd.DataFrame(tuned_model.cv_results_).T.to_html(classes='table', justify='left'),
            'confusion_jpg_path':  eval.confusion_matrix(target_names=dataset.target_names, save_jpg_base_path=MEDIA_BASE_JPG_PATH).split(os.sep, maxsplit=1)[-1],
            'clf_report': eval.clf_report(target_names=dataset.target_names).to_html(classes='table', justify='left'),
        })
        return self.render_to_response(ctxt)


class RandomizedSearchCVView(FormView):
    template_name = 'DataAnalysisApp/category/tuning/randomizedsearchcv.html'
    form_class = my_forms.RandomizedSearchCVMultiForm        

    def form_valid(self, form):
        params = form.cleaned_data
        params['randomizedsearchcv']['estimator'] = SVC(random_state=0)
        my_data = PrepareDataset(params['dataset']['dataset'])
        dataset = my_data.load_dataset()
        df_dataset = my_data.create_dataframe()
        _, _, y_train_val, y_test = my_data.data_train_test_split()
        x_train_val_scaled, x_test_scaled = my_data.normalize()

        ctxt = self.get_context_data(data=params, form=form)

        try:
            tuned_model = algorithm.RandomizedSearchCV(params=params['randomizedsearchcv']).fit(x_train_val_scaled, y_train_val)
        except ValueError as e:
            err_msg = traceback.format_exception_only(type(e), e)[0].rstrip('\n')
            ctxt.update({'err_msg': err_msg})
            return self.render_to_response(ctxt)
        best_model = tuned_model.best_estimator_
        prediction = best_model.predict(x_test_scaled)

        eval = ClfEval(y_test, prediction)

        ctxt.update({
            'dataset_msg': '上5行を抜粋したデータセットを確認',
            'df_dataset': df_dataset[:5].to_html(classes='table', index=False, justify='left'),
            'df_tuning': pd.DataFrame(tuned_model.cv_results_).T.to_html(classes='table', justify='left'),
            'confusion_jpg_path':  eval.confusion_matrix(target_names=dataset.target_names, save_jpg_base_path=MEDIA_BASE_JPG_PATH).split(os.sep, maxsplit=1)[-1],
            'clf_report': eval.clf_report(target_names=dataset.target_names).to_html(classes='table', justify='left'),
        })
        return self.render_to_response(ctxt)