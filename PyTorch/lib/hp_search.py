# -*- coding: utf-8 -*-
# ----------------------------------------------
# Project:  Infra
# Filename: gp.py
#
#                     Rubens Machado, 2017-11-06
# ----------------------------------------------
import os
import pickle
import time
import numpy as np

import torch.nn as nn
from torch.optim.lr_scheduler import StepLR, ExponentialLR, _LRScheduler
from torch.utils.data import DataLoader
from torch.optim import Optimizer, Adam, SGD

from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer

from sklearn.model_selection import PredefinedSplit

import pytorch_trainer as ptt


class SklEstimator(object):

    skl_id = 0
    fit_num = 0

    def __init__(self, model_spec=None, loss_spec='BCELoss', optim_spec='SGD', sched_spec='',
                 fit_trainer=ptt.DeepNetTrainer, fit_batch_size=16, fit_nb_epochs=100, fit_verbose=0,
                 fit_callbacks=None, fit_models_dir='', fit_use_best_model=False, fit_keep_models=False, **parameters):
        self.model_spec = model_spec
        self.loss_spec = loss_spec
        self.optim_spec = optim_spec
        self.sched_spec = sched_spec

        self.param_dict = {
            'model': {},
            'loss': {},
            'optim': {},
            'sched': {},
            'fit': {
                'batch_size': fit_batch_size,
                'nb_epochs': fit_nb_epochs,
                'verbose': fit_verbose,
                'callbacks': fit_callbacks,
                'trainer': fit_trainer,
                'models_dir': fit_models_dir,
                'use_best_model': fit_use_best_model,
                'keep_models': fit_keep_models,
            }
        }
        for pname, pvalue in parameters.items():
            prefix, par_name = pname.split('_', 1)
            if prefix not in self.param_dict.keys():
                continue
            self.param_dict[prefix][par_name] = pvalue

    def _initialize(self):
        SklEstimator.skl_id += 1
        self.idd = 'hpsmodel-{}-{}'.format(time.strftime('%Y-%m-%d', time.localtime()), SklEstimator.skl_id)

        # Model
        # =====
        if isinstance(self.model_spec, nn.Module):
            self.model = self.model_spec
        else:
            self.model = self.model_spec(**self.param_dict['model'])

        # Loss
        # ====
        if isinstance(self.loss_spec, str):
            if self.loss_spec == 'CrossEntropyLoss':
                self.loss = nn.CrossEntropyLoss()
            elif self.loss_spec == 'MSELoss':
                self.loss = nn.MSELoss()
            elif self.loss_spec == 'BCELoss':
                self.loss = nn.BCELoss()
            else:
                self.loss = None
                raise Exception("Invalid loss class name.")
        elif isinstance(self.loss_spec, nn.Module):
            self.loss = self.loss_spec
        elif self.loss_spec is not None:
            self.loss = self.loss_spec(**self.param_dict['loss'])
        else:
            raise Exception('Invalid loss.')

        # Optimizer
        # =========
        if isinstance(self.optim_spec, str):
            if self.optim_spec == 'Adam':
                self.optim = Adam(self.model.parameters(), lr=self.param_dict['optim']['lr'],
                                  weight_decay=self.param_dict['optim']['weight_decay'])
            elif self.optim_spec == 'SGD':
                self.optim = SGD(self.model.parameters(), lr=self.param_dict['optim']['lr'],
                                 momentum=self.param_dict['optim']['momentum'], nesterov=True,
                                 weight_decay=self.param_dict['optim']['weight_decay'])
            else:
                raise Exception("Invalid optimizer class name.")
        elif isinstance(self.optim_spec, Optimizer):
            self.optim = self.optim_spec
        elif self.optim_spec is not None:
            self.optim = self.optim_spec(**self.param_dict['optim'])
        else:
            raise Exception('Invalid optimizer.')

        # LR Scheduler
        # ============
        if isinstance(self.sched_spec, str):
            if self.sched_spec == 'StepLR':
                self.sched = StepLR(self.optim,
                                    step_size=self.param_dict['sched']['step_size'],
                                    gamma=self.param_dict['sched']['gamma'])
            elif self.sched_spec == 'ExponentialLR':
                self.sched = ExponentialLR(self.optim, gamma=self.param_dict['sched']['gamma'])
            else:
                self.sched = None
        elif isinstance(self.sched_spec, _LRScheduler):
            self.sched = self.sched_spec
        elif self.sched is not None:
            self.sched = self.sched_spec(**self.param_dict['sched'])

        # Model checkpoints
        # =================
        if self.param_dict['fit']['use_best_model'] or self.param_dict['fit']['keep_models']:
            if self.param_dict['fit']['callbacks'] is None:
                self.param_dict['fit']['callbacks'] = []
            os.makedirs(self.param_dict['fit']['models_dir'], exist_ok=True)
            self.model_basename = os.path.join(self.param_dict['fit']['models_dir'], self.idd)
            self.param_dict['fit']['callbacks'].append(ptt.ModelCheckpoint(self.model_basename))

        # Finally, the trainer
        # ====================
        self.trainer = self.param_dict['fit']['trainer'](model=self.model,
                                                         criterion=self.loss,
                                                         optimizer=self.optim,
                                                         lr_scheduler=self.sched,
                                                         callbacks=self.param_dict['fit']['callbacks'])

    def get_params(self, deep):
        params = {
            'model_spec': self.model_spec,
            'loss_spec': self.loss_spec,
            'optim_spec': self.optim_spec,
            'sched_spec': self.sched_spec,
        }
        for k1, v1 in self.param_dict.items():
            for k2, v2 in v1.items():
                pname = '{}_{}'.format(k1, k2)
                params[pname] = v2
        return params

    def set_params(self, **parameters):
        for pname, pvalue in parameters.items():
            if pname in ['model_spec', 'loss_spec', 'optim_spec', 'sched_spec']:
                setattr(self, pname, pvalue)
                continue
            prefix, par_name = pname.split('_', 1)
            if prefix not in self.param_dict.keys():
                continue
            self.param_dict[prefix][par_name] = pvalue
        self._initialize()
        return self

    def fit(self, dummyX, dummyY=None, train_ds=None, valid_ds=None):
        SklEstimator.fit_num += 1
        self.t0 = time.time()
        if self.param_dict['fit']['verbose'] > 0:
            print('***** Fit #{}: '.format(SklEstimator.fit_num), end=' ')
        self.train_dloader = DataLoader(train_ds, batch_size=self.param_dict['fit']['batch_size'], shuffle=True)
        self.valid_dloader = DataLoader(valid_ds, batch_size=self.param_dict['fit']['batch_size'], shuffle=False)
        self.trainer.fit_loader(self.param_dict['fit']['nb_epochs'], self.train_dloader, self.valid_dloader)
        if self.param_dict['fit']['use_best_model']:
            self.trainer.load_state(self.model_basename)
            if not self.param_dict['fit']['keep_models']:
                os.unlink(self.model_basename + '.model')
                os.unlink(self.model_basename + '.histo')

    def score(self, dummyX, dummyY=None):
        mdict = self.trainer.evaluate_loader(self.valid_dloader, verbose=0)
        score = - mdict['losses']    # negativo pois é busca por máximo score
        if self.param_dict['fit']['verbose'] > 0:
            print('{:.5f}  [{} epochs]  {:.2f}s'.format(score, self.trainer.last_epoch, time.time() - self.t0))
        return score


class HyperParSearch(BayesSearchCV):

    def __init__(self, parameters, n_iter=50, n_initial=10, n_jobs=1,
                 scoring=None, iid=True, verbose=0, pre_dispatch='2*n_jobs',
                 random_state=None, error_score='raise', return_train_score=False,
                 results_filename='hps_results.pkl'):

        fixed_params = {}
        search_spaces = {}
        for par, val in parameters.items():
            if val.__class__ in (Real, Integer, Categorical):
                search_spaces[par] = val
            else:
                fixed_params[par] = val

        self.estimator = SklEstimator(**fixed_params)
        self.results_filename = results_filename

        # para enganar o scikit-learn
        self.X = np.arange(10)
        vfold = np.zeros((10,), np.int)
        vfold[:5] = -1
        psplit = PredefinedSplit(vfold)

        super().__init__(self.estimator, search_spaces=search_spaces, optimizer_kwargs=dict(n_initial_points=n_initial),
                         n_iter=n_iter, n_jobs=n_jobs, scoring=scoring, fit_params=None, iid=iid, refit=False,
                         cv=psplit, verbose=verbose, pre_dispatch=pre_dispatch, random_state=random_state,
                         error_score=error_score, return_train_score=return_train_score)

    def fit(self, train_ds, valid_ds):
        self.fit_params = {
            'train_ds': train_ds,
            'valid_ds': valid_ds,
        }
        super().fit(self.X, y=None, groups=None)
        hps_results = self.cv_results_
        pickle.dump(hps_results, open(self.results_filename, 'wb'))
