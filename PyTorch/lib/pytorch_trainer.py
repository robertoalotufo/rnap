# -*- coding: utf-8 -*-
import os
import copy
import time
import pickle
import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader


class DeepNetTrainer(object):
    
    def __init__(self, model=None, criterion=None, metrics=None, optimizer=None, lr_scheduler=None,
                 callbacks=None, cbmetrics=None, use_gpu='auto'):
        
        assert (model is not None) and (criterion is not None) and (optimizer is not None)
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = lr_scheduler
        self.metrics = dict(train=dict(losses=[]), valid=dict(losses=[]))
        self.compute_metric = dict()
        self.last_epoch = 0
        
        if metrics is not None:
            for name, funct in metrics.items():
                self.metrics['train'][name] = []
                self.metrics['valid'][name] = []
                self.compute_metric[name] = funct

        self.callbacks = []
        if callbacks is not None:
            for cb in callbacks:
                self.callbacks.append(cb)
                cb.trainer = self

        self.cbmetrics = []
        if cbmetrics is not None:
            for cb in cbmetrics:
                self.cbmetrics.append(cb)
                cb.trainer = self

        self.use_gpu = use_gpu
        if use_gpu == 'auto':
            self.use_gpu = torch.cuda.is_available()

        if self.use_gpu:
            self.model.cuda()

    def _make_loader(self, Xin, yin, batch_size=10, shuffle=True):
        n_samples = Xin.size(0)
        n_batches = np.ceil(n_samples / batch_size)
        return self
        
    def fit(self, n_epochs, Xin, Yin, valid_data=None, valid_split=None, batch_size=10, shuffle=True):
        if valid_data is not None:
            train_loader = DataLoader(TensorDataset(Xin, Yin), batch_size=batch_size, shuffle=shuffle)
            valid_loader = DataLoader(TensorDataset(*valid_data), batch_size=batch_size, shuffle=shuffle)
        elif valid_split is not None:
            iv = int(valid_split * Xin.shape[0])
            Xval, Yval = Xin[:iv], Yin[:iv]
            Xtra, Ytra = Xin[iv:], Yin[iv:]
            train_loader = DataLoader(TensorDataset(Xtra, Ytra), batch_size=batch_size, shuffle=shuffle)
            valid_loader = DataLoader(TensorDataset(Xval, Yval), batch_size=batch_size, shuffle=shuffle)
        else:
            train_loader = DataLoader(TensorDataset(Xin, Yin), batch_size=batch_size, shuffle=shuffle)
            valid_loader = None
        self.fit_loader(n_epochs, train_loader, valid_data=valid_loader)

    def predict(self, Xin, batch_size=10):
        dloader = DataLoader(TensorDataset(Xin, Xin), batch_size=batch_size, shuffle=False)
        return self.predict_loader(dloader)

    def evaluate(self, Xin, Yin, metrics=None, batch_size=10):
        dloader = DataLoader(TensorDataset(Xin, Yin), batch_size=batch_size, shuffle=False)
        return self.evaluate_loader(dloader, metrics)

    def fit_loader(self, n_epochs, train_data, valid_data=None):
        self.has_validation = valid_data is not None
        try:
            # mini-batch metrics
            mb_metrics = dict()

            for cb in self.callbacks:
                cb.on_train_begin(n_epochs)

            # for each epoch
            for curr_epoch in range(self.last_epoch + 1, self.last_epoch + n_epochs + 1):

                # training phase
                # ==============
                for cb in self.callbacks:
                    cb.on_epoch_begin(curr_epoch)

                epo_samples = 0
                epo_batches = 0
                epo_loss = 0
                epo_metrics = dict([(n, 0) for n in self.compute_metric.keys()])

                self.model.train(True)
                if self.scheduler is not None:
                    self.scheduler.step()

                # for each minibatch
                for curr_batch, (X, Y) in enumerate(train_data):

                    mb_size = X.size(0)
                    epo_samples += mb_size
                    epo_batches += 1

                    for cb in self.callbacks:
                        cb.on_batch_begin(curr_epoch, curr_batch, mb_size)

                    if self.use_gpu:
                        X, Y = Variable(X.cuda()), Variable(Y.cuda())
                    else:
                        X, Y = Variable(X), Variable(Y)

                    self.optimizer.zero_grad()

                    Ypred = self.model.forward(X)
                    loss = self.criterion(Ypred, Y)
                    loss.backward()
                    self.optimizer.step()

                    mb_metrics['loss'] = loss.data.cpu().numpy()
                    if hasattr(self.criterion, 'size_average') and self.criterion.size_average:
                        epo_loss += mb_size * mb_metrics['loss']
                    else:
                        epo_loss += mb_metrics['loss']

                    for cb in self.cbmetrics:
                        cb.compute_batch_training_metric(curr_epoch, curr_batch, Ypred, Y)

                    for name, fun in self.compute_metric.items():
                        mb_metrics[name] = fun(Ypred, Y)
                        epo_metrics[name] += mb_metrics[name]

                    for cb in self.callbacks:
                        cb.on_batch_end(curr_epoch, curr_batch, mb_size, mb_metrics)

                # end of training minibatches
                eloss = float(epo_loss / epo_samples)
                self.metrics['train']['losses'].append(eloss)

                for name, fun in self.compute_metric.items():
                    metric = float(epo_metrics[name] / epo_samples)
                    self.metrics['train'][name].append(metric)

                # validation phase
                # ================
                if self.has_validation:
                    epo_samples = 0
                    epo_batches = 0
                    epo_loss = 0
                    epo_metrics = dict([(n, 0) for n in self.compute_metric.keys()])

                    self.model.train(False)

                    # for each minibatch
                    for curr_batch, (X, Y) in enumerate(valid_data):
                        mb_size = X.size(0)
                        epo_samples += mb_size
                        epo_batches += 1

                        if self.use_gpu:
                            X, Y = Variable(X.cuda()), Variable(Y.cuda())
                        else:
                            X, Y = Variable(X), Variable(Y)

                        Ypred = self.model.forward(X)
                        loss = self.criterion(Ypred, Y)

                        vloss = loss.data.cpu().numpy()
                        if hasattr(self.criterion, 'size_average') and self.criterion.size_average:
                            epo_loss += vloss * mb_size
                        else:
                            epo_loss += vloss

                        for cb in self.cbmetrics:
                            cb.compute_batch_validation_metric(curr_epoch, curr_batch, Ypred, Y)

                        for name, fun in self.compute_metric.items():
                            metric = fun(Ypred, Y)
                            epo_metrics[name] += metric

                    #end minibatches
                    eloss = float(epo_loss / epo_samples)
                    self.metrics['valid']['losses'].append(eloss)

                    for name, fun in self.compute_metric.items():
                        metric = float(epo_metrics[name] / epo_samples)
                        self.metrics['valid'][name].append(metric)

                else:
                    self.metrics['valid']['losses'].append(None)
                    for name, fun in self.compute_metric.items():
                        self.metrics['valid'][name].append(None)

                for cb in self.cbmetrics:
                    cb.compute_epoch_metrics(curr_epoch)

                for cb in self.callbacks:
                    cb.on_epoch_end(curr_epoch, self.metrics)

        except KeyboardInterrupt:
            pass

        for cb in self.callbacks:
            cb.on_train_end(n_epochs, self.metrics)

    def predict_loader(self, data_loader):
        predictions = []
        try:
            self.model.train(False)  # Set model to evaluate mode
            ii_n = len(data_loader)
            for ii, (X, _) in enumerate(data_loader):
                if self.use_gpu:
                    X = Variable(X.cuda())
                else:
                    X = Variable(X)
                outputs = self.model.forward(X)
                predictions.append(outputs.data.cpu())
                print('\rpredict: {}/{}'.format(ii, ii_n - 1), end='')
            print(' ok')

        except KeyboardInterrupt:
            print(' interrupted!')

        finally:
            if len(predictions) > 0:
                return torch.cat(predictions, 0)

    def evaluate_loader(self, data_loader, metrics=None):
        n_batches = 0
        epo_metrics = {}
        if metrics is None:
            metric_dict = self.compute_metric
        else:
            metric_dict = metrics

        epo_samples = 0
        epo_batches = 0
        epo_loss = 0
        for name in metric_dict.keys():
            epo_metrics[name] = 0

        try:
            self.model.train(False)
            ii_n = len(data_loader)

            for ii, (X, Y) in enumerate(data_loader):
                mb_size = X.size(0)
                epo_samples += mb_size
                epo_batches += 1

                if self.use_gpu:
                    X, Y = Variable(X.cuda()), Variable(Y.cuda())
                else:
                    X, Y = Variable(X), Variable(Y)

                Ypred = self.model.forward(X)
                loss = self.criterion(Ypred, Y)

                vloss = loss.data.cpu().numpy()
                if hasattr(self.criterion, 'size_average') and self.criterion.size_average:
                    epo_loss += vloss * mb_size
                else:
                    epo_loss += vloss

                for name, fun in metric_dict.items():
                    vmetric = fun(Ypred, Y)
                    epo_metrics[name] += vmetric

                print('\revaluate: {}/{}'.format(ii, ii_n - 1), end='')
            print(' ok')

        except KeyboardInterrupt:
            print(' interrupted!')

        if n_batches > 0:
            epo_loss /= epo_samples
            for name in epo_metrics.keys():
                epo_metrics[name] /= epo_samples

            return epo_loss, epo_metrics

    def load_state(self, file_basename):
        load_trainer_state(file_basename, self.model, self.metrics)

    def save_state(self, file_basename):
        save_trainer_state(file_basename, self.model, self.metrics)

    def summary(self):
        pass


def load_trainer_state(file_basename, model, metrics):
    model.load_state_dict(torch.load(file_basename + '.model'))
    if os.path.isfile(file_basename + '.histo'):
        metrics.update(pickle.load(open(file_basename + '.histo', 'rb')))


def save_trainer_state(file_basename, model, metrics):
    torch.save(model.state_dict(), file_basename + '.model')
    pickle.dump(metrics, open(file_basename + '.histo', 'wb'))


class MetricCallback(object):
    def __init__(self):
        pass

    def reset_metrics(self):
        pass

    def compute_batch_training_metric(self, epoch_num, batch_num, y_pred, y_true):
        pass

    def compute_batch_validation_metric(self, epoch_num, batch_num, y_pred, y_true):
        pass

    def compute_epoch_metrics(self, epoch_num):
        pass


class AccuracyMetric(MetricCallback):
    def __init__(self):
        super().__init__()
        self.name = 'accuracy'
        self.reset_metrics()

    def reset_metrics(self):
        self.train_accum = 0
        self.valid_accum = 0
        self.n_train_samples = 0
        self.n_valid_samples = 0

    def compute_batch_training_metric(self, epoch_num, batch_num, y_pred, y_true):
        _, preds = torch.max(y_pred.data, 1)
        self.train_accum += (preds == y_true.data).type(torch.FloatTensor).sum()
        self.n_train_samples += y_pred.size(0)

    def compute_batch_validation_metric(self, epoch_num, batch_num, y_pred, y_true):
        _, preds = torch.max(y_pred.data, 1)
        self.valid_accum += (preds == y_true.data).type(torch.FloatTensor).sum()
        self.n_valid_samples += y_pred.size(0)

    def compute_epoch_metrics(self, epoch_num):
        if epoch_num == 1:
            self.trainer.metrics['train'][self.name] = []
            self.trainer.metrics['valid'][self.name] = []
        if self.n_train_samples > 0:
            self.trainer.metrics['train'][self.name].append(1.0 * self.train_accum / self.n_train_samples)
        if self.n_valid_samples > 0:
            self.trainer.metrics['valid'][self.name].append(1.0 * self.valid_accum / self.n_valid_samples)
        self.reset_metrics()


class Callback(object):
    def __init__(self):
        pass
    
    def on_train_begin(self, n_epochs):
        pass
    
    def on_train_end(self, n_epochs, metrics):
        pass

    def on_epoch_begin(self, epoch):
        pass

    def on_epoch_end(self, epoch, metrics):
        pass
    
    def on_batch_begin(self, epoch, batch, batch_size):
        pass

    def on_batch_end(self, epoch, batch, mb_metrics):
        pass


class ModelCheckpoint(Callback):

    def __init__(self, model_basename, reset=False, verbose=0):
        super().__init__()
        os.makedirs(os.path.dirname(model_basename), exist_ok=True)
        self.basename = model_basename
        self.reset = reset
        self.verbose = verbose

    def on_train_begin(self, n_epochs):
        if (self.basename is not None) and (not self.reset) and (os.path.isfile(self.basename + '.model')):
            load_trainer_state(self.basename, self.trainer.model, self.trainer.metrics)
            if self.verbose > 0:
                print('Model loaded from', self.basename + '.model')

        self.trainer.last_epoch = len(self.trainer.metrics['train']['losses'])
        if self.trainer.scheduler is not None:
            self.trainer.scheduler.last_epoch = self.trainer.last_epoch

        self.best_model = copy.deepcopy(self.trainer.model)
        self.best_epoch = self.trainer.last_epoch
        self.best_loss = 1e10
        if self.trainer.last_epoch > 0:
            self.best_loss = self.trainer.metrics['valid']['losses'][-1] or \
                             self.trainer.metrics['train']['losses'][-1]
            
    def on_train_end(self, n_epochs, metrics):
        print('Best model was saved at epoch {} with loss {:.5f}: {}'
              .format(self.best_epoch, self.best_loss, self.basename))

    def on_epoch_end(self, epoch, metrics):
        eloss = metrics['valid']['losses'][-1] or metrics['train']['losses'][-1]
        if eloss < self.best_loss:
            self.best_loss = eloss
            self.best_epoch = epoch
            self.best_model = copy.deepcopy(self.trainer.model)
            if self.basename is not None:
                save_trainer_state(self.basename, self.trainer.model, self.trainer.metrics)
                if self.verbose > 1:
                    print('Model saved to', self.basename + '.model')


class PrintCallback(Callback):

    def __init__(self):
        super().__init__()

    def on_train_begin(self, n_epochs):
        print('Start training for {} epochs'.format(n_epochs))
    
    def on_train_end(self, n_epochs, metrics):
        n_train = len(metrics['train']['losses'])
        print('Stop training at epoch: {}/{}'.format(n_train, self.trainer.last_epoch + n_epochs))

    def on_epoch_begin(self, epoch):
        self.t0 = time.time()

    def on_epoch_end(self, epoch, metrics):
            is_best = ''
            has_valid = metrics['valid']['losses'][-1] is not None
            has_metrics = len(metrics['train'].keys()) > 1
            etime = time.time() - self.t0

            if has_valid:
                if epoch == int(np.argmin(self.trainer.metrics['valid']['losses'])) + 1:
                    is_best = 'best'
                if has_metrics:
                    # validation and metrics
                    mtrc = list(self.trainer.compute_metric.keys())[0]
                    print('{:3d}: {:5.1f}s   T: {:.5f} {:.5f}   V: {:.5f} {:.5f} {}'
                          .format(epoch, etime,
                                  self.trainer.metrics['train']['losses'][-1],
                                  self.trainer.metrics['train'][mtrc][-1],
                                  self.trainer.metrics['valid']['losses'][-1],
                                  self.trainer.metrics['valid'][mtrc][-1], is_best))
                else:
                    # validation and no metrics
                    print('{:3d}: {:5.1f}s   T: {:.5f}   V: {:.5f} {}'
                          .format(epoch, etime,
                                  self.trainer.metrics['train']['losses'][-1],
                                  self.trainer.metrics['valid']['losses'][-1], is_best))
            else:
                if epoch == int(np.argmin(self.trainer.metrics['train']['losses'])) + 1:
                    is_best = 'best'
                if has_metrics:
                    # no validation and metrics
                    mtrc = list(self.trainer.compute_metric.keys())[0]
                    print('{:3d}: {:5.1f}s   T: {:.5f} {:.5f} {}'
                          .format(epoch, etime,
                                  self.trainer.metrics['train']['losses'][-1],
                                  self.trainer.metrics['train'][mtrc][-1], is_best))
                else:
                    # no validation and no metrics
                    print('{:3d}: {:5.1f}s   T: {:.5f} {}'
                          .format(epoch, etime,
                                  self.trainer.metrics['train']['losses'][-1], is_best))
