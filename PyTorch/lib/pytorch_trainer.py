# -*- coding: utf-8 -*-
import os
import copy
import time
import pickle
import torch
import numpy as np
from torch.autograd import Variable


class DeepNetTrainer(object):
    
    def __init__(self, model=None, criterion=None, metrics=None, optimizer=None, lr_scheduler=None,
                 callbacks=None, use_gpu='auto'):
        
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
        
        self.use_gpu = use_gpu
        if use_gpu == 'auto':
            self.use_gpu = torch.cuda.is_available()

        if self.use_gpu:
            self.model.cuda()
        
    def fit(self, n_epochs, train_data, valid_data=None):

        self.has_validation = valid_data is not None

        try:
            mb_metrics = dict()

            for cb in self.callbacks:
                cb.on_train_begin(n_epochs)
                
            # for each epoch
            for i in range(self.last_epoch + 1, self.last_epoch + n_epochs + 1):
                
                # training phase
                # ==============
                for cb in self.callbacks:
                    cb.on_epoch_begin(i)
                    
                epo_samp = 0
                epo_loss = 0
                epo_metrics = dict([(n, 0) for n in self.compute_metric.keys()])

                self.model.train(True)
                if self.scheduler is not None:
                    self.scheduler.step()

                # for each minibatch
                for ii, (X, Y) in enumerate(train_data):
                    
                    for cb in self.callbacks:
                        cb.on_batch_begin(i, ii)

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
                        
                    epo_loss += mb_metrics['loss']
                    epo_samp += 1

                    for name, fun in self.compute_metric.items():
                        mb_metrics[name] = fun(Ypred, Y)
                        epo_metrics[name] += mb_metrics[name]
                    
                    for cb in self.callbacks:
                        cb.on_batch_end(i, ii, mb_metrics)
                    
                # end of training minibatches
                eloss = float(epo_loss / epo_samp)
                self.metrics['train']['losses'].append(eloss)

                for name, fun in self.compute_metric.items():
                    metric = float(epo_metrics[name] / epo_samp)
                    self.metrics['train'][name].append(metric)
                        
                # validation phase
                # ================
                if self.has_validation:
                    epo_samp = 0
                    epo_loss = 0
                    epo_metrics = dict([(n, 0) for n in self.compute_metric.keys()])
                    
                    self.model.train(False)
                    
                    # for each minibatch
                    for ii, (X, Y) in enumerate(valid_data):
                        if self.use_gpu:
                            X, Y = Variable(X.cuda()), Variable(Y.cuda())
                        else:
                            X, Y = Variable(X), Variable(Y)

                        Ypred = self.model.forward(X)
                        loss = self.criterion(Ypred, Y)

                        epo_loss += loss.data.cpu().numpy()
                        epo_samp += 1

                        for name, fun in self.compute_metric.items():
                            metric = fun(Ypred, Y)
                            epo_metrics[name] += metric
                    
                    #end minibatches
                    eloss = float(epo_loss / epo_samp)
                    self.metrics['valid']['losses'].append(eloss)
                    
                    for name, fun in self.compute_metric.items():
                        metric = float(epo_metrics[name] / epo_samp)
                        self.metrics['valid'][name].append(metric)
                
                else:
                    self.metrics['valid']['losses'].append(None)
                    for name, fun in self.compute_metric.items():
                        self.metrics['valid'][name].append(None)
                        
                for cb in self.callbacks:
                    cb.on_epoch_end(i, self.metrics)

        except KeyboardInterrupt:
            pass
            
        for cb in self.callbacks:
            cb.on_train_end(n_epochs, self.metrics)

    def predict(self, Xin):
        predictions = []
        try:
            self.model.train(False)  # Set model to evaluate mode
            ii_n = Xin.size(0)
            for ii, image in enumerate(Xin):
                if self.use_gpu:
                    image = Variable(image.cuda())
                else:
                    image = Variable(image)
                outputs = self.model.forward(image)
                predictions.append(outputs.data.cpu())
                print('\rpredict: {}/{}'.format(ii, ii_n - 1), end='')
            print(' ok')

        except KeyboardInterrupt:
            print(' interrupted!')

        finally:
            if len(predictions) > 0:
                return torch.cat(predictions, 0)

    def evaluate(self, Xin, yin, metrics=None):
        n_batches = 0
        epo_metrics = {}
        try:
            if metrics is None:
                metric_dict = self.compute_metric
            else:
                metric_dict = metrics
            for name in metric_dict.keys():
                epo_metrics[name] = 0
            self.model.train(False)
            ii_n = Xin.size(0)
            for ii, (X, Y) in enumerate(zip(Xin, yin)):
                if self.use_gpu:
                    X, Y = Variable(X.cuda()), Variable(Y.cuda())
                else:
                    X, Y = Variable(X), Variable(Y)
                Ypred = self.model.forward(X)
                for name, fun in metric_dict.items():
                    vmetric = fun(Ypred, Y)
                    epo_metrics[name] += vmetric
                n_batches += 1
                print('\revaluate: {}/{}'.format(ii, ii_n - 1), end='')
            print(' ok')

        except KeyboardInterrupt:
            print(' interrupted!')

        finally:
            if n_batches > 0:
                for name in epo_metrics.keys():
                    epo_metrics[name] /= n_batches
                return epo_metrics

    def predict_loader(self, data_loader):
        predictions = []
        try:
            self.model.train(False)  # Set model to evaluate mode
            ii_n = len(data_loader)
            for ii, (image, labels) in enumerate(data_loader):
                if self.use_gpu:
                    image = Variable(image.cuda())
                else:
                    image = Variable(image)
                outputs = self.model.forward(image)
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
        try:
            if metrics is None:
                metric_dict = self.compute_metric
            else:
                metric_dict = metrics
            for name in metric_dict.keys():
                epo_metrics[name] = 0
            self.model.train(False)
            ii_n = len(data_loader)
            for ii, (X, Y) in enumerate(data_loader):
                if self.use_gpu:
                    X, Y = Variable(X.cuda()), Variable(Y.cuda())
                else:
                    X, Y = Variable(X), Variable(Y)
                Ypred = self.model.forward(X)
                for name, fun in metric_dict.items():
                    vmetric = fun(Ypred, Y)
                    epo_metrics[name] += vmetric
                n_batches += 1
                print('\revaluate: {}/{}'.format(ii, ii_n - 1), end='')
            print(' ok')

        except KeyboardInterrupt:
            print(' interrupted!')

        finally:
            if n_batches > 0:
                for name in epo_metrics.keys():
                    epo_metrics[name] /= n_batches
                return epo_metrics


def load_trainer_state(file_basename, model, metrics):
    model.load_state_dict(torch.load(file_basename + '.model'))
    if os.path.isfile(file_basename + '.histo'):
        metrics.update(pickle.load(open(file_basename + '.histo', 'rb')))


def save_trainer_state(file_basename, model, metrics):
    torch.save(model.state_dict(), file_basename + '.model')
    pickle.dump(metrics, open(file_basename + '.histo', 'wb'))


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
    
    def on_batch_begin(self, epoch, batch):
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
            has_valid = metrics['valid']['losses'][-1] is not None
            has_metrics = len(metrics['train'].keys()) > 1
            etime = time.time() - self.t0

            is_best = ''
            valid_imin = int(np.argmin(self.trainer.metrics['valid']['losses'])) + 1
            train_imin = int(np.argmin(self.trainer.metrics['train']['losses'])) + 1
            if (has_valid and valid_imin == epoch) or (not has_valid and train_imin == epoch):
                is_best = 'best'

            if has_valid and has_metrics:
                # validation and metrics
                mtrc = list(self.trainer.compute_metric.keys())[0]
                print('{:3d}: {:5.1f}s   T: {:.5f} {:.5f}   V: {:.5f} {:.5f} {}'
                      .format(epoch, etime,
                              self.trainer.metrics['train']['losses'][-1],
                              self.trainer.metrics['train'][mtrc][-1],
                              self.trainer.metrics['valid']['losses'][-1],
                              self.trainer.metrics['valid'][mtrc][-1], is_best))
            elif has_valid:
                # validation and no metrics
                print('{:3d}: {:5.1f}s   T: {:.5f}   V: {:.5f} {}'
                      .format(epoch, etime,
                              self.trainer.metrics['train']['losses'][-1],
                              self.trainer.metrics['valid']['losses'][-1], is_best))
            elif not has_valid and has_metrics:
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
