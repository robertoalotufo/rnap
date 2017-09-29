import copy
import time
import pickle
import torch
import numpy as np
from torch.autograd import Variable

class DeepNetTrainer:
    
    def __init__(self, file_basename=None, model=None, criterion=None, metrics=None, 
                 optimizer=None, lr_scheduler=None, reset=False):
        assert (model is not None) and (criterion is not None) and (optimizer is not None)
        self.basename = file_basename
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = lr_scheduler
        self.metrics = dict(train=dict(losses=[]), valid=dict(losses=[]))
        self.compute_metric = dict()
        
        if metrics is not None:
            for name, funct in metrics.items():
                self.metrics['train'][name] = []
                self.metrics['valid'][name] = []
                self.compute_metric[name] = funct
        
        if (self.basename is not None) and (not reset) and (os.path.isfile(self.basename + '.model')):
            self.load_trainer_state(self.basename, self.model, self.optimizer, self.metrics)
            print('Model loaded from', self.basename + '.model')
            
        self.last_epoch = len(self.metrics['train']['losses'])
        if self.scheduler is not None:
            self.scheduler.last_epoch = self.last_epoch
            
    def fit(self, n_epochs, train_data, valid_data=None, use_gpu='auto'):
        data = dict(train=train_data, valid=valid_data)
        if valid_data is None:
            phases = [('train', True)]
        else:
            phases = [('train', True), ('valid', False)]
     
        if use_gpu == 'auto':
            use_gpu = torch.cuda.is_available()
        assert use_gpu == False or use_gpu == True
            
        try:
            best_model = copy.deepcopy(self.model)
            best_loss = 1e10
            best_epoch = self.last_epoch

            print('Starting training for {} epochs'.format(n_epochs))
            for i in range(self.last_epoch + 1, self.last_epoch + n_epochs + 1):
                t0 = time.time()
                
                for phase, is_train in phases:
                
                    epo_samp = 0
                    epo_loss = 0
                    epo_metrics = dict([(n, 0) for n in self.compute_metric.keys()])
                    
                    self.model.train(is_train)
                    if is_train:
                        self.scheduler.step()

                    for ii, (X, Y) in enumerate(data[phase]):
                        if use_gpu:
                            X, Y = Variable(X.cuda()), Variable(Y.cuda())
                        else:
                            X, Y = Variable(X), Variable(Y)

                        Ypred = self.model.forward(X)
                        loss = self.criterion(Ypred, Y)
                        if is_train:
                            self.optimizer.zero_grad()
                            loss.backward()
                            self.optimizer.step()

                        epo_loss += loss.data.cpu().numpy()
                        epo_samp += 1

                        for name, fun in self.compute_metric.items():
                            metric = fun(Ypred.data, Y.data)
                            epo_metrics[name] += metric

                    eloss = float(epo_loss / epo_samp)
                    self.metrics[phase]['losses'].append(eloss)
                    
                    for name, fun in self.compute_metric.items():
                        metric = float(epo_metrics[name] / epo_samp)
                        self.metrics[phase][name].append(metric)

                if valid_data is None:
                    self.metrics['valid']['losses'].append(None)
                    for name, fun in self.compute_metric.items():
                        self.metrics['valid'][name].append(None)
                        
                is_best = ''
                if eloss < best_loss:
                    is_best = 'best'
                    best_loss = eloss
                    best_epoch = i
                    best_model = copy.deepcopy(self.model)
                    if self.basename is not None:
                        self.save_trainer_state(self.basename, self.model, self.optimizer, self.metrics)

                if (valid_data is not None) and (len(self.compute_metric) > 0):
                    # validation and metrics
                    mtrc = list(self.compute_metric.keys())[0]
                    print('{:3d}: {:5.1f}s  T: {:10.5f} {:5.2f}  V: {:10.5f} {:5.2f} {}'
                          .format(i, time.time() - t0, 
                                  self.metrics['train']['losses'][-1], self.metrics['train'][mtrc][-1],
                                  self.metrics['valid']['losses'][-1], self.metrics['valid'][mtrc][-1], is_best))
                elif (valid_data is not None):
                    # validation and no metrics
                    print('{:3d}: {:5.1f}s  T: {:10.5f}  V: {:10.5f} {}'
                          .format(i, time.time() - t0, self.metrics['train']['losses'][-1], 
                                                       self.metrics['valid']['losses'][-1], is_best))
                elif (valid_data is None) and (len(self.compute_metric) > 0):
                    # no validation and metrics
                    mtrc = list(self.compute_metric.keys())[0]
                    print('{:3d}: {:5.1f}s  T: {:10.5f} {:5.2f} {}'
                          .format(i, time.time() - t0, self.metrics['train']['losses'][-1], 
                                                       self.metrics['train'][mtrc][-1], is_best))
                else:
                    # no validation and no metrics
                    print('{:3d}: {:5.1f}s  T: {:10.5f} {}'
                          .format(i, time.time() - t0, self.metrics['train']['losses'][-1], is_best))
                
                t0 = time.time()

        except KeyboardInterrupt:
            print('Interrupted!!')

        print('\nModel from epoch {} saved as {}.*, loss = {:.5f}'.format(best_epoch, self.basename, best_loss))

    @staticmethod
    def load_trainer_state(file_basename, model, optimizer, metrics):
        model.load_state_dict(torch.load(file_basename + '.model'))
        if os.path.isfile(file_basename + '.optim'):
            optimizer.load_state_dict(torch.load(file_basename + '.optim'))
        if os.path.isfile(file_basename + '.histo'):
            metrics.update(pickle.load(open(file_basename + '.histo', 'rb')))
    
    @staticmethod
    def save_trainer_state(file_basename, model, optimizer, metrics):
        torch.save(model.state_dict(), file_basename + '.model')
        torch.save(optimizer.state_dict(), file_basename + '.optim')
        pickle.dump(metrics, open(file_basename + '.histo', 'wb'))

        
def test_network(model, dataset, criterion, batch_size=32, use_gpu='auto'):
    temp_dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                  shuffle=False, num_workers=4)
    
    # desliga o treinamento para nao executar o dropout
    model.train(False)
    
    if use_gpu == 'auto':
        use_gpu = torch.cuda.is_available()
    assert use_gpu == False or use_gpu == True
    
    loss_sum = 0.0
    hit_sum = 0.0
    all_preds = np.zeros(len(dataset)).astype(int)
    all_probs = np.zeros(len(dataset)).astype(float)
    for i, data in enumerate(temp_dataloader):
        inputs, labels = data
        
        if use_gpu:
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        else:
            inputs, labels = Variable(inputs), Variable(labels)
    
        outputs = model(inputs)
        loss_sum += criterion(outputs, labels)

        outputs = torch.nn.functional.softmax(outputs)
        probs, preds = torch.max(outputs, 1)
        curr_img_index = i*temp_dataloader.batch_size
        all_preds[curr_img_index:curr_img_index+labels.size(0)] = preds.data.cpu().numpy()
        all_probs[curr_img_index:curr_img_index+labels.size(0)] = probs.data.cpu().numpy()
        hit_sum += torch.sum(preds.data==labels.data)
    
    loss = loss_sum.data.cpu()[0] / len(temp_dataloader)
    accuracy = hit_sum / len(dataset)
    
    print("\nAccuracy on the test data set: {:.2f}% [{:.5f}]".format(accuracy * 100, loss))
    return (all_preds, all_probs)
