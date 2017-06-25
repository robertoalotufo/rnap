
# coding: utf-8

# In[1]:

import os, time
import numpy as np
import pickle
from IPython import display
import matplotlib.pyplot as plot
from keras.models import load_model
from keras.callbacks import Callback

try:
    from tensorflow.python.client import device_lib
    def get_available_gpus():
        local_device_protos = device_lib.list_local_devices()
        return [x.name for x in local_device_protos if x.device_type == 'GPU']
except:
    pass

def load_model_and_history(name):
    model_fn = name + '.model'
    history_fn = name + '.history'
    model, callback = None, None
    if os.path.isfile(model_fn):
        model = load_model(model_fn)
        if os.path.isfile(history_fn):
            callback = pickle.load(open(history_fn, 'rb'))
    return model, callback

def save_model_and_history(name, model, histo):
    model.save(name + '.model', overwrite=True)
    pickle.dump(histo, open(name + '.history', 'wb'))
    
class TrainingPlotter(Callback):
    """
    History + ModelCheckpoint + EarlyStopping + PlotLosses
    """
    def __init__(self, n=1, filepath=None, patience=10, axis=None, plot_losses=True):
        self.history = []
        self.best_loss = np.inf
        self.best_epoch = 0
        self.filepath = filepath
        self.patience = patience
        self.plot_losses = plot_losses
        
        self.n = n
        self.line1 = None
        self.line2 = None
        self.axis = axis
                
    def __getstate__(self):
        # we do not want to pickle the matplotlib line1/line2/axis
        return dict(n=self.n, history=self.history, best_loss=self.best_loss, 
                    best_epoch=self.best_epoch, filepath=self.filepath, 
                    plot_losses=self.plot_losses, patience=self.patience, 
                    line1=None, line2=None, axis=None)

    def get_nepochs(self):
        return len(self.history)
    
    def on_epoch_begin(self, epoch, logs={}):
        self.epoch_t0 = time.time()

    def checkpoint(self, epoch, value):
        if value < self.best_loss:
            self.best_loss = value
            self.best_epoch = epoch
            if self.filepath is not None:
                save_model_and_history(self.filepath, self.model, self)
        
    def early_stop(self, value):
        early_stop_msg = ''
        if value < self.best_loss:
            self.waiting = 0
        else:
            self.waiting += 1
            if self.waiting > self.patience:
                self.model.stop_training = True
                early_stop_msg = 'Early Stopped.'
        return early_stop_msg
    
    def on_epoch_end(self, epoch, logs={}):
        # {'acc': 0.97, 'loss': 0.08, 'val_acc': 0.98, 'val_loss': 0.06}
        epoch += self.nepochs
        epoch_time = time.time() - self.epoch_t0
        self.history.append(logs)
        
        if 'val_loss' in logs.keys():
            chk_value = logs['val_loss']
            val = True
            lab = 'validation'
        else:
            chk_value = logs['loss']
            val = False
            lab = 'training'

        early_stop_msg = self.early_stop(chk_value)
        self.checkpoint(epoch, chk_value)
        
        if not self.plot_losses:
            return
            
        if self.axis is None:
            self.axis = plot

        try:
            if (len(self.history) % self.n) == 0:
                htrain = np.array([v['loss'] for v in self.history], np.float32)
                if val:
                    hvalid = np.array([v['val_loss'] for v in self.history], np.float32)

                if self.line2 is None:
                    self.line2 = self.axis.plot(htrain, linewidth=2, label='training mse')[0]
                    if val:
                        self.line1 = self.axis.plot(hvalid, linewidth=2, label='validation mse')[0]
                    self.axis.vlines(self.best_epoch, 0, 100, colors='#EBDDE2', linestyles='dashed', 
                                     label='{} min'.format(lab))
                else:
                    self.line2.set_xdata(np.arange(htrain.shape[0]))
                    self.line2.set_ydata(htrain)
                    if val:
                        self.line1.set_xdata(np.arange(hvalid.shape[0]))
                        self.line1.set_ydata(hvalid)
                    self.axis.vlines(self.best_epoch, 0, 100, colors='#EBDDE2', linestyles='dashed')
                    
                self.axis.legend()
                if 'val_acc' in logs.keys():
                    acc = ' Accuracy = {:.2f}%'.format(100.0 * self.history[self.best_epoch]['val_acc'])
                else:
                    acc = ''

                self.axis.title('{} Best loss is {:.5f} on epoch {:d}.{}'.format(early_stop_msg, self.best_loss, 
                                                                                 self.best_epoch, acc), weight='bold')
                if val:
                    self.axis.ylabel('Losses [{:.5f} / {:.5f}]'.format(htrain[-1], hvalid[-1]))
                else:
                    self.axis.ylabel('Loss {:.5f}'.format(htrain[-1]))
                
                self.axis.xlabel('Epoch [{}: {:.2f}s]'.format(epoch, epoch_time))
                
                display.display(plot.gcf())
                display.clear_output(wait=True)

        except Exception as e:
            print('=*' * 40)
            print('Error while trying to plot losses...')
            print(e)
            print('*=' * 40)
            raise
            
    def on_batch_begin(self, batch, logs={}):
        pass

    def on_batch_end(self, batch, logs={}):
        pass

    def on_train_begin(self, logs={}):
        self.nepochs = len(self.history)
        self.waiting = 0

    def on_train_end(self, logs={}):
        pass


  



