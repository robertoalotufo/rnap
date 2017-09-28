from torchvision.datasets import CIFAR10
import numpy as np

CIFAR_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

class CIFARX(CIFAR10):
    def __init__(self, root, classes, train=True, **kwargs):
        super(CIFARX, self).__init__(root, train, **kwargs)
        
        # check wrong parameters
        if not isinstance(classes, list):
            raise Exception('Argument classes must be a list')
        
        if max(classes) > 9 or min(classes) < 0:
            raise Exception('Value of elements in classes must be in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]')
        
        new_data = []
        new_labels = []
        if train:
            for i, label in enumerate(self.train_labels):
                if label in classes:
                    new_data.append(self.train_data[i])
                    new_labels.append(label)
                    
            new_data = np.array(new_data)
            self.train_data = new_data
            self.train_labels = new_labels
        else:  # test set
            for i, label in enumerate(self.test_labels):
                if label in classes:
                    new_data.append(self.test_data[i])
                    new_labels.append(label)

            new_data = np.array(new_data)
            self.test_data = new_data
            self.test_labels = new_labels           
        
        self.classes = np.array(CIFAR_CLASSES)[classes]