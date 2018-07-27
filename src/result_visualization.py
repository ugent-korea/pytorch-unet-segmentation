
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np

'''
For members who did not yet install the module "matplotlib",

    python3 -mpip install -U matplotlib

installation of tkinter is a prerequisite. If you do not have it,

    sudo apt install python3.6-tk

Now you won't have problems running this python file.
'''


def plotloss(csvfile):
    '''
    Args
        csvfile: name of the csv file

    Returns
        graph_loss: trend of loss values over epoch
    '''

    # Bring in the csv file
    loss_values = pd.read_csv(csvfile)

    # Initiation
    epoch = loss_values.iloc[:, 0]
    tr_loss = loss_values.iloc[:, 1]
    tr_acc = loss_values.iloc[:, 2]
    val_loss = np.asarray(loss_values.iloc[:, 3])
    val_acc = np.asarray(loss_values.iloc[:, 4])

    # Reduce the volume of data
    epoch_skip = epoch[::20]
    tr_loss_skip = tr_loss[::20]
    tr_acc_skip = tr_acc[::20]
    val_loss_skip = val_loss[::20]
    val_acc_skip = val_acc[::20]

    # Label loss graph with dash
    train_loss = plt.plot(epoch_skip, tr_loss_skip, linewidth=2,
                          ls='--', color='#0059ba', label='Train loss')
    valid_loss = plt.plot(epoch_skip, val_loss_skip, linewidth=2, ls='--',
                          color='#47ba00', label='Validation loss')
    # Label acc graph with regular line
    train_acc = plt.plot(epoch_skip, tr_acc_skip, linewidth=2,
                         color='#0059ba', label='Train Accuracy')
    valid_acc = plt.plot(epoch_skip, val_acc_skip, linewidth=2,
                         color='#47ba00', label='Validation Accuracy')

    # Red line to indicate the optimal epoch (absolute value)

    plt.legend()
    plt.xlim(xmin=0)
    plt.xlabel('epochs')

    return train_loss, valid_loss


if __name__ == '__main__':
    file = '../history/RMS/history_RMS.csv'
    plt.show(plotloss(file))
