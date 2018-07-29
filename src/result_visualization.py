
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

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    # Label loss graph with dash
    ax1.set_xlabel('Epoch', fontsize=13)
    ax1.set_ylabel('Loss', fontsize=13)
    ax2.set_ylabel('Accuracy', fontsize=13)

    ax1.plot(epoch_skip, tr_loss_skip, linewidth=2,
             ls='--', color='#0059ba', label='Train loss')
    ax1.plot(epoch_skip, val_loss_skip, linewidth=2, ls='--',
             color='#47ba00', label='Validation loss')
    # Label acc graph with regular line
    ax2.plot(epoch_skip, tr_acc_skip, linewidth=2,
             color='#0059ba', label='Train Accuracy')
    ax2.plot(epoch_skip, val_acc_skip, linewidth=2,
             color='#47ba00', label='Validation Accuracy')

    # Red line to indicate the optimal epoch (absolute value)
    # can resize the font size of legend by modifying 'size'
    fig.legend(ncol=4, loc=9, fontsize=12)
    plt.xlim(xmin=0)
    ax2.set_ylim(ymax=0.98, ymin=0.77)
    ax1.set_ylim(ymin=0.05, ymax=0.6)
    # ax1.set_ylim(ymax=)
    plt.xlabel('epochs')
    plt.title("SGD optimizer", weight="bold")
    plt.grid(True, axis='y')

    # return train_loss, valid_loss


if __name__ == '__main__':
    file = '../history/SGD/history_SGD4.csv'
    #file = '../history/SGD/history_SGD4.csv'
    plt.show(plotloss(file))
