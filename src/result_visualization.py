
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
    epoch_skip = epoch[::5]
    tr_loss_skip = tr_loss[::5]
    tr_acc_skip = tr_acc[::5]
    val_loss_skip = val_loss[::5]
    val_acc_skip = val_acc[::5]

    fig, ax1 = plt.subplots(figsize=(8, 6))
    ax2 = ax1.twinx()

    # Label and color the axes
    ax1.set_xlabel('Epoch', fontsize=16)
    ax1.set_ylabel('Loss', fontsize=16, color='black')
    ax2.set_ylabel('Accuracy', fontsize=16, color='black')

    # Plot valid/train losses
    ax1.plot(epoch_skip, tr_loss_skip, linewidth=2,
             ls='--', color='#c92508', label='Train loss')
    ax1.plot(epoch_skip, val_loss_skip, linewidth=2,
             color='#c92508', label='Validation loss')
    ax1.spines['left'].set_color('#f23d1d')
    # Coloring the ticks
    for label in ax1.get_yticklabels():
        label.set_color('#c92508')
        label.set_size(12)

    # Plot valid/trian accuracy
    ax2.plot(epoch_skip, tr_acc_skip, linewidth=2, ls='--',
             color='#2348ff', label='Train Accuracy')
    ax2.plot(epoch_skip, val_acc_skip, linewidth=2,
             color='#2348ff', label='Validation Accuracy')
    ax2.spines['right'].set_color('#2348ff')
    # Coloring the ticks
    for label in ax2.get_yticklabels():
        label.set_color('#2348ff')
        label.set_size(12)

    # Manually setting the y-axis ticks
    yticks = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    ax1.set_yticks(yticks)
    ax2.set_yticks(yticks)

    for label in ax1.get_xticklabels():
        label.set_size(12)

    # Modification of the overall graph
    fig.legend(ncol=4, loc=9, fontsize=12)
    plt.xlim(xmin=0)
    ax2.set_ylim(ymax=1, ymin=0)
    ax1.set_ylim(ymax=1, ymin=0)
    plt.xlabel('epochs')
    plt.title("Adam optimizer", weight="bold")
    plt.grid(True, axis='y')

    # return train_loss, valid_loss


if __name__ == '__main__':
    file = '../history/csv/Adam.csv'
    #file = '../history/SGD/history_SGD4.csv'
    plt.show(plotloss(file))
