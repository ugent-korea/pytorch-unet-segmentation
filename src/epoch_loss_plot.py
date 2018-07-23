
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np

'''
For members who did not yet install the module "matplotlib",

    python3 -mpip install -U matplotlib

In case your system has error: no module tkinter,

    sudo apt install python3-tk

if you have python version >= 3.6,

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
    epoch = loss_values.iloc[:,0]
    tr_loss = loss_values.iloc[:,1]
    val_loss = np.asarray(loss_values.iloc[:,2])

    # Reduce the volume of data
    epoch_skip = epoch[::10]
    tr_loss_skip = tr_loss[::10]
    val_loss_skip = val_loss[::10]

    train_loss = plt.plot(epoch_skip, tr_loss_skip, 'b-', label='Train loss')
    valid_loss = plt.plot(epoch_skip, val_loss_skip, 'g-', label='Validation loss')

    # Red line to indicate the optimal epoch (absolute value)
    minimum = np.argmin(val_loss)
    optimal = plt.axvline(x=minimum, color='r', label='Optimal epoch')

    plt.legend()

    plt.xlabel('epochs')
    plt.ylabel('loss')

    return train_loss, valid_loss

if __name__ == '__main__':
    file = 'progress_report.csv'
    plt.show(plotloss(file))
