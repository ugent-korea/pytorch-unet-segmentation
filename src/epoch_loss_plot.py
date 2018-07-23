
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd

'''
For members who did not yet install the module "matplotlib",

    python3 -mpip install -U matplotlib

In case your system has error: no module tkinter,

    sudo apt install python3-tk

if you have python version >= 3.6,

    sudo apt install python3.6-tk

Now you won't have problems running this python file.
'''

loss_values = pd.read_csv('train_loss.csv')
#SANITY CHECK
print(loss_values.head)

loss_values.plot()
