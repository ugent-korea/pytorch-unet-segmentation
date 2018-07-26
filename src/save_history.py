import os
import csv
import torch


def export_history(header, value, folder, file_name):
    if not os.path.exists(folder):
        os.makedirs(folder)

    file_existence = os.path.isfile(file_name)
<<<<<<< HEAD
    if file_existence == False:
        file = open(file_name, 'w', newline='')
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerow(value)
    else:
        file = open(file_name, 'a', newline='')
=======
    if file_existence == False :
        file = open(file_name,'w',newline='')
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerow(value)
    else :
        file = open(file_name,'a',newline='')
>>>>>>> 281f3e6f260880d1f0e81e5bf1c030cafdf0ac85
        writer = csv.writer(file)
        writer.writerow(value)

    file.close()


def save_models(model, path, epoch):
    if not os.path.exists(path):
        os.makedirs(path)
    torch.save(model, path+"/model_epoch_{0}.pwf".format(epoch))
