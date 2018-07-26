import os
import csv
import torch


def export_history(header, value, folder, file_name):
    if not os.path.exists(folder):
        os.makedirs(folder)

    file_existence = os.path.isfile(file_name)
    if file_existence == False:
        file = open(file_name, 'w', newline='')
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerow(value)
    else:
        file = open(file_name, 'a', newline='')
        writer = csv.writer(file)
        writer.writerow(value)

    file.close()


def save_models(model, path, epoch):
    if not os.path.exists(path):
        os.makedirs(path)
    torch.save(model, path+"/model_epoch_{0}.pwf".format(epoch))
