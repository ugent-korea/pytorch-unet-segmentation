import os
import csv

def export_history(header,value,folder,file) :
    if not os.path.exists(folder) :
        os.makedirs(folder)

    file_existence = os.path.isfile(file)
    if file_existence == False :
        file = open(file,'w',newline='')
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerow(value)
    else :
        file = open(file,'a',newline='')
        writer = csv.writer(file)
        writer.writerow(value)

    file.close()

a = export_history(['epoch','train','validation'],[1,0.6,0.7],'/home/mijeong/Desktop/internship/pytorch-unet-segmentation/src','history.csv')
