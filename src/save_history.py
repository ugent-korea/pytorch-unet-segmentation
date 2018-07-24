import os
import csv

def export_history(header,value,folder,file_name) :
    if not os.path.exists(folder) :
        os.makedirs(folder)

    file_existence = os.path.isfile(file_name)
    if file_existence == False :
        file = open(file_name,'w',newline='')
        writer = csv.writer(file_name)
        writer.writerow(header)
        writer.writerow(value)
    else :
        file = open(file_name,'a',newline='')
        writer = csv.writer(file_name)
        writer.writerow(value)

    file.close()
