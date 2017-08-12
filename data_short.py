#!/usr/bin/python3
# _*_coding:utf-8_*_

import csv

def short_data(csv_name,need_row_num):
    with open(csv_name+'.csv','r') as file:
        rows=csv.reader(file)
        # for row in rows:
        #     if i>=need_row_num:break
        #     print(row)
        with open(csv_name+'_short.csv','w') as file:
            i = 0
            writer=csv.writer(file)
            for row in rows:
                if i>=need_row_num:break
                i=i+1
                writer.writerow(row)

if __name__=='__main__':
    size=3000
    short_data("test",size)
    short_data("train",size)
    short_data("knn_benchmark",size)

