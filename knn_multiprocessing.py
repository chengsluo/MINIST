import numpy as np
import multiprocessing as mp
import csv

train_data=None
train_label=None
test_data=None
test_label=None
k=None
train_m=None
train_n=None


# 归一化，将数据中所有非零值置为1
def normalizing(data):
    m, n = np.shape(data)
    for i in range(m):
        for j in range(n):
            data[i, j] = (data[i, j] - 0) / 255.0
    return data


def load_train_data():
    lines = np.loadtxt('train_short.csv', delimiter=',', dtype='str')
    img = lines[1:, 1:].astype('float')
    label = lines[1:, 0].astype('int')
    # label 4*1 img 4*784
    return normalizing(img), label


def load_test_data():
    lines = np.loadtxt('test_short.csv', delimiter=',', dtype='str')
    img = lines[1:, :].astype('float')
    # img 28000*784
    return normalizing(img)


def load_test_label():
    lines = np.loadtxt('knn_benchmark_short.csv', delimiter=',', dtype='str')
    return lines[1:, 1].astype('int')


def classify(sn):
    diff_mat = np.tile(test_data[sn, :], (train_m, 1)) - train_data
    # 在列方向将数据重复size次,行方向数据重复1次(不变)
    dist = (np.add.reduce((diff_mat) ** 2, axis=1)) ** 0.5
    ###
    class_ct = {}
    min_temp = -1
    min_list = []
    for i in range(k):  # 最相邻的k张图片的序号
        min_now_index = 0
        for j in range(train_m):
            if (dist[min_now_index] >= dist[j] >= min_temp):
                if (j not in min_list):
                    min_now_index = j
        label = train_label[min_now_index]
        min_list.append(min_now_index)
        min_temp = dist[min_now_index]
        class_ct[label] = class_ct.get(label, 0) + 1  # 通过字典获取类别及其次数
    sorted_class_ct = sorted(class_ct.items(), key=lambda item: item[1], reverse=True)
    ###
    # 　字典的值按降序排列
    return sorted_class_ct[0][0]  # 取字典的第一个的key,也就是个数最多的标签


def save_result(result):
    with open("result.csv", "w") as file:
        writer = csv.writer(file)
        writer.writerow(["ImageId", "Label"])
        for i in result:
            writer.writerow(i)

def job_handler(sn):
    # built-in id() returns unique memory ID of a variable
    label = classify(sn)
    print('%d,%d' % (sn + 1, label))
    return [sn + 1, label]

def launch_jobs(num_worker=mp.cpu_count()):
    global train_data
    global test_data
    global train_m
    global train_n
    print("processing begin")
    pool = mp.Pool(num_worker)
    return pool.map(job_handler, range(test_m))
    
if __name__ == '__main__':
    k = 5
    train_data, train_label = load_train_data()
    print(np.shape(train_data))
    
    test_data = load_test_data()
    print(np.shape(test_data))
    
    test_label = load_test_label()
    train_m, train_n = np.shape(train_data)
    test_m, test_n = np.shape(test_data)
 
    # create some random data and execute the child jobs
    results= launch_jobs()
    # print(results)

    # 保存所需结果
    results = sorted(results, key=lambda item:item[0], reverse=False)
    save_result(results)
    i = 0
    err_ct = 0
    for item in results:
        
        if item[1] != test_label[i]:
            err_ct += 1
        i += 1
    print("失误率:%f" % (err_ct / i))

    print("退出主进程")