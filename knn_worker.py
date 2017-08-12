import time, sys, queue
from multiprocessing.managers import BaseManager
import numpy

global train_data
global train_label
global test_data
global k
global train_m
global train_n
# 创建类似的QueueManager:
class QueueManager(BaseManager):
    pass

# 归一化，将数据中所有非零值置为1
def normalizing(data):
    m, n = numpy.shape(data)
    for i in range(m):
        for j in range(n):
            data[i, j] = (data[i, j] - 0) / 255.0
    return data

# 加载原始数据
def load_train_data():
    lines = numpy.loadtxt('train_short.csv', delimiter=',', dtype='str')
    img = lines[1:, 1:].astype('float')
    label = lines[1:, 0].astype('int')
    # label 42000*1 img 42000*784
    return normalizing(img), label

# 加载测试数据
def load_test_data():
    lines = numpy.loadtxt('test_short.csv', delimiter=',', dtype='str')
    img = lines[1:, :].astype('float')
    # img 28000*784
    return normalizing(img)

# knn分类器
def classify(sn):
    diff_mat = numpy.tile(test_data[sn, :], (train_m, 1)) - train_data
    # 在列方向将数据重复size次,行方向数据重复1次(不变)
    
    dist = (numpy.add.reduce((diff_mat) ** 2, axis=1)) ** 0.5
    ###
    class_ct = {}
    min_temp = -1
    min_list = []
    for i in range(k):  # 最相邻的k张图片的序号
        min_now_index = 0
        for j in range(train_m):
            if (dist[j] <= dist[min_now_index] and dist[j] >= min_temp):
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


if __name__ == '__main__':
    k=5
    train_data, train_label = load_train_data()
    train_m,train_n=numpy.shape(train_data)
    print(numpy.shape(train_data))
    
    test_data = load_test_data()
    test_m,test_n= numpy.shape(test_data)
    print(numpy.shape(test_data))
    
    # 由于这个QueueManager只从网络上获取Queue，所以注册时只提供名字:
    QueueManager.register('get_task_queue')
    QueueManager.register('get_result_queue')
    # 连接到服务器，也就是运行task_master.py的机器:
    server_addr = '127.0.0.1'
    print('Connect to server %s...' % server_addr)
    # 端口和验证码注意保持与task_master.py设置的完全一致:
    m = QueueManager(address=(server_addr, 5000), authkey=b'abc')
    # 从网络连接:
    m.connect()
    # 获取Queue的对象:
    task = m.get_task_queue()
    result = m.get_result_queue()
    print("begin work...")
    # 从task队列取任务,并把结果写入result队列:
    while True:
        try:
            sn = task.get(timeout=1)
            label = classify(sn)
            print('(%d,%d)' % (sn, label))
            result.put('%d,%d' % (sn, label))
            task.task_done()
        
        except Exception:
            print('Finish.')
            break
    
    # 处理结束:
    print('worker exit.')