import numpy,csv,time
from multiprocessing import managers,Queue,JoinableQueue
#--------远程任务队列配置-------
# 发送任务的队列:
task_queue = JoinableQueue()
# 接收结果的队列:
result_queue = Queue()

# 从BaseManager继承的QueueManager:
class QueueManager(managers.BaseManager):
    pass

# 把两个Queue都注册到网络上, callable参数关联了Queue对象:
QueueManager.register('get_task_queue', callable=lambda: task_queue)
QueueManager.register('get_result_queue', callable=lambda: result_queue)
# 绑定端口5000, 设置验证码'abc':
manager = QueueManager(address=('', 5000), authkey=b'abc')
# 启动Queue:
manager.start()
# 获得通过网络访问的Queue对象:
task = manager.get_task_queue()
result = manager.get_result_queue()

#------------master数据加载---------

# 加载数据测试结果
def load_test_label():
    lines = numpy.loadtxt('knn_benchmark_short.csv', delimiter=',', dtype='str')
    return lines[1:, 1].astype('int')

test_label=load_test_label()
scale=test_label.shape[0]
# 将任务编号放入任务队列
for i in range(scale):
    task.put(i)

#-----------获取结果-------------
# 从result队列读取结果:
print('Try get results...,Please run some worker for this')

results=[]
task_queue.join()
while True:
    if result.empty():
        break
    item=result.get()
    print(item)
    results.append(item)
# 将结果排序后，存入文件
results = sorted(results, key=lambda item: int(item.split(',')[0]), reverse=False)
with open("result.csv", "w") as file:
    writer = csv.writer(file)
    writer.writerow(["ImageId", "Label"])
    for i in results:
        writer.writerow(i.split(','))

# 统计错误率
i = 0
err_ct = 0
for item in results:
    label = int(item.split(',')[1])
    # print(item)
    if label != test_label[i]:
        err_ct += 1
    i += 1
print("失误率:%f" % (err_ct / i))

# 关闭multiprocessing.manager
manager.shutdown()
print('master exit.')