import numpy
import csv
import queue
import threading

exitFlag = 0
results = []


def to_int(array):
	array = numpy.mat(array)  # 将数组转化为矩阵
	m, n = numpy.shape(array)
	# 返回数组各维长度.例如二维数组中，表示将字符型转换为浮点型数组的“行数”和“列数”
	new_array = numpy.zeros((m, n))  # 深复制数据
	print("%d,%d" % (m, n))
	for i in range(m):
		for j in range(n):
			new_array[i][j] = int(array[i, j])
	return new_array


# 归一化，将数据中所有非零值置为1
def normalizing(array):
	m, n = numpy.shape(array)
	for i in range(m):
		for j in range(n):
			if array[i, j] != 0:
				array[i, j] = 1
	return array


def load_train_data():
	data = []
	with open("train_short.csv") as file:
		lines = csv.reader(file)
		for line in lines:
			data.append(line)  # (1+42000)*(1+784)
	data.remove(data[0])
	data = numpy.array(data)  # 将列表转换numpy处理的数组
	label = data[:, 0]  # 此写法的后一个参数是针对数组中字符串的
	img = data[:, 1:]
	return normalizing(to_int(img)), label


def load_test_data():
	data = []
	with open("test_short.csv") as file:
		lines = csv.reader(file)
		for line in lines:
			data.append(line)  # (1+28000)*784
	
	data.remove(data[0])
	img = numpy.array(data)
	
	temp = numpy.mat(normalizing((to_int(img))))
	m, n = numpy.shape(img)
	
	new_array = numpy.zeros((m, n + 1))  # 深复制数据
	print("%d,%d" % (m, n))
	for i in range(m):
		for j in range(n):
			new_array[i][j] = temp[i, j]
		new_array[i][n] = i + 1
	return new_array


def load_test_label():
	data = []
	with open("knn_benchmark_short.csv") as file:
		lines = csv.reader(file)
		for line in lines:
			data.append(line)  # (1+28000)*784
	data.remove(data[0])
	num_label = numpy.array(data)
	return num_label[:, 1]


def classify(test, data_set, train_label, k):
	data_set_size = data_set.shape[0]
	sn = test[-1]
	diff_mat = numpy.tile(test[:-1], (data_set_size, 1)) - data_set
	# 在列方向将数据重复size次,行方向数据重复1次(不变)
	
	dist = (numpy.add.reduce((diff_mat) ** 2, axis=1)) ** 0.5
	index_list = dist.argsort()  # 返回排序后的索引数组,从小到大
	class_ct = {}
	for i in range(k):  # 最相邻的k张图片的序号
		label = train_label[index_list[i]]
		class_ct[label] = class_ct.get(label, 0) + 1  # 通过字典获取类别及其次数
	sorted_class_ct = sorted(class_ct.items(), key=lambda item: item[1], reverse=True)
	# 　字典的值按降序排列
	return sn, sorted_class_ct[0][0]  # 取字典的第一个的key,也就是个数最多的标签


def save_result(result):
	with open("result.csv", "w") as file:
		writer = csv.writer(file)
		writer.writerow([["ImageId", "Label"]])
		for i in result:
			writer.writerow(i.split(','))


class myThread(threading.Thread):
	def __init__(self, threadID, name, q):
		threading.Thread.__init__(self)
		self.threadID = threadID
		self.name = name
		self.q = q
	
	def run(self):
		print("开启线程：" + self.name)
		process_data(self.name, self.q)
		print("退出线程：" + self.name)


def process_data(threadName, q):
	while not exitFlag:
		queueLock.acquire()
		if not workQueue.empty():
			data = q.get()
			queueLock.release()
			sn, label = classify(data, train_data, train_label, 5)
			resultLock.acquire()
			results.append('%d,%s' % (sn, label))
			resultLock.release()
			print("%s finish imgId= %d" % (threadName, sn))
		else:
			queueLock.release()


if __name__ == '__main__':
	train_data, train_label = load_train_data()
	test_data = load_test_data()
	test_label = load_test_label()
	
	m, n = numpy.shape(test_data)  # m=28000,n=784
	threadList = []
	for i in range(8):
		threadList.append(("Thread-%d" % i))
	
	queueLock = threading.Lock()
	resultLock = threading.Lock()
	workQueue = queue.Queue(m)
	threads = []
	threadID = 1
	
	# 创建新线程
	for tName in threadList:
		thread = myThread(threadID, tName, workQueue)
		thread.start()
		threads.append(thread)
		threadID += 1
	
	# 填充队列
	queueLock.acquire()
	for test in test_data:
		workQueue.put(test)
	queueLock.release()
	
	# 等待队列清空
	while not workQueue.empty():
		pass
	
	# 通知线程是时候退出
	exitFlag = 1
	
	# 等待所有线程完成
	for t in threads:
		t.join()
	
	results = sorted(results, key=lambda item: int(item.split(',')[0]), reverse=False)
	save_result(results)
	i = 0
	err_ct = 0
	for item in results:
		label = item.split(',')[1]
		if label != test_label[i]:
			err_ct += 1
		i += 1
	print("失误率:%f" % (err_ct / i))
	
	print("退出主线程")
