import numpy
import csv


def to_int(array):
	array = numpy.mat(array)  # 将数组转化为矩阵
	m, n = numpy.shape(array)
	# 返回数组各维长度.例如二维数组中，表示将字符型转换为浮点型数组的“行数”和“列数”
	new_array = numpy.zeros((m, n))  # 深复制数据
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
	# print( numpy.shape(img))
	# label 42000*1 img 42000*784
	return normalizing(to_int(img)), label


def load_test_data():
	data = []
	with open("test_short.csv") as file:
		lines = csv.reader(file)
		for line in lines:
			data.append(line)  # (1+28000)*784
	data.remove(data[0])
	img = numpy.array(data)
	return normalizing((to_int(img)))


def load_test_label():
	data = []
	with open("knn_benchmark_short.csv") as file:
		lines = csv.reader(file)
		for line in lines:
			data.append(line)  # (1+28000)*784
	data.remove(data[0])
	num_label = numpy.array(data)
	return num_label[:, 1]


def classify(test_data, data_set, train_label, k):
	data_set_size = data_set.shape[0]
	
	diff_mat = numpy.tile(test_data, (data_set_size, 1)) - data_set
	# 在列方向将数据重复size次,行方向数据重复1次(不变)
	
	diff_mat_sqrt = numpy.array(diff_mat) ** 2  # 将矩阵中的每个元素平方
	dist_sqrt = diff_mat_sqrt.sum(axis=1)  # 将每一行求和
	dist = dist_sqrt ** 0.5  # 再开方
	index_list = dist.argsort()  # 返回排序后的索引数组,从小到大
	
	class_ct = {}
	for i in range(k):  # 最相邻的k张图片的序号
		label = train_label[index_list[i]]
		class_ct[label] = class_ct.get(label, 0) + 1  # 通过字典获取类别及其次数
	sorted_class_ct = sorted(class_ct.items(), key=lambda item: item[1], reverse=True)
	# 　字典的值按降序排列
	return sorted_class_ct[0][0]  # 取字典的第一个的key,也就是个数最多的标签


def save_result(result):
	with open("result.csv", "w") as file:
		writer = csv.writer(file)
		writer.writerow(["ImageId","Label"])
		for i in result:
			writer.writerow(i)


if __name__=='__main__':
	train_data, train_label = load_train_data()
	test_data = load_test_data()
	test_label = load_test_label()
	
	m, n = numpy.shape(test_data)  # m=28000,n=784
	error_count = 0
	result_list = []
	for i in range(m):
		label = classify(test_data[i], train_data, train_label, 5)
		result_list.append('%d,%s' % (i + 1, label))
		
		print("the classifier came back with: %s, the real answer is: %s\n" % (label, test_label[i]))
		if label != test_label[i]:  error_count += 1
	
	print("the total number of errors is: %d" % error_count)
	print("the total error rate is:%f" % (error_count / float(m)))
	save_result(result_list)

