# MINIST 测试
这是我用来学习ML相关算法的第一步

## KNN
### 减少数据量
测试时一定要减少数据量,不然很痛苦,此函数可以使任意csv文件缩短到你想要的行数
```angular2html
/data_short.py
```
### 单线程单进程KNN
这是我第一个原始版本
```angular2html
/knn_begin.py
```
### 多线程单进场KNN
由于CIL的原因，对于计算密集型。这种写法并无卵用
```angular2html
knn_multi_threading.py
```
### 多进程单线程KNN
手写KNN的最佳实践了,算是
```angular2html
/knn_multiprocessing.py
```
### 多进程分布式KNN
如果是同一个网络或者master能有一个固定IP的话,可以把程序跑在不同的机器上

通过预设的IP:port访问任务队列,暂时还没做到失败后任务重做。

缺点是每个worker都要把数据加载一次,数据量大时会有点慢，不过对于真正的计算密集型任务，这是值得的。
```angular2html
/knn_master.py
/knn_worker.py
```
