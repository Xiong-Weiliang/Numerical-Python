# TODO Chapter2 向量矩阵与多维数组
# ''' Numpy 是同质的，带有数据类型的，固定长度的数组。即数据同类型并且无法调整数组的大小————网站www.numpy.org
# 主要数据结构为ndarry类。属性有shape,size,ndim,nbytes,dtype.'''
# import numpy as np
# data=np.array([[1,2],[3,4],[5,6]],dtype='float64') # array()函数————新建数组，注意有两层中括号，并且修改了数据类型。
# # data=np.array([[11],[54],[96]],dtype=np.complex) 注意这种修改类型已经被淘汰了。
# print(type(data))
# print(data.dtype)   #默认输出 int32，但是上面改为了float64
# # 复制式的类型转换如下
# data=np.array(data,dtype='complex')
# print(data.dtype)
# # 使用astype实现复制功能
# data.astype('int32')     # 去除虚部
# print(data.dtype)
# # 注意使用加法的时候，数据类型会自动转化的。
# data1=np.array([[1,3+3j],[2,0.2],[3-5j,5+1j]])   #1j表明虚数
# data1=data1+data
# print(data1)
# # 可能需要指定数组的类型，否则无法使用例如sqrt()等函数
# # 任何array都具有real与imag部分，例如上面的int，注意python本身也具有这样的功能。即使不依赖于numpy
# print(data.dtype)
# print(data.imag)   #虚部输出全0
# # C语言一般按照行储存，Format按照列储存，因此使用order='C'或者order='F'设置。这引发了一个新的变量strides，用于表示内存的偏移量。
# # 在转置的时候，可以仅仅转换strides,不同的地址表示但是本质是同一个数组，即构成了'视图'————可能对视图的修改也会改变真实值。
'''numpy常用的建立数组的函数如下
zeros,ones,diag
arange 均匀间隔数，指定开始结束和间隔。  #默认无结束值  可以使用endpoint来改变相关设定。
linspace 均匀间隔数，指定开始结束和数据个数。  #默认有结束值
logspace 等比数列，指定开始和结束值
meshgrid 从一维数组生成坐标矩阵 即生成多维度的网络坐标。 类似的函数有 np.mgrid 和 np.ogrid
fromfunction 从原始的数据的函数计算构建
fromfile 与tofile函数，用于从二进制文件中读取和储存数据，类似有用于文本文件的genfromtxt,loadtxt
ramdom.rand [0,1]之间均匀分布
'''
# import numpy as np
# data2=np.array([1,2,3,5])
# print(data2.ndim)
# print(data2.shape)
# data3=np.ones((2,3))    #这里传入的参数必须是一个元组，所以是两层号
# print(data3)
# data4=np.full((2,5),10.5641)   #填充函数full()
# print(data4)
# data5=np.empty((2,7))   #和zeros()的区别在于。empty()更快但随机初始化，所以ones()更加安全
# print(data5)
# data5.fill(0.5646)
# x=np.array([-1,0,1])
# y=np.array([-2,0,2])
# X,Y=np.meshgrid(x,y)
# print(X) #(3,3)的数组
# '''此外有一些类似函数如下：
# ones_like，np.zeros_like,np.full_like,empty_like函数.一个例子如下
# '''
# z=np.ones_like(X)  # z的尺寸和维度与X相同，同时以1填充，其余类似。
# print(z)
import numpy as np
# x=np.identity(3)   #三维单位阵
# print(x)
# y=np.eye(3,k=1)
# print(y)
# z=np.diag(np.arange(0,20,5),k=2)
# print(z)
# 索引查找
# print(x[0:2])   #相当于取0-1行，注意下标从0开始。不含末尾元素。
# f=lambda n,m: m+10*n
# A=np.fromfunction(f,(6,6))      #  意思就是将(6,6)传递给函数f，然后使用np进行排列组合式的计算。
# print(A[:3,:3])
# print(A[::2,::2])  #从首到尾，间隔取值
# B=A[1:5,1:5]     #注意这样直接等于的是得到的strides而不是系的数据，对B的改变会造成A的同样变化。
# B[:,:]=0
# print(A)
# '''显式复制数据使用ndarray类的自带的copy()函数，或者直接显式调用np.copy'''
# # C=A[2:6,2:6].copy()
# # print(C)
# C=np.copy(A[0:3,0:3])
# print(C)
# 花式索引和bool索引，注意这样的索引得到的是迟来的独立数据而非strides__注意任一索引都可以独立的使用于矩阵的任一维度。
# import numpy as np
# A=np.linspace(0,1,11)
# print(A[np.array([0,2,4])])    # 花式索引,即序列或者列表的嵌套使用
# print(A>0.5)   # 打印出一堆false 和 true
# # 一些函数操作，用于数组的变换
'''有如下的函数和方法，注意是否返回值。
np.reshape(), flatten()创建副本，折叠为一维数组。
reval() 创建N维数组的视图（如果不能，那么创建副本），将其折叠为一维的数组
squeeze() 删除长度为1的维度
newaxis() 增加长度为1的新维度，其中新的索引np.newaxis
transpose和T，进行转置
h/v/dstack  水平叠加，垂直叠加，深度叠加（分别按照维度1 0 2）
concatenare 按照指定维度堆叠数组
resize  根据给定的大小创作原始数据的新副本，如果有需要，使用原始数据填充新副本。
append 增加新元素
insert 插入元素
delete 删除指定元素，创建新的副本
 例子如下：'''
# data=np.array([[1,2],[3,4],[6,9]])
# data1=np.reshape(data,(1,6))  #作为函数调用
# # print(data1)
# # data2=data.reshape(4)  # 作为方法         #注意这里仅仅返回新的视图，而不会返回新的值（使用copy()函数）
# data3=data.flatten()
# print(data3.shape)
'''创建新的轴方法如下，注意也可使用expand_dims(data,axis)来增加新的维度'''
# data=np.arange(0,8)
# col=data[:,np.newaxis]   # 注意这个调用方式，等价于expand_dims(data,axis=1)，相当于数据data按照轴为1的方向所展开
# print(col)
# r=data[np.newaxis,:]  # 等价于expand_dims(data,axis=0)
# print(r.T)    #输出转置
'''Note: 一旦一个array数组创建完成之后，就无法改变数组之中元素的个数，使用insert,append,delete时必须创建一个新的数组，然后复制进去'''
# 向量化表达式(vecorized expression),Numpy的广播规则，相等或者有一个值为 1 可用于广播（broadcast），使得最终匹配
# 尝试对不兼容的数组进行计算时，会引发ValueError
# x=np.array([[1,2],[3,4]])
# y=np.array([[1,2],[3,4]])
# # print(x*y)     #注意这里是元素操作，不是矩阵乘法，其余运算类似。
# x+=y   #原位运算可以用于降低内存加提高性能
# '''一些常用的函数与常数，例如 sin, sinh, sqrt, log2,log10等等，'''
# print(np.pi)
# '''元素操作方法有add,substract,multiple,dividi,power,reciprocal（倒数），floor, ceil, rint(转化为整数)'''
# print(np.multiply(x,y))  #其实就是等价于*元素作用。
# print(x)
'''一个很重要的方法是将标量函数适用于向量化操作，例如如下的阶跃函数'''
# def heaviside(x):
#     if x<=0:return 0
#     else:return 1
# print(heaviside(5))    #输出1
# # 转化为向量函数可以使用 vectorize()函数
# heaviside=np.vectorize(heaviside)
# x=np.array([12,3,4,-5])
# print(heaviside(x))
'''聚合函数：输入数组，输出一个标量函数mean,std,var,sum,prod（所有元素的乘法），cunsum（累计和），cumprod(累计乘法)，
min,max,argmin,argmax（返回最大值索引）,all,any (是否全部或者存在)如果希望在某个轴上面进行聚合，那么可使用axis参数。'''
# x=np.random.normal(size=(5,10,15))
# x1=x.sum(axis=0).shape
# print(x1)
# x2=np.sum(x)   #注意加np.前缀，使得其成为聚合函数，  1.sum 不能处理二维及二维以上数组，对于1维列表，sum(a)和numpy.sum(a)效果相同，对于二维列表，sum(a)会报错，用法非法。
# print(x2)
# print(x.sum()) #效果相同
# 布尔数组和条件表达式
# a=np.array([1,2,3,4,5])
# b=np.array([1,2,5,4,7])
# print(all(a<b+10))   #每个b的元素相加10
'''逻辑运算函数和条件函数，where,choose,select,nonzero(花式索引),logical_and,logical_or,logocal_xor,logical_not（逐元素进行操作）'''
# x=np.linspace(-4,4,9)
# print(x)
# x1=np.where(x<0,x**2,x**3)   #如果为true选择第二个元素操作，否则第三个操作
# print(x1)
# x2=np.select([x<-1,x>=2],[x**3,x**4])     #没有被选中的使用0值。
# print(x2)
'''集合运算符： 集合：对无序对象进行的统一管理方法'''
# a=np.unique([1,2,3,4]) #建立唯一元素数组，即每个元素只出现一次,即把重复的删除掉
# b=np.unique([2,3,4,4,5,6,5])
# print(a)
# print(np.in1d(a,b))
# s= 1 in a  # 检查1位于a之中吗？
# print(s)
# 其余的函数 intersectld, setdiff1d, union1d 查核交集，只出现于第一个集合中的元素，并集。
# fliplr,dlipud 翻转每行每列之间的元素， rot90 沿着前面两个轴将元素旋转90度。sort沿着指定轴对元素进行排序。
'''矩阵和张量操作
dot   点积，就是矩阵乘法
inner  内积，矩阵映射为标量
cross   叉积
tensordot  指定轴的点积
outer   外积，标量映射为矩阵。
kron    Kronecker积
einsum  爱因斯坦求和约定操作   
'''
# 使用np.matrix或者np.asmatrix 函数声明或者使用matrix类，类似的使用asarray也可以将其转换为array格式。
x=np.array([1,2,3,4])
y=x
z=np.einsum('k, k',x,y)  #  向量的爱因斯坦求和
print(z) # 30=1+4+9+16
A=np.arange(9).reshape(3,3)
B=A.T
C=np.einsum('mk,kn',A,B)   #对下标k进行遍历，m和n是C中的位置
print(C)

