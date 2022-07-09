# TODO 机器学习 (ML)
''' 常用的极其学习库： scikit-learn, pytorch, tensorflow; ML中一般不关心模型的可靠和假设检验的合理性. '''
from sklearn import datasets   # sklearn是在python中键入scikit-learn名称的方式。二者等价
from sklearn import model_selection
from sklearn import linear_model
from sklearn import metrics
from sklearn import tree
from sklearn import neighbors
from sklearn import svm
from sklearn import ensemble
from sklearn import cluster
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
'''交叉验证：sklearn.model_selection ; 特征提取：feature_extraction, decomposition和feature_selection提供了降维方法(e.g. PCA or SVD) . '''
# sklearn的datasets提供了数据加载方法,
# Built in datasets
# datasets.load_boston                        # load_用于内置数据集合
# datasets.fetch_california_housing         # fetch_用于导入外部数据集合
# datasets.make_regression                  # make_随机数生成数据集合
'''Regression'''
# np.random.seed(123)
# X_all, y_all = datasets.make_regression(n_samples=50, n_features=50, n_informative=10, noise=0)    # 50个样本，50个特征但是其中仅仅有10个有用特征。返回都是 array. noise为标准差
# X_train, X_test, y_train, y_test = model_selection.train_test_split(X_all, y_all, train_size=0.5)    # 训练集合与测试集合分割函数 train_test_spilt()，训练集合比例 0.5
# print(X_train.shape, y_train.shape)   # shape属性用于返回函数.
# print(X_test.shape, y_test.shape)
# model = linear_model.LinearRegression()   # 建立一个回归类————这等价于使用statsmodel.api.OLS进行操作
# model.fit(X_train, y_train)   # 使用类里面的 fit 回归方法. 结果保留于model之中，例如系数可以使用 model.coef_
# def sse(resid):
#     return sum(resid ** 2)    # 返回向量平方和函数sse   ( index: sum of squared error )
# 使用 predict 方法进行测试集合的测试。
# resid_train = y_train - model.predict(X_train)
# sse_train = sse(resid_train)
# resid_test = y_test - model.predict(X_test)
# sse_test = sse(resid_train)
# print(sse_train, sse_test)    # 计算训练集合的残差与测试集合的残差 其值  9.701348303535586e-25 9.701348303535586e-25
# model.score(X_train, y_train) # 1.0
# model.score(X_test, y_test)   # 0.314074值
# def plot_residuals_and_coeff(resid_train, resid_test, coeff):
#     fig, axes = plt.subplots(1, 3, figsize=(12, 3))
#     axes[0].bar(np.arange(len(resid_train)), resid_train)    # bar参数横纵轴
#     axes[0].set_xlabel("sample number")
#     axes[0].set_ylabel("residual")
#     axes[0].set_title("training data")
#     axes[1].bar(np.arange(len(resid_test)), resid_test)
#     axes[1].set_xlabel("sample number")
#     axes[1].set_ylabel("residual")
#     axes[1].set_title("testing data")
#     axes[2].bar(np.arange(len(coeff)), coeff)
#     axes[2].set_xlabel("coefficient number")
#     axes[2].set_ylabel("coefficient")          # 上文中的 noise 越大.其
#     fig.tight_layout()
#     return fig, axes
# fig, ax = plot_residuals_and_coeff(resid_train, resid_test, model.coef_)
# plt.show(block=True)

# model = linear_model.Ridge(alpha=2.5)    # Linear least squares with l2 regularization. alpha为l2权重，用于降低模型复杂度
# model.fit(X_train, y_train)
# resid_train = y_train - model.predict(X_train)
# sse_train = sum(resid_train ** 2)
# resid_test = y_test - model.predict(X_test)
# sse_test = sum(resid_test ** 2)
# print(sse_train, sse_test)                                    # 过拟合导致后者误差很大。
# model.score(X_train, y_train), model.score(X_test, y_test)    # score用于测定R-squared变量.
# fig, ax = plot_residuals_and_coeff(resid_train, resid_test, model.coef_)
# plt.show(block=True)
# model = linear_model.Lasso(alpha=1.0)      # 用于消除无关特性，LASSO回归，alpha即 l1范数权重，同样的，范数施加于线性回归系数之上。
# model.fit(X_train, y_train)
# resid_train = y_train - model.predict(X_train)
# sse_train = sse(resid_train)
# resid_test = y_test - model.predict(X_test)
# sse_test = sse(resid_test)
# print(sse_train, sse_test)
# fig, ax = plot_residuals_and_coeff(resid_train, resid_test, model.coef_)
# plt.show(block=True)

# 以下是一种系统的选择权重alpha 的方法，即在一定的范围内部使用遍历方法。
# alphas = np.logspace(-4, 2, 100)   # 从0.0001到100之间的100个数值
# coeffs = np.zeros((len(alphas), X_train.shape[1]))   # 返回(25,50)的第二个值
# sse_train = np.zeros_like(alphas)  # 返回一个与给定数组具有相同形状和类型的零数组。
# sse_test = np.zeros_like(alphas)
# #对于一些参数alpha可能会出现警告：ConvergenceWarning: Objective did not converge.这是正常的，并不是所有的参数设定都能完美拟合，有些可能效果很差。
# for n, alpha in enumerate(alphas):
#     model = linear_model.Lasso(alpha=alpha)
#     model.fit(X_train, y_train)
#     coeffs[n, :] = model.coef_        # 保存所有的系数和残差结果.
#     resid = y_train - model.predict(X_train)
#     sse_train[n] = sum(resid ** 2)
#     resid = y_test - model.predict(X_test)
#     sse_test[n] = sum(resid ** 2)
# fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True)
# for n in range(coeffs.shape[1]):
#     axes[0].plot(np.log10(alphas), coeffs[:, n], color='k', lw=0.5)
# axes[1].semilogy(np.log10(alphas), sse_train, label="train")       # 再将log轴转化为普通常数的值.
# axes[1].semilogy(np.log10(alphas), sse_test, label="test")
# axes[1].legend(loc=0)
# axes[0].set_xlabel(r"${\log_{10}}\alpha$", fontsize=18)
# axes[0].set_ylabel(r"coefficients", fontsize=18)
# axes[1].set_xlabel(r"${\log_{10}}\alpha$", fontsize=18)
# axes[1].set_ylabel(r"sse", fontsize=18)
# fig.tight_layout()
# plt.show(block=True)
'''也可以使用LassoCV()和RidgeCV()函数进行自动进行权重的选择'''
# model = linear_model.LassoCV()
# model.fit(X_all, y_all)
# model.alpha_    # 自动选择的模型参数
# resid_train = y_train - model.predict(X_train)
# sse_train = sse(resid_train)
# resid_test = y_test - model.predict(X_test)
# sse_test = sse(resid_test)
# print(sse_train,sse_test)
# model.score(X_train, y_train), model.score(X_test, y_test)
# fig, ax = plot_residuals_and_coeff(resid_train, resid_test, model.coef_)
# plt.show(block=True)
# # 同时结合了 Lasso 和 Ridge 的elastic-net回归.
# model = linear_model.ElasticNetCV()
# model.fit(X_all, y_all)
# model.alpha_  # 对应于 L1 范数
# model.l1_ratio   # 对应于 l1 范数值权重   #即L1范数权重实际上是 alpha_ * l1_ratio；对应的L2权重是 alpha_ * (1- l1_ratio)
# resid_train = y_train - model.predict(X_train)
# sse_train = sum(resid_train ** 2)
# resid_test = y_test - model.predict(X_test)
# sse_test = sum(resid_test ** 2)
# print(sse_train,sse_test)
# model.score(X_train, y_train), model.score(X_test, y_test)
# fig, ax = plot_residuals_and_coeff(resid_train, resid_test, model.coef_)
# plt.show(block=True)

'''  Classification分类任务  '''
# iris = datasets.load_iris()  #4个输入(特征)，1个输出特征(结果输出值—— 3 选 1)
# type(iris)   # sklearn.utils.Bunch 类型
# print(iris.target_names,iris.feature_names)
# target_names是三个目标特征的名字：['setosa' 'versicolor' 'virginica']
# feature_names是特征名字 ['sepal length (cm)','sepal width (cm)','petal length (cm)','petal width (cm)']
# print(iris.data.shape,iris.target.shape)   # 导出数据和目标集合的大小
# # print(iris['DESCR'])                       # DESCR 用于打印数据集合的备注信息，也就是数据集合的信息.
# X_train, X_test, y_train, y_test = model_selection.train_test_split(iris.data, iris.target, train_size=0.7)
# classifier = linear_model.LogisticRegression()    # 第一步创建实例
# classifier.fit(X_train, y_train)                  # 数据拟合
# y_test_pred = classifier.predict(X_test)
# print(metrics.classification_report(y_test, y_test_pred))    # 构建一个显示主要分类指标的文本报告。  输入预测和真实值的数组，输出假阴性率和假阳性率.
# np.bincount(y_test)  # 计数每个值在非负整型数组中的出现次数。 即三类样本数目
# # sklearn 库中提供了用于评估分类器的性能和辅助函数
# metrics.confusion_matrix(y_test, y_test_pred)   # 混淆矩阵，ij对应的位置是i类被划分到j类的比例。对角线是正确划分的

# classifier = tree.DecisionTreeClassifier()  #这里仅仅使用了不同的分类器，其余操作完全一致
# classifier.fit(X_train, y_train)
# y_test_pred = classifier.predict(X_test)
# metrics.confusion_matrix(y_test, y_test_pred)

# classifier = neighbors.KNeighborsClassifier()
# classifier.fit(X_train, y_train)
# y_test_pred = classifier.predict(X_test)
# metrics.confusion_matrix(y_test, y_test_pred)

# classifier = svm.SVC()  #每一个分类器都来自于不同的模块，但都是scikit库内部的。
# classifier.fit(X_train, y_train)
# y_test_pred = classifier.predict(X_test)
# metrics.confusion_matrix(y_test, y_test_pred)
# classifier = ensemble.RandomForestClassifier()
# classifier.fit(X_train, y_train)
# y_test_pred = classifier.predict(X_test)
# metrics.confusion_matrix(y_test, y_test_pred)
# train_size_vec = np.linspace(0.1, 0.9, 30)       #  0.1 - 0.9 中间 30 个值.
# classifiers = [tree.DecisionTreeClassifier,  # 多个分类算法的合成.
#                neighbors.KNeighborsClassifier,
#                svm.SVC,
#                ensemble.RandomForestClassifier]
# cm_diags = np.zeros((3, len(train_size_vec), len(classifiers)), dtype=float)    # 一个3*30*4数组.
# for n, train_size in enumerate(train_size_vec):
#     X_train, X_test, y_train, y_test =\
#         model_selection.train_test_split(iris.data, iris.target, train_size=train_size)  # 一般直接等于就可以，但是一行代码太长的时候，就会使用enter键，
#                  # 自动填充为‘ =\ ’(line continuation character,行继续符)，注意注释不能放在其后，否则会报错.
#     for m, Classifier in enumerate(classifiers):
#         classifier = Classifier()
#         classifier.fit(X_train, y_train)
#         y_test_pred = classifier.predict(X_test)
#         cm_diags[:, n, m] = metrics.confusion_matrix(y_test, y_test_pred).diagonal()  #储存混淆矩阵对角值
#         cm_diags[:, n, m] /= np.bincount(y_test)   #  除以理想值的数目
# fig, axes = plt.subplots(1, len(classifiers), figsize=(12, 3))
# for m, Classifier in enumerate(classifiers):
#     axes[m].plot(train_size_vec, cm_diags[2, :, m], label=iris.target_names[2]) # 在4个图形(m)中，每一个绘制三条线.
#     axes[m].plot(train_size_vec, cm_diags[1, :, m], label=iris.target_names[1])
#     axes[m].plot(train_size_vec, cm_diags[0, :, m], label=iris.target_names[0])
#     axes[m].set_title(type(Classifier()).__name__)
#     axes[m].set_ylim(0, 1.1)
#     axes[m].set_xlim(0.1, 0.9)
#     axes[m].set_ylabel("classification accuracy")
#     axes[m].set_xlabel("training size ratio")
#     axes[m].legend(loc=4)
# fig.tight_layout()
# plt.show(block=True)
# 实际任务需要反复尝试不同分类器的性能，一般而言随机森林等决策树方法是不错起点.
''' Clustering 任务 '''
# 常用的K-mean算法, mean-shift算法(通过将数据拟合到某个密度函数)
iris=datasets.load_iris()        # 数据集合加载
X, y = iris.data, iris.target
np.random.seed(123)
n_clusters = 3   #3类
c = cluster.KMeans(n_clusters=n_clusters)   #建立分类为3类的类.
c.fit(X)
y_pred = c.predict(X) # 应当注意有的算法不支持 predict，此时应该使用 fit_predict方法
y_pred[::8]      # s[::k]意思是“每k个项目”，也就是从第一个元素开始，每8个选择一个元素.
y[::8]           # 同样的，每 8 个元素显示一个.
idx_0, idx_1, idx_2 = (np.where(y_pred == n) for n in range(3))
y_pred[idx_0], y_pred[idx_1], y_pred[idx_2] = 2, 0, 1
y_pred[::8]    #目的：使用统一数值表示聚类结果.
metrics.confusion_matrix(y, y_pred)
N = X.shape[1]   # N = 4. 4个特征
fig, axes = plt.subplots(N, N, figsize=(12, 12), sharex=True, sharey=True)
colors = ["coral", "blue", "green"]   # 珊瑚红色
markers = ["^", "v", "o"]
for m in range(N):     #遍历4*4的特征对(即16个子图)+用红的方框表示分类错误的样本。
    for n in range(N):
        for p in range(n_clusters):    # 4 * 4 * 3
            mask = y_pred == p
            axes[m, n].scatter(X[:, m][mask], X[:, n][mask],
                               marker=markers[p], s=30,
                               color=colors[p], alpha=0.25)

        for idx in np.where(y != y_pred):
            axes[m, n].scatter(X[idx, m], X[idx, n],
                               marker="s", s=30,
                               edgecolor="red",
                               facecolor=(1, 1, 1, 0))
    axes[N - 1, m].set_xlabel(iris.feature_names[m], fontsize=10)
    axes[m, 0].set_ylabel(iris.feature_names[m], fontsize=10)
plt.show(block=True)




