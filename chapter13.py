# TODO 基本统计方法
# 过去是使用S语言或者R语言，现在python也已经集成了一些库，statsmodels 和 scikit-learn提供了一些高级的统计方法。本章主要关心于基于Numpy的Scipy中的stats模块.
from scipy import stats
from scipy import optimize
import numpy as np
import random
#  matplotlib inline 只在jupyter notebook中需要这一行，用于输出图片到网页.
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")   # seaborn格式设定
x = np.array([3.5, 1.1, 3.2, 2.8, 6.7, 4.4, 0.9, 2.2])   #建立numpy数组
np.mean(x)
np.median(x)
print(x.min(), x.max(), x.var(),x.std(),x.var(ddof=1),x.std(ddof=1))    # ddof为方差之中的分母，设定为 1 时为 n-1 无偏估计.
print(type(x))   # type()是函数，不是属性，这一行输出： <class 'numpy.ndarray'>
''' Random numbers '''
# pyhton内置的random模块，numpy有random模块，但后者能生成支持的随机向量。更好的是scipy.stats.
# random.seed(123456789)      # 随机数种子，一般随便设定就好
# random.random()
# random.randint(0, 10)  #返回一个[0,10)之间整数
# np.random.seed(123456789)
# np.random.rand()       # [0,1)均匀分布.
# np.random.randn()      # Return a sample (or samples) from the "standard normal" distribution.
# np.random.rand(5)      # 1*5的数组
# np.random.randn(2, 4)  # 2*4数组
# np.random.randint(10, size=50)    # 默认(包含的)下限为0值.
# np.random.randint(low=10, high=20, size=(2, 10))    # [10,20)中2*10大小。
# fig, axes = plt.subplots(1, 3, figsize=(12, 3))
# axes[0].hist(np.random.rand(10000),color="red",bins=50)  # 直方图,红色,50条柱
# axes[0].set_title("rand")
# axes[1].hist(np.random.randn(10000))                # 默认为蓝色, 10条柱
# axes[1].set_title("randn")
# axes[2].hist(np.random.randint(low=1, high=10, size=10000), bins=9, align='left')
# axes[2].set_title("randint(low=1, high=10)")
# fig.tight_layout()
# plt.show(block=True)
''' choice中的不放回抽样, 意指每一元素只会抽到一次. '''
# np.random.choice(10, 5, replace=False)   # [0,10),1*5数组.不放回 注意5<10否则报错
# np.random.seed(123456789)                # 基于设定的随机数种子，使得生成的样本一样.Seed是全局变量.
# np.random.seed(123456789); np.random.rand()
# np.random.seed(123456789); np.random.rand()
# # RandomState可以用于维护多个随机种子. 多用于多线程系统.
# prng = np.random.RandomState(123456789)
# prng.rand(2, 4)    # 这些都是基于其的各类随机sample,返回array
# prng.chisquare(1, size=(2, 2))
# prng.standard_t(1, size=(2, 3))
# prng.f(5, 2, size=(2, 4))   # F分布
# prng.binomial(10, 0.5, size=10)
# prng.poisson(5, size=10)
'''随机变量及其分布'''
# scipy的stats提供了很多类，连续和离散随机变量的类为： rv_continous / rv_discrete  (random_variable)
# 常用函数：pdf/pmf; cdf=1-sf; ppf(cdf的反函数), moment(各阶矩)，stats(数值特征，均值方差等), fit(连续下的极大似然拟合), expect(分布期望);
# interval(求分布在给定置信区间(百分比)下的置信区间); rvs(随机变量的采样); mean, median, std, var (各类描述)
# np.random.seed(123456789)
# X = stats.norm(1, 0.5)    #normal随机变量
# print(X.mean(), X.median(),X.std(),X.var(),[X.moment(n) for n in range(5)])   # 不能使用 mean(x)
# X.stats() # array统计信息
# X.pdf([0, 1, 2])   # 返回对应的概率值
# X.cdf([0, 1, 2])
# X.rvs(10)          # 采样
# 另外一种调用方法. 以数字特征
# stats.norm(1, 0.5).stats()
# X=stats.norm.stats(loc=2, scale=0.5)   # 参数loc位置(mean), scale放缩(variance)
# X.interval(0.95)     # 求解 0.95 置信区间，均值为 1 对称分布.
# def plot_rv_distribution(X, axes=None):
#     """Plot the PDF, CDF, SF and PPF of a given random variable"""
#     if axes is None:
#         fig, axes = plt.subplots(1, 3, figsize=(12, 3))
#     x_min_999, x_max_999 = X.interval(0.999)
#     x999 = np.linspace(x_min_999, x_max_999, 1000)
#     x_min_95, x_max_95 = X.interval(0.95)
#     x95 = np.linspace(x_min_95, x_max_95, 1000)
#     if hasattr(X.dist, 'pdf'):    # hasattr 用于检查函数是否具有某个属性值.(即仅仅可以用于 rv_cotinous)
#         axes[0].plot(x999, X.pdf(x999), label="PDF")   # 在0.95的区间之中进行pdf数值计算.
#         axes[0].fill_between(x95, X.pdf(x95), alpha=0.25)   # 覆盖图，仅仅覆盖0.95的中间部分.
#     else:
#         x999_int = np.unique(x999.astype(int))
#         axes[0].bar(x999_int, X.pmf(x999_int), label="PMF")      #  rv_discrete 则使用 bar 图操作.
#     axes[1].plot(x999, X.cdf(x999), label="CDF")
#     axes[1].plot(x999, X.sf(x999), label="SF")
#     axes[2].plot(x999, X.ppf(x999), label="PPF")
#     for ax in axes:
#         ax.legend()     #为所有行为指定标签
#     return axes         # 返回坐标轴
# fig, axes = plt.subplots(3, 3, figsize=(12, 9))
# X = stats.norm()
# plot_rv_distribution(X, axes=axes[0, :])   #在第一行进行norm的绘制
# axes[0, 0].set_ylabel("Normal dist.")
# X = stats.f(2, 50)
# plot_rv_distribution(X, axes=axes[1, :])
# axes[1, 0].set_ylabel("F dist.")
# X = stats.poisson(5)
# plot_rv_distribution(X, axes=axes[2, :])
# axes[2, 0].set_ylabel("Poisson dist.")
# fig.tight_layout()
# plt.show(block=True)
# 以下为从大量样本中恢复概率分布.
# def plot_dist_samples(X, X_samples, title=None, ax=None):
#     """ Plot the PDF and histogram of samples of a continuous random variable"""
#     if ax is None:
#         fig, ax = plt.subplots(1, 1, figsize=(8, 4))
#     x_lim = X.interval(.99)  # 返回一个元组
#     x = np.linspace(*x_lim, num=100)    # 元组的拆包操作.
#     ax.plot(x, X.pdf(x), label="PDF", lw=3)
#     ax.hist(X_samples, label="samples", bins=75, density=True)   # normed进行归一化处理，使得一整个图形形成一个概率分布——目前版本已改为使用density参数
#     ax.set_xlim(*x_lim)
#     ax.legend()
#     if title:
#         ax.set_title(title)
#     return ax
# fig, axes = plt.subplots(1, 3, figsize=(12, 3))
# X = stats.t(7.0)
# plot_dist_samples(X, X.rvs(2000), "Student's t dist.", ax=axes[0])   #基于样本绘制如上函数.
# X = stats.chi2(5.0)
# plot_dist_samples(X, X.rvs(2000), r"$\chi^2$ dist.", ax=axes[1])
# X = stats.expon(0.5)
# plot_dist_samples(X, X.rvs(2000), "exponential dist.", ax=axes[2])
# fig.tight_layout()
# plt.show(block=True)
#
# X = stats.chi2(df=5)
# X_samples = X.rvs(500)
# df, loc, scale = stats.chi2.fit(X_samples)
# print(df, loc, scale)
# Y = stats.chi2(df=df, loc=loc, scale=scale)
# fig, ax = plt.subplots(1, 1, figsize=(8, 3))
# x_lim = X.interval(.99)
# x = np.linspace(*x_lim, num=100)
# ax.plot(x, X.pdf(x), label="original")   #绘制理想状态，初始的的pdf
# ax.plot(x, Y.pdf(x), label="recreated")   #基于数据信息重建的分布
# ax.legend()
# fig.tight_layout()
# plt.show(block=True)
#
# fig, axes = plt.subplots(1, 2, figsize=(12, 4))
# x_lim = X.interval(.99)
# x = np.linspace(*x_lim, num=100)
# axes[0].plot(x, X.pdf(x), label="original")
# axes[0].plot(x, Y.pdf(x), label="recreated")
# axes[0].legend()
# axes[1].plot(x, X.pdf(x) - Y.pdf(x), label="error")   # 绘制差值图形.
# axes[1].legend()
# fig.tight_layout()
# plt.show(block=True)
'''Hypothesis testing：仅仅以概率成立，而不能断定一定可靠：使用scipy.stats模块中的函数'''
# 原假设H0,备选假设HA, 二者仅仅成立一个. 选择阈值alpha, 如果数据的提供的信息低于阈值那就判定为HA。一般而言最难的是获取统计值的样本分布；
# 对应于(特别的)给定的任务，有特定的分布与其对应。
# 单个总体均值————正态或者t分布(ttest_lsamp); 两个随机变量的均值————t分布(ttest_ind/ttest_rel);检验连续分布对数据集合的拟合程度(Komogorov-Smornov分布)————kstest;
# 检验分类是否以给定的频率出现(卡方)————chisquare; 列连表的相关性(卡方)————chi2_contingency; 检验两个或者以上的方差是否相等(F分布)————barlett/levene;
# 检验两个变量的非相关性(Beta分布)————pearsonr/spearmanr; 两个或者两个以上的变量是否具有相同的总体均值即ANOVA分析(F分布)————f_oneway/kruskal
# np.random.seed(123456789)
# mu, sigma = 1.0, 0.5
# X = stats.norm(mu-0.2, sigma)       # 均值 0.8, 方差 0.5
# n = 100
# X_samples = X.rvs(n)
# z = (X_samples.mean() - mu)/(sigma/np.sqrt(n))                     # 求取标准差(方差已知对应于正态分布)
# t = (X_samples.mean() - mu)/(X_samples.std(ddof=1)/np.sqrt(n))     # 求取样本的标准差(方差未知情况对应于t分布)
# print(z,t)                       # 值 z=-2.8338979550098298;  t=-2.9680338545657845
# stats.norm().ppf(0.025)          # 求取阈值，标准normal分布的值，cdf反函数
# 2 * stats.norm().cdf(-abs(z))    # 可以使用cdf方法反向计算对应的阈值,双边检验*2, 即如果假设对应的分布成立，那么z位于那个概率水平
# 2 * stats.t(df=(n-1)).cdf(-abs(t))          # df为t分布的参数，表明其自由度.
# t, p = stats.ttest_1samp(X_samples, mu)     # 直接使用该函数，效果和上面是一样的.
# print(t,p)
# fig, ax = plt.subplots(figsize=(8, 3))
# sns.distplot(X_samples, ax=ax)              # 绘制hist与核分布函数.
# x = np.linspace(*X.interval(0.999), num=100)
# ax.plot(x, stats.norm(loc=mu, scale=sigma).pdf(x))
# fig.tight_layout()
# plt.show(block=True)
# n = 50   # 双变量问题.
# mu1, mu2 = np.random.rand(2)
# X1 = stats.norm(mu1, sigma)
# X1_sample = X1.rvs(n)
# X2 = stats.norm(mu2, sigma)
# X2_sample = X2.rvs(n)
# t, p = stats.ttest_ind(X1_sample, X2_sample)
# print(t,p,mu1, mu2)
# sns.distplot(X1_sample)
# sns.distplot(X2_sample)
# plt.show(block=True)
'''对未知分布基于数据进行分布的构建，即核密度估计(KDE), 可以视作直方图的平滑版本, bw表示带宽, K为核函数 '''
# 带宽过大无法实现选择，带宽过小会产生噪声估计，同时核函数一般选择为高斯：stats.kde.gaussian_kde 函数，返回一个可调节的对象，
np.random.seed(0)
X = stats.chi2(df=5)
X_samples = X.rvs(100)
kde = stats.kde.gaussian_kde(X_samples)
kde_low_bw = stats.kde.gaussian_kde(X_samples, bw_method=0.25)
x = np.linspace(0, 20, 100)
fig, axes = plt.subplots(1, 3, figsize=(12, 3))
axes[0].hist(X_samples, density=True, alpha=0.5, bins=25)
axes[1].plot(x, kde(x), label="KDE")
axes[1].plot(x, kde_low_bw(x), label="KDE (low bw)")
axes[1].plot(x, X.pdf(x), label="True PDF")
axes[1].legend()
sns.distplot(X_samples, bins=25, ax=axes[2])
fig.tight_layout()
plt.show(block=True)

kde.resample(10)     # 从拟合得到的分布函数中得到10个样本
def _kde_cdf(x):
    return kde.integrate_box_1d(-np.inf, x)     #计算两个形参之中的pdf积分
kde_cdf = np.vectorize(_kde_cdf)
fig, ax = plt.subplots(1, 1, figsize=(8, 3))
sns.distplot(X_samples, bins=25, ax=ax)
x = np.linspace(0, 20, 100)
ax.plot(x, kde_cdf(x))
fig.tight_layout()
def _kde_ppf(q):             # 求取cdf的反函数
    return optimize.fsolve(lambda x, q: kde_cdf(x) - q, kde.dataset.mean(), args=(q,))[0]    # fsolve函数用于求解方程的根.
kde_ppf = np.vectorize(_kde_ppf)  # 将函数向量化，使得可以从函数的标量输入扩展为向量输入
kde_ppf([0.05, 0.95])
X.ppf([0.05, 0.95])
plt.show(block=True)



