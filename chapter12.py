# TODO 数据处理与分析
# 常用的数据分析库有Pandas,对于更加复杂的数据可以使用：statsmodels, skicit-learn, pasty等……
# 在图形绘制上，使用Matplotlib为基础的Seaborn库——以默认绘制的图好看著称.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.style.use('ggplot')
import seaborn as sns
'''pandas中有两个主要的数据结构为 Dataframe和 Series.'''
# # 在series对象中可以使用标签而不是数字对其进行索引.
# s = pd.Series([909976, 8615246, 2872086, 2273305])   #建立一个该结构
# print(s,type(s))    # dtype是数据类型(int64)，type是结构的类型(pandas.core.series.Series)，astype是庚饭糕
# print(s.dtype,s.index,s.values)  # int64 RangeIndex(start=0, stop=4, step=1) [ 909976 8615246 2872086 2273305]
# s.index = ["Stockholm", "London", "Rome", "Paris"] #用城市名字索引代替索引中的数值
# s.name = "Population"  # 整个series对象的名称.
# print(s)
# s = pd.Series([909976, 8615246, 2872086, 2273305],
#               index=["Stockholm", "London", "Rome", "Paris"], name="Population")
# print(s["London"])
# print(s.Stockholm, s[["Paris", "Rome"]])  #这里必须形成两个中括号，否则会出错.
# print( s.median(), s.mean(), s.std(), s.min(), s.max(), s.var() )     # 常用的数据分析函数，针对series对象.e.g.这里的max(s)与s.max()可以混用.
# s.quantile(q=0.25), s.quantile(q=0.5), s.quantile(q=0.75)    # 分位数
# print(s.describe())    #该函数用于统一输出函数. 打印出所有的信息.
# fig, axes = plt.subplots(1, 4, figsize=(12, 3.5))     # 大小指定和分图
# s.plot(ax=axes[0], kind='line', title="line")   # 在kind中选择想要的图形类型.
# s.plot(ax=axes[1], kind='bar', title="bar")
# s.plot(ax=axes[2], kind='box', title="box")
# s.plot(ax=axes[3], kind='pie', title="pie")
# fig.tight_layout()
# plt.show(block=True)
'''Dataframe可以视作具有公共索引的Series对象的集合'''
# 最简单的方法是将Python的字典传递给DataFrame初始化，
# df = pd.DataFrame([[909976, 8615246, 2872086, 2273305],   # 字典格式
#                    ["Sweden", "United kingdom", "Italy", "France"]])
# # print(df)
# df = pd.DataFrame([[909976, "Sweden"],        # 另一种构建方法
#                    [8615246, "United kingdom"],
#                    [2872086, "Italy"],
#                    [2273305, "France"]])
# df.index = ["Stockholm", "London", "Rome", "Paris"]  # series同理，每一个元素设定一个名称.
# df.columns = ["Population", "State"]   # 给每一列设定标签
# 2个统一的等价表述如下：
# df = pd.DataFrame([[909976, "Sweden"],
#                    [8615246, "United kingdom"],
#                    [2872086, "Italy"],
#                    [2273305, "France"]],
#                   index=["Stockholm", "London", "Rome", "Paris"],
#                   columns=["Population", "State"])
# df = pd.DataFrame({"Population": [909976, 8615246, 2872086, 2273305],
#                    "State": ["Sweden", "United kingdom", "Italy", "France"]},
#                   index=["Stockholm", "London", "Rome", "Paris"])
# df
# print(df.index,df.columns,df.values,df.Population)  # 各种属性的调用
# print(df["Population"],type(df.Population),df.Population.Stockholm, type(df.index) )
# df.loc["Stockholm"]    # loc()用于抽取————抽出这一行的数据.构成一个series,如下
# type(df.loc["Stockholm"])   #pandas.core.series.Series
# df.loc[["Paris", "Rome"]]    #抽取多个行，会构成一个原DataFrame的子集，也是一个DataFrame
# df.loc[["Paris", "Rome"], "Population"]
# df.loc["Paris", "Population"]
# print(df.mean(),df.info(),df.dtypes,df.head())      # max这些操作会对数值列有作用. info()用于返回信息.
'''Pandas在处理大型数据时候很有用，此时文件不会从字典建立，而是从文件导入，e.g. csv文件等'''
# # 严格而言需要控制读取格式，但是直接默认也可以，因为delimiter默认是 comma“,”;header默认为第一行
# df_pop = pd.read_csv("european_cities.csv")   # 文件读取，第一个参数名称或者URL
# print(df_pop.head())   #head默认打印5行.
# df_pop = pd.read_csv("european_cities.csv", delimiter=",", encoding="utf-8", header=0)   #以格式读取
# print(df_pop.info())   #csv文件信息
# df_pop["NumericPopulation"] = df_pop.Population.apply(lambda x: int(x.replace(",", "")))   # 去除comma然后强制转化为int形式. 赋值给新的列名为NumericPopulation
# print(df_pop["State"].values[:3])  #只显式前n=3行
# df_pop["State"] = df_pop["State"].apply(lambda x: x.strip())   #strip 去除一整个字符转中的开始与末尾的空格
# print(df_pop.head(),df_pop.dtypes)
# df_pop2 = df_pop.set_index("City")   #重新设定索引.
# df_pop2 = df_pop2.sort_index()   # 将索引作为关键字对所有信息进行排列. 从 A-Z
# print(df_pop2.head())
# df_pop3 = df_pop.set_index(["State", "City"]).sort_index(level=0)   # level分层索引的第一层state进行索引.
# print(df_pop3.head(7))
# print(df_pop3.loc["Sweden"])    # 部分索引定位
# print(df_pop3.loc[("Sweden", "Gothenburg")])   #详细索引定位
# df_pop.set_index("City").sort_values(["State", "NumericPopulation"], ascending=[False, True]).head()   #按照某一列而不是索引进行排序.
# print(df_pop.State)  #效果是先按照State下降，随后在同一个State内部按照NumericPopulation上升排列
# city_counts = df_pop.State.value_counts()  #统计具有多少个不同State数据，每一类多少个
# city_counts.name = "# cities in top 105"
# df_pop3 = df_pop[["State", "City", "NumericPopulation"]].set_index(["State", "City"])
# df_pop4 = df_pop3.sum(level="State").sort_values("NumericPopulation", ascending=False)  # 在State的层面进行人口统计。
# df_pop5 = (df_pop.drop("Rank", axis=1)   #删除了rank列，(axis=0对应于删除行)
#                  .groupby("State").sum()     #groupby进行统计，根据给定的属性的不同值进行抽取, 然后求和
#                  .sort_values("NumericPopulation", ascending=False))   #随后使用人口进行排序
# df_pop5.head()
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
# city_counts.plot(kind='barh', ax=ax1)
# ax1.set_xlabel("# cities in top 105")
# df_pop5.NumericPopulation.plot(kind='barh', ax=ax2)
# ax2.set_xlabel("Total pop. in top 105 cities")
# fig.tight_layout()
# fig.savefig("ch12-state-city-counts-sum.pdf")
# plt.show(block=True)
'''Time Sequence'''
# 在pands中的时间序列带有规则或者不规则的时间戳。pandas的时间序列索引DatetimeIndex和PeriodIndex可以对日期，时间，周期以及日历进行有效操作, e.g.重采样和时间移动.
# 使用date_range来创建可以用于series和DataFrame的序列。
import datetime
# pd.date_range("2015-1-1", periods=31)   # 该函数返回一个 DatatimeIndex 实例, 该类继承于datetime对象. 优势在于继承了纳秒级别精度.而datetime只有毫秒级别.
# pd.date_range(datetime.datetime(2015, 1, 1), periods=31 )     # 起始日期加上时间戳个数，默认频率为day
# pd.date_range("2015-1-1 00:00", "2015-1-1 12:00", freq="H")   # 首尾时间加上时间戳频率 hour
# ts1 = pd.Series(np.arange(31), index=pd.date_range("2015-1-1", periods=31))
# print(ts1.head())
# ts1["2015-1-3"]           #  series对象，输入时间，返回对应于时间戳的信息 2
# print(ts1.index[2])       #  0表示自身, 增大1, 信息就后推1.
# print(ts1.index[2].year, ts1.index[2].month, ts1.index[2].day, ts1.index[2].nanosecond)  # 纳米秒
# ts1= ts1.index[2].to_pydatetime()   # 退化转换为标准的 datetime 对象.
# ts2 = pd.Series(np.random.rand(2),
#                 index=[datetime.datetime(2015, 1, 1), datetime.datetime(2015, 2, 1)])
# print(ts2)
# ''' 时间间隔定义的时间序列,用 PeriodIndex 类所表示, 可以使用Period列表作为产生'''
# periods = pd.PeriodIndex([pd.Period('2015-01'), pd.Period('2015-02'), pd.Period('2015-03')])   # 数据类型以Month显示.
# ts3 = pd.Series(np.random.rand(3), periods)
# print(ts3)
# print(ts3.index)
# # 使用 to_period 用于指定间隔的时间段.
# ts2.to_period('M')  #将DatetimeIndex转化为PeriodIndex对象
# ts4= pd.date_range("2015-1-1", periods=12, freq="M").to_period()    # 建立到转化.
# '''以下读取tsv文件, 是csv文件的一种变体，也可使用read_csv()读取.'''
# ts2.to_period('M')
# pd.date_range("2015-1-1", periods=12, freq="M").to_period()
# # Temperature time series example
# # !head -n 5 temperature_outdoor_2014.tsv
# df1 = pd.read_csv('temperature_outdoor_2014.tsv', delimiter="\t", names=["time", "outdoor"])   # 分隔符是Tab符号.(这里是默认的UNIX格式)
# df2 = pd.read_csv('temperature_indoor_2014.tsv', delimiter="\t", names=["time", "indoor"])     # 显式的指明时间列和数据列的名称
# print(df1.head())
# print(df2.head())
# df1.time = (pd.to_datetime(df1.time.values, unit="s")    # 转化为datetime格式,单位为 s
#               .tz_localize('UTC').tz_convert('Europe/Stockholm'))   # 时区定义为UTC下的欧洲
# df1 = df1.set_index("time")   # 设定时间列为索引
# df2.time = (pd.to_datetime(df2.time.values, unit="s")
#               .tz_localize('UTC').tz_convert('Europe/Stockholm'))
# df2 = df2.set_index("time")
# print(df1.head())
# print(df1.index[0])     # 输出第一行的信息
# fig, ax = plt.subplots(1, 1, figsize=(12, 4))   #注意绘制的第八个月的信息有缺失—————处理缺失数据也是Pandas的一项重要任务.
# df1.plot(ax=ax)
# df2.plot(ax=ax)
# fig.tight_layout()
# plt.show(block=True)

# # select january data
# df1.info()   #信息查看
# df1_jan = df1[(df1.index > "2014-1-1") & (df1.index < "2014-2-1")]   # 时间序列的筛选.使用dateframe的bool索引.
# print(df1.index < "2014-2-1")   #输出一个bool数组
# print(df1_jan.info())
# df2_jan = df2["2014-1-1":"2014-1-31"]
# fig, ax = plt.subplots(1, 1, figsize=(12, 4))
# df1_jan.plot(ax=ax)
# df2_jan.plot(ax=ax)
# fig.tight_layout()
# fig.savefig("ch12-timeseries-selected-month.pdf")
# plt.show(block=True)
'''使用apply方法建立一个新的 month 列 '''
# df1_month = df1.reset_index()  #数据清洗时，会将带空值的行删除，此时DataFrame或Series类型的数据不再是连续的索引，可以使用reset_index()重置索引。得到一个新的连续标量索引.
# df1_month["month"] = df1_month.time.apply(lambda x: x.month)
# print(df1_month.head())
# df1_month = df1_month.groupby("month").aggregate(np.mean)   # aggregate聚合一次性对所有对象获取均值.
# df2_month = df2.reset_index()
# df2_month["month"] = df2_month.time.apply(lambda x: x.month)
# df2_month = df2_month.groupby("month").aggregate(np.mean)
# df_month = df1_month.join(df2_month)   # 2个对象合成
# df_month = pd.concat([df.to_period("M").groupby(level=0).mean() for df in [df1, df2]], axis=1)  # concat多个对象合成.
# print(df_month.head(3))
# fig, axes = plt.subplots(1, 2, figsize=(12, 4))
# df_month.plot(kind='bar', ax=axes[0])
# df_month.plot(kind='box', ax=axes[1])
# fig.tight_layout()
# fig.savefig("ch12-grouped-by-month.pdf")
# print(df_month)
# plt.show(block=True)
# # resampling方法可以用于对数据进行升采样或者降采样. 返回一个resamper对象.使用聚合方法mean等可以对其操作.
# df1_hour = df1.resample("H").max()       # hour  重采样到max值
# df1_hour.columns = ["outdoor (hourly avg.)"]
# df1_day = df1.resample("D").mean()        # day   重采样到mean值
# df1_day.columns = ["outdoor (daily avg.)"]
# df1_week = df1.resample("7D").mean()      # 7 days
# df1_week.columns = ["outdoor (weekly avg.)"]
# df1_month = df1.resample("M").mean()      # month
# df1_month.columns = ["outdoor (monthly avg.)"]
# # df1.resample("D")
# df_diff = (df1.resample("D").mean().outdoor - df2.resample("D").mean().indoor)
# fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6))
# df1_hour.plot(ax=ax1, alpha=0.25)
# df1_day.plot(ax=ax1)
# df1_week.plot(ax=ax1)
# df1_month.plot(ax=ax1)
# df_diff.plot(ax=ax2)
# ax2.set_title("temperature difference between outdoor and indoor")
# fig.tight_layout()
# fig.savefig("ch12-timeseries-resampled.pdf")
# plt.show(block=True)
# ''' '''
# df1_dec25 = df1[(df1.index < "2014-9-1") & (df1.index >= "2014-8-1")].resample("D")    # 这里后面调用max()等函数才可以进行数据的输出.
# df1_dec25 = df1.loc["2014-12-25"]
# print(df1_dec25.head(5))
# df2_dec25 = df2.loc["2014-12-25"]
# df2_dec25.head(5)
# df1_dec25.describe().T   # describe()各种函数信息，随后转置.
# fig, ax = plt.subplots(1, 1, figsize=(12, 4))
# df1_dec25.plot(ax=ax)
# fig.savefig("ch12-timeseries-selected-month.pdf")
# df1.index   # 是一个时间索引
# plt.show(block=True)
'''seaborn绘图————可以绘制分布图，核密度图, 联合分布图，分类图，热度图，网格绘图等等.'''
sns.set(style="darkgrid")    # 生成灰色背景图,
# sns.set(style="whitegrid")     # 也可以设定为 whitegrid 函数.
df1 = pd.read_csv('temperature_outdoor_2014.tsv', delimiter="\t", names=["time", "outdoor"])
df1.time = pd.to_datetime(df1.time.values, unit="s").tz_localize('UTC').tz_convert('Europe/Stockholm')
df1 = df1.set_index("time").resample("10min").mean()
df2 = pd.read_csv('temperature_indoor_2014.tsv', delimiter="\t", names=["time", "indoor"])
df2.time = pd.to_datetime(df2.time.values, unit="s").tz_localize('UTC').tz_convert('Europe/Stockholm')
df2 = df2.set_index("time").resample("10min").mean()
df_temp = pd.concat([df1, df2], axis=1)
# fig, ax = plt.subplots(1, 1, figsize=(8, 4))
# df_temp.resample("D").mean().plot(y=["outdoor", "indoor"], ax=ax)
# fig.tight_layout()
# plt.show(block=True)
'''除此之外，还可以使用kdeplot和distplot绘制核密度估计与直方图'''
# sns.kdeplot(df_temp["outdoor"].dropna().values, shade=True, cumulative=True);
# sns.distplot(df_temp.to_period("M")["outdoor"]["2014-04"].dropna().values, bins=50);   # distplot: Flexibly plot a univariate distribution of observations.
# sns.distplot(df_temp.to_period("M")["indoor"]["2014-04"].dropna().values, bins=50);    # dropna: Return a new Series with missing values removed.
# plt.show(block=True)

# with sns.axes_style("white"):
#     sns.jointplot(df_temp.resample("H").mean()["outdoor"].values,   # 等高线图
#                   df_temp.resample("H").mean()["indoor"].values, kind="hex");
# plt.show(block=True)
#
# sns.kdeplot(df_temp.resample("H").mean()["outdoor"].dropna().values,  # 联合分布图
#             df_temp.resample("H").mean()["indoor"].dropna().values, shade=False);
# plt.show(block=True)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
sns.boxplot(data =df_temp.dropna(), ax=ax1, palette="pastel")   # 注意！！！新版本必须显式指定data=绘制的数据，否则会报错。
sns.violinplot(data=df_temp.dropna(), ax=ax2, palette="pastel")
fig.tight_layout()
plt.show(block=True)

# sns.violinplot(x=df_temp.dropna().index.month, y=df_temp.dropna().outdoor, color="skyblue");
# plt.show(block=True)

# df_temp["month"] = df_temp.index.month
# df_temp["hour"] = df_temp.index.hour
# print(df_temp.head() )
# # 使用平均值计算每个 month-hour的透视平均值图形
# table = pd.pivot_table(df_temp, values='outdoor', index=['month'], columns=['hour'], aggfunc=np.mean)    # 建立透视表.
# print(table)
#
# fig, ax = plt.subplots(1, 1, figsize=(8, 4))
# sns.heatmap(table, ax=ax);    #热力图
# fig.tight_layout()
# plt.show(block=True)

