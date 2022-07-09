# TODO 绘图与可视化 Matplotlib
'''一些其他的库：Brokeh,Plotly,Seaborn,Mayavi,ParaVIew等都可以用于作图'''
# matlab.pyplot提供了2个API，面向对象的和面向状态的，强烈建议只使用对象API
# % matploylib inline 这个魔术命令使得直接在网页前端显示，而不会新弹出一个窗口
import matplotlib as mpl       # 一个最简单的示例如下所示
# mpl.use('qt4agg')    # 调用后端，必须紧接在导入matlpotlib之后。（这个好像不支持）
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d.axis3d import *   #版本问题，这行代码似乎有误
import numpy as np
import sympy
# x = np.linspace(-5, 2, 100) #在-5到2中选取100个点
# y1 = x**3 + 5*x**2 + 10  #计算函数值
# y2 = 3*x**2 + 10*x
# y3 = 6*x + 10
# fig, ax = plt.subplots()      #plt.subplots()函数生成画布和轴
# ax.plot(x, y1, color="blue", label="y(x)")   #设定线颜色和标记
# ax.plot(x, y2, color="red", label="y'(x)")
# ax.plot(x, y3, color="green", label="y''(x)")   #注意后续的使用作图全部是通过ax的函数plot()实现的。
# ax.set_xlabel("x")  #设定坐标轴
# ax.set_ylabel("y")
# ax.legend()         # 显示所有的线的标注
# plt.show(block=True)   #  注意：注意版本问题，最新版本的要加这一行，加在绘制完了后所有图像的后面
# fig.savefig("ch4-figure-1.pdf")   # 保持为pdf格式的文件

# mpl.rcParams["font.family"] = "serif"
# mpl.rcParams["font.size"] = "12"
# fig, ax = plt.subplots()
# ax.plot(x, y1, lw=1.5, color="blue", label=r"$y(x)$")   # r:横着写标注
# ax.plot(x, y2, lw=1.5, color="red", label=r"$y'(x)$")
# ax.plot(x, y3, lw=1.5, color="green", label=r"$y''(x)$")
# ax.plot(x, np.zeros_like(x), lw=0.5, color="black")
# ax.plot([-3.33, -3.33], [0, (-3.3)**3 + 5*(-3.3)**2 + 10], ls='--', lw=0.5, color="black")
# ax.plot([0, 0], [0, 10], lw=0.5, ls='--', color="black")
# ax.plot([0], [10], lw=0.5, marker='o', color="blue")
# ax.plot([-3.33], [(-3.3)**3 + 5*(-3.3)**2 + 10], lw=0.5, marker='o', color="blue")
# ax.set_ylim(-15, 40)    #轴的范围
# ax.set_yticks([-10, 0, 10, 20, 30])  #具体的轴标记
# ax.set_xticks([-4, -2, 0, 2])
# ax.set_xlabel("$x$", fontsize=18)    #横纵轴和字号
# ax.set_ylabel("$y$", fontsize=18)
# ax.legend(loc=0, ncol=3, fontsize=14, frameon=False)  #不使用边框
# fig.tight_layout()   #自动调整子图参数，使之填充整个图像区域
# plt.show(block=True)
# TODO 一个困难是交互式后端(backend)的使用（不过好像没有什么实际意义）  P100
# fig = plt.figure(figsize=(8, 2.5), facecolor="#f1f1f1")
# # axes coordinates as fractions of the canvas width and height
# left, bottom, width, height = 0.1, 0.1, 0.8, 0.8    #比例设定
# ax = fig.add_axes((left, bottom, width, height), facecolor="#e1e1e1")
# x = np.linspace(-2, 2, 1000)
# y1 = np.cos(40 * x)
# y2 = np.exp(-x**2)
# ax.plot(x, y1 * y2)
# ax.plot(x, y2, 'g')
# ax.plot(x, -y2, 'g')
# ax.set_xlabel("x")
# ax.set_ylabel("y")
# plt.show(block=True)    #不加这个东西好像也可以
# fig.savefig("graph.png", dpi=100, facecolor="#f1f1f1")
# fig.savefig("graph.pdf", dpi=300, facecolor="#f1f1f1")
# fignum = 0
# def hide_labels(fig, ax): #定义函数，隐藏其坐标，画无坐标图
#     global fignum
#     ax.set_xticks([])
#     ax.set_yticks([])
#     ax.xaxis.set_ticks_position('none')   # 设定ticks（刻度）位置
#     ax.yaxis.set_ticks_position('none')
#     ax.axis('tight')
#     fignum += 1   # 递归式的保存图片
#     fig.savefig("plot-types-%d.pdf" % fignum)  #来一个图片保存一次，且作为pdf格式
# x = np.linspace(-3, 3, 25)
# y1 = x**3+ 3 * x**2 + 10
# y2 = -1.5 * x**3 + 10*x**2 - 15
# fig, ax = plt.subplots(figsize=(4, 3))
# ax.plot(x, y1)
# ax.plot(x, y2)
# hide_labels(fig, ax)    #  隐藏坐标轴
# plt.show(block=True)
#
# fig, ax = plt.subplots(figsize=(4, 3)) #设定画布大小
# ax.step(x, y1)   #话阶梯
# ax.step(x, y2)
# hide_labels(fig, ax)  #隐藏轴的坐标
# plt.show(block=True)
#
# fig, ax = plt.subplots(figsize=(4, 3))
# width = 6/50.0
# ax.bar(x - width/2, y1, width=width, color="blue")   #bar图，设置颜色
# ax.bar(x + width/2, y2, width=width, color="green")
# hide_labels(fig, ax)
# plt.show(block=True)
#
# fig, ax = plt.subplots(figsize=(4, 3))
# ax.fill_between(x, y1, y2)   #fill填充区域
# hide_labels(fig, ax)
# plt.show(block=True)
#
# fig, ax = plt.subplots(figsize=(4, 3))
# ax.hist(y2, bins=30)      #hist直方图
# ax.hist(y1, bins=30)
# hide_labels(fig, ax)
# plt.show(block=True)
#
# fig, ax = plt.subplots(figsize=(4, 3))
# ax.errorbar(x, y2, yerr=y1, fmt='o-')     #设定点的误差条图 第二项为中心，带三项为误差值，最后数据标记参数
# hide_labels(fig, ax)
# plt.show(block=True)
#
# fig, ax = plt.subplots(figsize=(4, 3))
# ax.stem(x, y2, 'b', markerfmt='bs')    #
# ax.stem(x, y1, 'r', markerfmt='ro')
# hide_labels(fig, ax)
# plt.show(block=True)
#
# fig, ax = plt.subplots(figsize=(4, 3))
# x = np.linspace(0, 5, 50)
# ax.scatter(x, -1 + x + 0.25 * x**2 + 2 * np.random.rand(len(x)))    #scatter散点图
# ax.scatter(x, np.sqrt(x) + 2 * np.random.rand(len(x)), color="green")
# hide_labels(fig, ax)
# plt.show(block=True)
# fig, ax = plt.subplots(figsize=(3, 3))
# colors = ['yellowgreen', 'gold', 'lightskyblue', 'lightcoral']
# x = y = np.linspace(-2, 2, 10)
# X, Y = np.meshgrid(x, y)
# U = np.sin(X)
# V = np.sin(Y)
# ax.quiver(X, Y, U, V)    #箭头图形
# hide_labels(fig, ax)
# plt.show(block=True)
# 帮助文档的调用方式 e.g. help(plt.Axes.bar)
'''注释与标记'''
# fig, ax = plt.subplots(figsize=(8, 4))
# x = np.linspace(-20, 20, 100)
# y = np.sin(x) / x
# ax.plot(x, y)
# ax.set_ylabel("y label")   #标记
# ax.set_xlabel("x label")
# for label in ax.get_xticklabels() + ax.get_yticklabels(): #所有坐标轴旋转45度
#     label.set_rotation(45)
# plt.show(block=True)
''' 注释方法箭头，matplotlib支持Latex格式的使用，但是建议使用原始字符串更加方便，防止转义字符的歧义问题'''
# print(mpl.rcParams)  #查看当前注释格式
# fig, ax = plt.subplots(figsize=(12, 3))
# ax.set_yticks([])
# ax.set_xticks([])
# ax.set_xlim(-0.5, 3.5)
# ax.set_ylim(-0.05, 0.25)
# ax.axhline(0)    #绘制平行于x轴的水平参考线
# ax.text(0, 0.1, "Text label", fontsize=14, family="serif")
# ax.plot(1, 0, 'o')
# ax.annotate("Annotation",    #注释函数
#             fontsize=10, family="serif", #字号，字体
#             xy=(1, 0), xycoords='data',   #xy被注释的地方,xytext插入文本的地方，xycoords：被注释点的坐标系属性，
#             xytext=(+30, +50), textcoords='offset points',       #设定位置，'data'：以被注释的坐标点xy为参考 (默认值)
#             arrowprops=dict(arrowstyle="->", connectionstyle="arc3, rad=1"))   #注释和文本的连接方式
# ax.text(2, 0.1, r"Equation: $i\hbar\partial_t \Psi = \hat{H}\Psi$", fontsize=14, family="serif")
# fig.savefig("ch4-text-annotation.pdf")
# plt.show(block=True)
''' 多个子图 '''
# fig, axes = plt.subplots(ncols=2, nrows=3)   #横纵子图
# plt.show(block=True)
'''线形与线宽'''
# list(range(1, 12, 2))
# import sympy as s
# import numpy as np
# # a symbolic variable for x, and a numerical array with specific values of x
# sym_x = s.Symbol("x")  # 符号声明
# x = np.linspace(-2 * np.pi, 2 * np.pi, 100)
# def sin_expansion(x, n):
#     return s.lambdify(sym_x, s.sin(sym_x).series(n=n + 1).removeO(), 'numpy')(x)
# fig, ax = plt.subplots (figsize=(10, 6))
# ax.plot(x, np.sin(x), linewidth=4, color="red", label='sin(x)')
# colors = ["blue", "black"]
# linestyles = [':', '-.', '--']
# for idx, n in enumerate(range(1, 12, 2)):
#     ax.plot(x, sin_expansion(x, n), color=colors[idx // 3],
#             linestyle=linestyles[idx % 3], linewidth=3,
#             label="O(%d) approx." % (n + 1))
# ax.set_ylim(-1.1, 1.1)
# ax.set_xlim(-1.5 * np.pi, 1.5 * np.pi)
# ax.legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=5.0)  #前两个是定义注释大小，最后的参数是定义和图像之间的位置
# fig.subplots_adjust(right=.75)    #右边调整，使得图例说明可以显示
# plt.show(block=True)
'''
图例 
'''
# fig, axes = plt.subplots(1, 4, figsize=(16, 4))
# x = np.linspace(0, 1, 100)
# for n in range(4):  # 从1-4的元组
#     axes[n].plot(x, x, label="y(x) = x")   #使用axis的序号来确定位置
#     axes[n].plot(x, x + x**2, label="y(x) = x + x**2")
#     axes[n].legend(loc=n+1) # loc控制图例的位置。1~4为右上，左上，左下，右下
#     axes[n].set_title("legend(loc=%d)" % (n+1))
# fig.tight_layout()    # tight_layout会自动调整子图参数，使之填充整个图像区域
# fig.savefig("legend-loc.pdf")
# plt.show(block=True)

# fig, ax = plt.subplots(1, 1, figsize=(8.5, 3))
# x = np.linspace(-1, 1, 100)
# for n in range(1, 9):
#     ax.plot(x, n * x, label="y(x) = %d*x" % n)
# ax.legend(ncol=4, loc=3, bbox_to_anchor=(0, 1), fontsize=12)   #每一行ncol个元素，每一列col个元素
# #bbox_to_anchor控制位置，(0,1)表示在上方，字体12pt
# fig.subplots_adjust(top=.75)    #上面空出来一部分
# fig.savefig("legend-loc-2.pdf")
# plt.show(block=True)

'''轴标签和标题'''
# fig, ax = plt.subplots(figsize=(8, 3), subplot_kw={'facecolor': "#ebf5ff"})
# x = np.linspace(0, 50, 500)
# ax.plot(x, np.sin(x) * np.exp(-x/10), lw=2)
# ax.set_xlabel("x", labelpad=5,   # labelpad为x标签和图像之间的间距
#               fontsize=18, fontname='serif', color="blue")
# ax.set_ylabel("f(x)", labelpad=15,      # 设置横纵轴
#               fontsize=18, fontname='serif', color="blue")
# ax.set_title("axis labels and title example", loc='left',
#              fontsize=16, fontname='serif', color="blue")
# fig.tight_layout()    #贴紧边框
# fig.savefig("ch4-axis-labels.pdf")
# plt.show(block=True)

# 明确设置轴的范围，使用set_xlim,或者set_ylim,或者使用axis参数设置。

# x = np.linspace(0, 30, 500)
# y = np.sin(x) * np.exp(-x/10)
# fig, axes = plt.subplots(1, 3, figsize=(9, 3), subplot_kw={'facecolor': "#ebf5ff"})
# axes[0].plot(x, y, lw=2)   #线宽
# axes[0].set_xlim(-5, 35)   # 人为设定x轴的范围
# axes[0].set_ylim(-1, 1)
# axes[0].set_title("set_xlim / set_y_lim")
# axes[1].plot(x, y, lw=2)
# axes[1].axis('tight')   # 表示坐标轴紧密匹配绘制，相当于绘制尽可能紧的框子。
# axes[1].set_title("axis('tight')")
# axes[2].plot(x, y, lw=2)
# axes[2].axis('equal')    # 表示每个坐标轴的长度包含相同的像素点，也就是保持坐标轴的比例不变化。
# axes[2].set_title("axis('equal')")
# fig.savefig("ch4-axis-ranges.pdf")
# # plt.autoscale=True;'''注意也可使用autoscale方法进行打开和自动关闭功能'''
# plt.show(block=True)

'''设置轴的刻度线，刻度标签和网格'''
# mpl.ticker为刻度管理系统，Matplotlib将刻度分为主要(major)和次要刻度(minor tick).
# 主要刻度总是有默认的标签，而次要刻度还需要进一步标明
# x = np.linspace(-2 * np.pi, 2 * np.pi, 500)
# y = np.sin(x) * np.exp(-x**2/20)
# fig, axes = plt.subplots(1, 4, figsize=(12, 3))
# axes[0].plot(x, y, lw=2)
# axes[0].set_title("default ticks")
# axes[1].plot(x, y, lw=2)
# axes[1].set_yticks([-1, 0, 1])  #设置刻度
# axes[1].set_xticks([-5, 0, 5])
# axes[1].set_title("set_xticks")
# axes[2].plot(x, y, lw=2)
# # 函数set_major_locator接受mpl.ticker类。用于设置刻度
# axes[2].xaxis.set_major_locator(mpl.ticker.MaxNLocator(4))                # 设置主要刻度，设置最大刻度值
# axes[2].yaxis.set_major_locator(mpl.ticker.FixedLocator([-1, 0, 1]))      # 显式指定未知放置刻度值
# axes[2].xaxis.set_minor_locator(mpl.ticker.MaxNLocator(2))   #次要刻度仅仅给出标记，而不会给出具体的值。
# axes[2].yaxis.set_minor_locator(mpl.ticker.MaxNLocator(2))
# axes[2].set_title("set_major_locator") #设置标题
# axes[3].plot(x, y, lw=2)
# axes[3].set_yticks([-1, 0, 1])
# axes[3].set_xticks([-2 * np.pi, -np.pi, 0, np.pi, 2 * np.pi])
# axes[3].set_xticklabels(['$-2\pi$', '$-\pi$', 0, r'$\pi$', r'$2\pi$'])   #设定刻度标签，这里$……$中间为转义字符，表明按照符号显示
# axes[3].xaxis.set_minor_locator(mpl.ticker.FixedLocator([-3 * np.pi / 2, -np.pi/2, 0, np.pi/2, 3 * np.pi/2]))
# axes[3].yaxis.set_minor_locator(mpl.ticker.MaxNLocator(4))
# axes[3].set_title("set_xticklabels")
# fig.tight_layout()
# fig.savefig("ch4-axis-ticks.pdf")
# plt.show(block=True)
#
# ''' 网格线函数grid()设定如下'''
# fig, axes = plt.subplots(1, 3, figsize=(12, 4))
# x_major_ticker = mpl.ticker.MultipleLocator(4)  #以4为单位设定刻度线。
# x_minor_ticker = mpl.ticker.MultipleLocator(1)   #等式左边的主要克服和次要刻度分开赋值。
# y_major_ticker = mpl.ticker.MultipleLocator(0.5)
# y_minor_ticker = mpl.ticker.MultipleLocator(0.25)
# for ax in axes:
#     ax.plot(x, y, lw=2)
#     ax.xaxis.set_major_locator(x_major_ticker)
#     ax.yaxis.set_major_locator(y_major_ticker)
#     ax.xaxis.set_minor_locator(x_minor_ticker)
#     ax.yaxis.set_minor_locator(y_minor_ticker)
# axes[0].set_title("default grid")
# axes[0].grid()  # 默认栅格线
# axes[1].set_title("major/minor grid")
# axes[1].grid(color="blue", which="both", linestyle=':', linewidth=0.5)
# axes[2].set_title("individual x/y major/minor grid")
# axes[2].grid(color="grey", which="major", axis='x', linestyle='-', linewidth=0.5) #设定网格线
# axes[2].grid(color="grey", which="minor", axis='x', linestyle=':', linewidth=0.25) #which指定到使用于哪一种刻度
# axes[2].grid(color="grey", which="major", axis='y', linestyle='-', linewidth=0.5)
# fig.tight_layout()
# fig.savefig("ch4-axis-grid.pdf")
# plt.show(block=True)
'''使用ScalarFormatter设置科学计数法的显示格式'''
# fig, axes = plt.subplots(1, 2, figsize=(8, 3))
# x = np.linspace(0, 1e5, 100)
# y = x ** 2
# axes[0].plot(x, y, 'b.')
# axes[0].set_title("default labels", loc='right') #位置放在右边
# axes[1].plot(x, y, 'b')
# axes[1].set_title("scientific notation labels", loc='right')
# formatter = mpl.ticker.ScalarFormatter(useMathText=True)   # 以数学符号显式而非显示指数
# formatter.set_scientific(True)       # 选择科学计数法
# formatter.set_powerlimits((-1,1))    #  控制科学计数法的阈值
# axes[1].xaxis.set_major_formatter(formatter)
# axes[1].yaxis.set_major_formatter(formatter)
# fig.tight_layout()
# fig.savefig("ch4-axis-scientific.pdf")
# plt.show(block=True)
'''对数坐标显示 loglog(), semilog(), semilogy()，
或者使用axis()的轴，set_yscale()函数，显式指定log作为第一个参数'''
# fig, axes = plt.subplots(1, 3, figsize=(12, 3))
# x = np.linspace(0, 1e3, 100)
# y1, y2 = x**3, x**4
# axes[0].set_title('loglog')
# axes[0].loglog(x, y1, 'b', x, y2, 'r')
# axes[1].set_title('semilogy')
# axes[1].semilogy(x, y1, 'b', x, y2, 'r')
# axes[2].set_title('plot / set_xscale / set_yscale')
# axes[2].plot(x, y1, 'b', x, y2, 'r')
# axes[2].set_xscale('log')
# axes[2].set_yscale('log')
# fig.tight_layout()
# fig.savefig("ch4-axis-log-plots.pdf")
''' 双轴图 '''
# fig, ax1 = plt.subplots(figsize=(8, 4))
# r = np.linspace(0, 5, 100)
# a = 4 * np.pi * r ** 2  # area
# v = (4 * np.pi / 3) * r ** 3  # volume
# ax1.set_title("surface area and volume of a sphere", fontsize=16)
# ax1.set_xlabel("radius [m]", fontsize=16)
# ax1.plot(r, a, lw=2, color="blue")
# ax1.set_ylabel(r"surface area ($m^2$)", fontsize=16, color="blue")
# for label in ax1.get_yticklabels():     # 为轴标签设计颜色，使用for提取元素，然后为其指定颜色
#     label.set_color("blue")
# ax2 = ax1.twinx()  # 共享x轴，而分离y轴
# ax2.plot(r, v, lw=2, color="red")
# ax2.set_ylabel(r"volume ($m^3$)", fontsize=16, color="red")
# for label in ax2.get_yticklabels():
#     label.set_color("red")
# fig.tight_layout()         #铺满平面，让其显示更加合适
# fig.savefig("ch4-axis-twin-ax.pdf")
# plt.show(block=True)
''' 去除边框 spine()函数 '''
# x = np.linspace(-10, 10, 500)
# y = np.sin(x) / x
# fig, ax = plt.subplots(figsize=(8, 4))
# ax.plot(x, y, linewidth=2)
# # remove top and right spines
# ax.spines['right'].set_color('none')    #右边的颜色设定为无
# ax.spines['top'].set_color('none')
# # # remove top and right spine ticks
# ax.xaxis.set_ticks_position('bottom')   #仅仅使用底部和左边的刻度线
# ax.yaxis.set_ticks_position('left')
# # # move bottom and left spine to x = 0 and y = 0
# ax.spines['bottom'].set_position(('data', 0))
# ax.spines['left'].set_position(('data', 0))
# ax.set_xticks([-10, -5, 5, 10])
# ax.set_yticks([0.5, 1])
# # # give each label a solid background of white, to not overlap with the plot line
# for label in ax.get_xticklabels() + ax.get_yticklabels():
#     label.set_bbox({'facecolor': 'white',
#                     'edgecolor': 'white'})
# fig.tight_layout()
# # fig.savefig("ch4-axis-spines.pdf")
# plt.show(block=True)

'''图中图设计'''
# fig = plt.figure(figsize=(8, 4))
# def f(x):
#     return 1 / (1 + x ** 2) + 0.1 / (1 + ((3 - x) / 0.1) ** 2)
# def plot_and_format_axes(ax, x, f, fontsize):
#     ax.plot(x, f(x), linewidth=2)  # 画图
#     ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(5))  # 参数设定，刻度与坐标
#     ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(4))
#     ax.set_xlabel(r"$x$", fontsize=fontsize)
#     ax.set_ylabel(r"$f(x)$", fontsize=fontsize)
# # main graph主图
# ax = fig.add_axes([0.1, 0.15, 0.8, 0.8], facecolor="#f5f5f5")
# x = np.linspace(-4, 14, 1000)
# plot_and_format_axes(ax, x, f, 18)
# # inset内图
# x0, x1 = 2.5, 3.5
# ax.axvline(x0, ymax=0.3, color="grey", linestyle=":")  #设定截断的位置
# ax.axvline(x1, ymax=0.3, color="grey", linestyle=":")
# ax = fig.add_axes([0.5, 0.5, 0.38, 0.42], facecolor='none')  # 加入附图
# x = np.linspace(x0, x1, 1000)
# plot_and_format_axes(ax, x, f, 14)
# fig.savefig("ch4-advanced-axes-inset.pdf")
# plt.show(block=True)

''' plt.subplots()函数。'''
# ncols, nrows = 3, 3
# fig, axes = plt.subplots(nrows, ncols)
# for m in range(nrows):
#     for n in range(ncols):
#         axes[m, n].set_xticks([])
#         axes[m, n].set_yticks([])
#         axes[m, n].text(0.5, 0.5, "axes[%d, %d]" % (m, n),   #%d插入符号
#                         horizontalalignment='center')     #文本管理。位置放在中心
# plt.show(block=True)
'''sharex和sharey用于共享x,y轴，
而plt.subplot()返回的axis的实例是压缩的，squeeze()使其不会忽视维数为1的值
如果下面不同squeeze=0，那么刻度尺只有一个，加了一个变为两个'''
# fig, axes = plt.subplots(2, 2, figsize=(6, 6), sharex=True, sharey=True, squeeze=False)
# x1 = np.random.randn(100)
# x2 = np.random.randn(100)
# axes[0, 0].set_title("Uncorrelated")
# axes[0, 0].scatter(x1, x2)
# axes[0, 1].set_title("Weakly positively correlated")
# axes[0, 1].scatter(x1, x1 + x2)
# axes[1, 0].set_title("Weakly negatively correlated")
# axes[1, 0].scatter(x1, -x1 + x2)
# axes[1, 1].set_title("Strongly correlated")
# axes[1, 1].scatter(x1, x1 + 0.15 * x2)
# axes[1, 1].set_xlabel("x")
# axes[1, 0].set_xlabel("x")
# axes[0, 0].set_ylabel("y")
# axes[1, 0].set_ylabel("y")
# #subplots_adjust()用于适应性调整间距，左右上下加上间距宽度和高度值。
# plt.subplots_adjust(left=0.1, right=0.95, bottom=0.1, top=0.95, wspace=0.1, hspace=0.2)
# fig.savefig("ch4-advanced-axes-subplots.pdf")
# plt.show(block=True)

'''plt.subplot2grid()提供呢更加灵活axes管理布局，'''
# fig = plt.figure()
# def clear_ticklabels(ax):
#     ax.set_yticklabels([])
#     ax.set_xticklabels([])
# ax0 = plt.subplot2grid((3, 3), (0, 0))
# ax1 = plt.subplot2grid((3, 3), (0, 1))
# ax2 = plt.subplot2grid((3, 3), (1, 0), colspan=2)    # 横向和纵向跨度
# ax3 = plt.subplot2grid((3, 3), (2, 0), colspan=3)
# ax4 = plt.subplot2grid((3, 3), (0, 2), rowspan=2)
# axes = [ax0, ax1, ax2, ax3, ax4]
# [ax.text(0.5, 0.5, "ax%d" % n, horizontalalignment='center') for n, ax in enumerate(axes)]
# # [clear_ticklabels(ax) for ax in axes]       # 清楚所有图上的刻度
# fig.savefig("ch4-advanced-axes-subplot2grid.pdf")
# plt.show(block=True)

'''通用框架 GridSpec,仅仅用于网络布局，可以创建所有行和列等高，或者成比例的网络'''
# from matplotlib.gridspec import GridSpec
# fig = plt.figure(figsize=(6, 4))
# gs = mpl.gridspec.GridSpec(4, 4)    #创建一个用于布局的网格，然后给axis用于绘制
# ax0 = fig.add_subplot(gs[0, 0])
# ax1 = fig.add_subplot(gs[1, 1])     #使用add_subplot()将函数传给
# ax2 = fig.add_subplot(gs[2, 2])
# ax3 = fig.add_subplot(gs[3, 3])
# ax4 = fig.add_subplot(gs[0, 1:])
# ax5 = fig.add_subplot(gs[1:, 0])
# ax6 = fig.add_subplot(gs[1, 2:])
# ax7 = fig.add_subplot(gs[2:, 1])
# ax8 = fig.add_subplot(gs[2, 3])
# ax9 = fig.add_subplot(gs[3, 2])
# def clear_ticklabels(ax):
#     ax.set_yticklabels([])
#     ax.set_xticklabels([])
# axes = [ax0, ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9]
# [ax.text(0.5, 0.5, "ax%d" % n, horizontalalignment='center') for n, ax in enumerate(axes)]
# [clear_ticklabels(ax) for ax in axes]
# # fig.savefig("ch4-advanced-axes-gridspec-1.pdf")
# plt.show(block=True)
# # 另一个例子
# fig = plt.figure(figsize=(4, 4))
# gs = mpl.gridspec.GridSpec(2, 2,
#                            width_ratios=[4, 1],
#                            height_ratios=[1, 4],
#                            wspace=0.05, hspace=0.05)
# ax0 = fig.add_subplot(gs[1, 0])
# ax1 = fig.add_subplot(gs[0, 0])
# ax2 = fig.add_subplot(gs[1, 1])
# def clear_ticklabels(ax):
#     ax.set_yticklabels([])
#     ax.set_xticklabels([])
# axes = [ax0, ax1, ax2]
# [ax.text(0.5, 0.5, "ax%d" % n, horizontalalignment='center') for n, ax in enumerate(axes)]
# [clear_ticklabels(ax) for ax in axes]
# fig.savefig("ch4-advanced-axes-gridspec-2.pdf")
# plt.show(block=True)
'''绘制色图，使用pcolor()和imshow()函数，此外还有contour()和contourf()函数'''
# x = y = np.linspace(-2, 2, 500)
# X, Y = np.meshgrid(x, y)     # 网格创建
# R1 = np.sqrt((X+0.5)**2 + (Y+0.5)**2)
# R2 = np.sqrt((X+0.5)**2 + (Y-0.5)**2)
# R3 = np.sqrt((X-0.5)**2 + (Y+0.5)**2)
# R4 = np.sqrt((X-0.5)**2 + (Y-0.5)**2)
# Z = np.sin(10 * R1) / (10 * R1) + np.sin(20 * R4) / (20 * R4)
# fig, ax = plt.subplots(figsize=(6, 5))
# p = ax.pcolor(X, Y, Z, cmap='seismic', vmin=-abs(Z).max(), vmax=abs(Z).max())
# ax.axis('tight')
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# cb = fig.colorbar(p, ax=ax)
# plt.show(block=True)
# # 另外一个例子
# Z = 1/R1 - 1/R2 - 1/R3 + 1/R4
# fig, ax = plt.subplots(figsize=(6, 5))
# im = ax.imshow(Z, vmin=-1, vmax=1, cmap=mpl.cm.bwr,        #camp为颜色设定，bwr应该是blue white red演化的意思
#                extent=[x.min(), x.max(), y.min(), y.max()])
# im.set_interpolation('bilinear')     #设定解释器
# ax.axis('tight')
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# cb = fig.colorbar(p, ax=ax)
# plt.show(block=True)
'''一个更加复杂的例子'''
# x = y = np.linspace(-2, 2, 150)
# X, Y = np.meshgrid(x, y)
# R1 = np.sqrt((X+0.5)**2 + (Y+0.5)**2)
# R2 = np.sqrt((X+0.5)**2 + (Y-0.5)**2)
# R3 = np.sqrt((X-0.5)**2 + (Y+0.5)**2)
# R4 = np.sqrt((X-0.5)**2 + (Y-0.5)**2)
# fig, axes = plt.subplots(1, 4, figsize=(14, 3))
# Z = np.sin(10 * R1) / (10 * R1) + np.sin(20 * R4) / (20 * R4)
# p = axes[0].pcolor(X, Y, Z, cmap='seismic', vmin=-abs(Z).max(), vmax=abs(Z).max())
# axes[0].axis('tight')
# axes[0].set_xlabel('x')
# axes[0].set_ylabel('y')
# axes[0].set_title("pcolor")
# axes[0].xaxis.set_major_locator(mpl.ticker.MaxNLocator(4))
# axes[0].yaxis.set_major_locator(mpl.ticker.MaxNLocator(4))
# cb = fig.colorbar(p, ax=axes[0])
# cb.set_label("z")
# cb.set_ticks([-1, -.5, 0, .5, 1])
# Z = 1/R1 - 1/R2 - 1/R3 + 1/R4
# im = axes[1].imshow(Z, vmin=-1, vmax=1, cmap=mpl.cm.bwr,
#                extent=[x.min(), x.max(), y.min(), y.max()])
# im.set_interpolation('bilinear')         # 双线性插值
# axes[1].axis('tight')
# axes[1].set_xlabel('x')
# axes[1].set_ylabel('y')
# axes[1].set_title("imshow")
# cb = fig.colorbar(im, ax=axes[1])
# axes[1].xaxis.set_major_locator(mpl.ticker.MaxNLocator(4))
# axes[1].yaxis.set_major_locator(mpl.ticker.MaxNLocator(4))
# #cb.ax.set_axes_locator(mpl.ticker.MaxNLocator(4))
# cb.set_label("z")
# cb.set_ticks([-1, -.5, 0, .5, 1])
# x = y = np.linspace(0, 1, 75)
# X, Y = np.meshgrid(x, y)
# Z = - 2 * np.cos(2 * np.pi * X) * np.cos(2 * np.pi * Y) - 0.7 * np.cos(np.pi - 4*np.pi*X)
# c = axes[2].contour(X, Y, Z, 15, cmap=mpl.cm.RdBu, vmin=-1, vmax=1)
# axes[2].axis('tight')
# axes[2].set_xlabel('x')
# axes[2].set_ylabel('y')
# axes[2].set_title("contour")
# axes[2].xaxis.set_major_locator(mpl.ticker.MaxNLocator(4))
# axes[2].yaxis.set_major_locator(mpl.ticker.MaxNLocator(4))
# c = axes[3].contourf(X, Y, Z, 15, cmap=mpl.cm.RdBu, vmin=-1, vmax=1)
# axes[3].axis('tight')
# axes[3].set_xlabel('x')
# axes[3].set_ylabel('y')
# axes[3].set_title("contourf")
# axes[3].xaxis.set_major_locator(mpl.ticker.MaxNLocator(4))
# axes[3].yaxis.set_major_locator(mpl.ticker.MaxNLocator(4))
# fig.tight_layout()
# fig.savefig('ch4-colormaps.pdf')
# plt.show(block=True)

'''三维作图方法 Axes3D对象，'''
from mpl_toolkits.mplot3d.axes3d import Axes3D
x = y = np.linspace(-3, 3, 74)
X, Y = np.meshgrid(x, y)
R = np.sqrt(X**2 + Y**2)
Z = np.sin(4 * R) / R
fig, axes = plt.subplots(1, 3, figsize=(14, 4), subplot_kw={'projection': '3d'})   #3D图设计显式声明
def title_and_labels(ax, title):
    ax.set_title(title)
    ax.set_xlabel("$x$", fontsize=16)
    ax.set_ylabel("$y$", fontsize=16)
    ax.set_zlabel("$z$", fontsize=16)
norm = mpl.colors.Normalize(-abs(Z).max(), abs(Z).max())
p = axes[0].plot_surface(X, Y, Z, rstride=1, cstride=1, linewidth=0, antialiased=False, norm=norm, cmap=mpl.cm.Blues)
# rstride选择行步长，antialiased用于反锯齿，提升图形质量的
cb = fig.colorbar(p, ax=axes[0], shrink=0.6)   #缩小尺寸
title_and_labels(axes[0], "plot_surface")
p = axes[1].plot_wireframe(X, Y, Z, rstride=2, cstride=2, color="grey")
title_and_labels(axes[1], "plot_wireframe")
cset = axes[2].contour(X, Y, Z, zdir='z', offset=0, norm=norm, cmap=mpl.cm.Blues)
# zdir='z'，offset=-2：设置一个z=-2的高度，在z轴的方向将这个3d图像压到一个平面上。
cset = axes[2].contour(X, Y, Z, zdir='y', offset=3, norm=norm, cmap=mpl.cm.Blues)
title_and_labels(axes[2], "contour")
fig.tight_layout()
fig.savefig("ch4-3d-plots.png", dpi=200)
plt.show(block=True)
'''matplotlib可能在交互性和3D上有些缺陷，太专业的不建议使用'''





