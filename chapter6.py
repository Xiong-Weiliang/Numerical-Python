# TODO 优化
''' cvxopt也是可行的解决凸优化的库,主要使用scipy的optimize模块进行优化，其中的gloden实现了黄金分割法,但是作为二分法，其收敛很慢;
可使用二阶近似法（Newton）,但是初始点很远的情况下可能不会收敛, 于是可以两种方法结合,e.g. optimize.brent就是黄金分割法的变体，使用逆抛物线插值.
使用optimize.minimize_scalar通过method参数(brent, gloden,fminbound函数有界限制)决定使用什么方法求解'''
'''一般对于可以解析解进行解决的问题可以使用scipy的数值化方法进行解决如下'''
import numpy as np
import sympy
sympy.init_printing()
from scipy import optimize
import matplotlib.pyplot as plt
import cvxopt
# import matplotlib      # ctrl+/ 批量注释
# r, h = sympy.symbols("r, h")  # 符号定义
# Area = 2 * sympy.pi * r**2 + 2 * sympy.pi * r * h
# Volume = sympy.pi * r**2 * h
# h_r = sympy.solve(Volume - 1)[0]
# Area_r=Area.subs(h_r)        # subs函数带入值
# # 使用函数定义如下, 注意函数定义必须位于调用之前
# def f(r):   # 函数定义
#     return 2* np.pi * r ** 2 + 2/r
# # 不同的接口调用
# r_min = optimize.brent(f, brack=(0.1, 4))     # 指定算法启动区间
# print(r_min)
# print(f(r_min))
# optimize.minimize_scalar(f, bracket=(0.1, 4))        # bracket 用于指定算法起始区间. 对应接口 minimize_scalar
# r = np.linspace(0, 2, 100)[1:]
# fig, ax = plt.subplots(figsize=(8, 4))
# ax.plot(r, f(r), lw=2, color='b')
# ax.plot(r_min, f(r_min), 'r*', markersize=15)
# ax.set_title(r"$f(r) = 2\pi r^2+2/r$", fontsize=18)
# ax.set_xlabel(r"$r$", fontsize=18)
# ax.set_xticks([0, 0.5, 1, 1.5, 2])
# ax.set_ylim(0, 30)
# fig.tight_layout()
# fig.savefig('ch6-univariate-optimization-example.pdf')
# plt.show(block=True)
'''对于多变量调用问题, optimize.fmin_ncg实现了牛顿法, 需要计算Hessian矩阵和梯段'''
# x1, x2 = sympy.symbols("x_1, x_2")
# f_sym = (x1-1)**4 + 5 * (x2-1)**2 - 2*x1*x2
# fprime_sym = [f_sym.diff(x_) for x_ in (x1, x2)]   # 分别对x1与x2求导数
# # Gradient
# print(sympy.Matrix(fprime_sym))   # 输出矩阵形式
# fhess_sym = [[f_sym.diff(x1_, x2_) for x1_ in (x1, x2)] for x2_ in (x1, x2)]    # 上述求取导数的循环调用.
# # Hessian
# sympy.Matrix(fhess_sym)
# f_lmbda = sympy.lambdify((x1, x2), f_sym, 'numpy')  # lambdify 函数将 SymPy 表达式转换为 Python 函数。相当于创建了倒数函数.
# fprime_lmbda = sympy.lambdify((x1, x2), fprime_sym, 'numpy')
# fhess_lmbda = sympy.lambdify((x1, x2), fhess_sym, 'numpy')
# def func_XY_X_Y(f):
#     """
#     Wrapper for f(X) -> f(X[0], X[1])
#     """
#     return lambda X: np.array(f(X[0], X[1]))      # scipy的优化要求信息位于同一个数组内部，因此重新封装.
# f = func_XY_X_Y(f_lmbda)       # 经过转化成为矢量化的Python函数
# fprime = func_XY_X_Y(fprime_lmbda)
# fhess = func_XY_X_Y(fhess_lmbda)
# X_opt = optimize.fmin_ncg(f, (0, 0), fprime=fprime, fhess=fhess)  # fmin_ncg()函数用于
# print(X_opt)
# fig, ax = plt.subplots(figsize=(6, 4))
# x_ = y_ = np.linspace(-1, 4, 100)
# X, Y = np.meshgrid(x_, y_)
# c = ax.contour(X, Y, f_lmbda(X, Y), 50)    # 使用x,y的值进行等高线绘制.
# ax.plot(X_opt[0], X_opt[1], 'r*', markersize=15)
# ax.set_xlabel(r"$x_1$", fontsize=18)
# ax.set_ylabel(r"$x_2$", fontsize=18)
# plt.colorbar(c, ax=ax)
# fig.tight_layout()
# fig.savefig('ch6-examaple-two-dim.pdf');
# plt.show(block=True)
'''如果不好计算各阶的导数矩阵，可以近似估计使用BFGS方法(fmin_bfgs)和共轭梯度法(fmin_cg),这些方法的优势在于不需要计算梯度,更加稳定，但是函数自身计算量远远增大，
 再实际使用时候，应该首先使用 BFGS 方法. '''
# def f(X):
#     x, y = X
#     return (4 * np.sin(np.pi * x) + 6 * np.sin(np.pi * y)) + (x - 1)**2 + (y - 1)**2
# x_start = optimize.brute(f, (slice(-3, 5, 0.5), slice(-3, 5, 0.5)), finish=None)    # 使用暴力进行搜索初始点.
#    # slice依次对坐标轴进行切片处理, finish防止初始点就进行优化.
# x_opt = optimize.fmin_bfgs(f, x_start)
# print(x_opt)      # [1.47 1.48]
# print(f(x_opt))   # [-9.520]
def func_X_Y_to_XY(f, X, Y):      # 使用封装函数对其f函数的输入参数进行重新排列.
    s = np.shape(X)
    return f(np.vstack([X.ravel(), Y.ravel()])).reshape(*s)  # ravel()方法将数组维度拉成一维数组; 列表前加*号，会将列表拆分成一个一个的独立元素
# fig, ax = plt.subplots(figsize=(6, 4))
# x_ = y_ = np.linspace(-3, 5, 100)
# X, Y = np.meshgrid(x_, y_)
# c = ax.contour(X, Y, func_X_Y_to_XY(f, X, Y), 25)
# ax.plot(x_opt[0], x_opt[1], 'r*', markersize=15)
# ax.set_xlabel(r"$x_1$", fontsize=18)
# ax.set_ylabel(r"$x_2$", fontsize=18)
# plt.colorbar(c, ax=ax)
# fig.tight_layout()
# plt.show(block=True)
''' 事实上可以使用统一接口minimize_scalar, minimize(method=指定方法) '''
# 再多变量优化方法中可以使用Levenberg_Marquardt方法进行求解, 属于非线性多变量求解方法, 具体是在每次迭代中进行线性化, optimize.leastsq进行调用。
# def f(x, beta0, beta1, beta2):
#     return beta0 + beta1 * np.exp(-beta2 * x**2)
# beta = (0.25, 0.75, 0.5)
# xdata = np.linspace(0, 5, 50)
# y = f(xdata, *beta)
# ydata = y + 0.05 * np.random.randn(len(xdata))
# def g(beta):
#     return ydata - f(xdata, *beta)
# beta_start = (3, 4, 6)   # 初始点设计
# beta_opt, beta_cov = optimize.leastsq(g, beta_start)
# print(beta_opt)    # [0.23373487 0.76033771 0.44128751]
# fig, ax = plt.subplots()
# ax.scatter(xdata, ydata, label="samples")
# ax.plot(xdata, y, 'r', lw=2, label="true model")
# ax.plot(xdata, f(xdata, *beta_opt), 'b', lw=2, label="fitted model")
# ax.set_xlim(0, 5)
# ax.set_xlabel(r"$x$", fontsize=18)
# ax.set_ylabel(r"$f(x, \beta)$", fontsize=18)
# ax.legend()
# fig.tight_layout()
# fig.savefig('ch6-nonlinear-least-square.pdf')
# beta_opt, beta_cov = optimize.curve_fit(f, xdata, ydata)   # 统一接口, curve_fit()，可以不再显式指定函数g
# beta_opt
# plt.show(block=True)
# print(xdata)
''' 有约束的情况下可以使用L-BFGS-B算法, 利用bound设定边界 '''
# Bounds
# def f(X):
#     x, y = X
#     return (x-1)**2 + (y-1)**2
# x_opt = optimize.minimize(f, [0, 0], method='BFGS').x   # 导出最优化的解值.
# bnd_x1, bnd_x2 = (2, 3), (0, 2) # 边界设定
# x_cons_opt = optimize.minimize(f, [0, 0], method='L-BFGS-B', bounds=[bnd_x1, bnd_x2]).x
# fig, ax = plt.subplots(figsize=(6, 4))
# x_ = y_ = np.linspace(-1, 3, 100)
# X, Y = np.meshgrid(x_, y_)
# c = ax.contour(X, Y, func_X_Y_to_XY(f, X, Y), 50)
# ax.plot(x_opt[0], x_opt[1], 'b*', markersize=15)
# ax.plot(x_cons_opt[0], x_cons_opt[1], 'r*', markersize=15)
# bound_rect = plt.Rectangle((bnd_x1[0], bnd_x2[0]),        # 顶点
#                            bnd_x1[1] - bnd_x1[0], bnd_x2[1] - bnd_x2[0],
#                            facecolor="grey")
# ax.add_patch(bound_rect)
# ax.set_xlabel(r"$x_1$", fontsize=18)
# ax.set_ylabel(r"$x_2$", fontsize=18)
# plt.colorbar(c, ax=ax)      # 颜色板
# fig.tight_layout()          #tight_layout ()函数用于自动调整子图参数以提供指定的填充。
# fig.savefig('ch6-example-constraint-bound.pdf');
# plt.show(block=True)
''' Lagrange multiplier '''
# x = x1, x2, x3, l = sympy.symbols("x_1, x_2, x_3, lambda")
# f = x1 * x2 * x3
# g = 2 * (x1 * x2 + x2 * x3 + x3 * x1) - 1
# L = f + l * g      # 拉格朗日函数方法.
# grad_L = [sympy.diff(L, x_) for x_ in x]
# sols = sympy.solve(grad_L)    #求解梯度为0的值.
# print(sols)
# g.subs(sols[0])
# f.subs(sols[0])
# def f(X):
#     return -X[0] * X[1] * X[2]
# def g(X):
#     return 2 * (X[0]*X[1] + X[1] * X[2] + X[2] * X[0]) - 1
# constraints = [dict(type='eq', fun=g)]    # SLSQP 序列二次规划方法. 使用字典形式导入约束.
# result = optimize.minimize(f, [0.5, 1, 1.5], method='SLSQP', constraints=constraints)
# plt.show(block=True)
#
# def f(X):
#     return (X[0] - 1)**2 + (X[1] - 1)**2
# def g(X):
#     return X[1] - 1.75 - (X[0] - 0.75)**4
# x_opt = optimize.minimize(f, (0, 0), method='BFGS').x
# constraints = [dict(type='ineq', fun=g)]    # 不等式约束加上指定的函数.
# x_cons_opt = optimize.minimize(f, (0, 0), method='SLSQP', constraints=constraints).x
# x_cons_opt = optimize.minimize(f, (0, 0), method='COBYLA', constraints=constraints).x     #对于仅仅含有不等式约束的问题, 可以使用 COBYLA 求解.
# fig, ax = plt.subplots(figsize=(6, 4))
# x_ = y_ = np.linspace(-1, 3, 100)
# X, Y = np.meshgrid(x_, y_)
# c = ax.contour(X, Y, func_X_Y_to_XY(f, X, Y), 50)
# ax.plot(x_opt[0], x_opt[1], 'b*', markersize=15)
# ax.plot(x_, 1.75 + (x_-0.75)**4, 'k-', markersize=15)
# ax.fill_between(x_, 1.75 + (x_-0.75)**4, 3, color="grey")
# ax.plot(x_cons_opt[0], x_cons_opt[1], 'r*', markersize=15)
# ax.set_ylim(-1, 3)
# ax.set_xlabel(r"$x_0$", fontsize=18)
# ax.set_ylabel(r"$x_1$", fontsize=18)
# plt.colorbar(c, ax=ax)
# fig.tight_layout()
# fig.savefig('ch6-example-constraint-inequality.pdf');
# plt.show(block=True)
## cvxopt库中的 Linear programming
c = np.array([-1.0, 2.0, -3.0])
A = np.array([[ 1.0, 1.0, 0.0],
              [-1.0, 3.0, 0.0],
              [ 0.0, -1.0, 1.0]])
b = np.array([1.0, 2.0, 3.0])
A_ = cvxopt.matrix(A)   # 必须使用特定的结构对numpy进行转化才可以.
b_ = cvxopt.matrix(b)
c_ = cvxopt.matrix(c)
sol = cvxopt.solvers.lp(c_, A_, b_)
x = np.array(sol['x'])
print(x)










