# TODO 方程求解   使用基于numpy上的科学计算库scipy进行符号化求解，使用sympy进行数值求解
"""对于非线性函数，使用scipy中的root_finding函数进行求解。
对于线性方程组，使用sympy.linalg模块进行计算，非线性使用optimize模块计算"""
#  注意在pycharm中，不需要matplotlin online 的操作
from scipy import linalg as la
from scipy import optimize
import sympy
sympy.init_printing()     # 初始化打印，能够用print()为数学表达式打印unicode字符。
import numpy as np
import matplotlib.pyplot as plt
# from __future__ import division   # 使得整数除法在python2，3中表现相同。
# __future__”是一个模块而非单独的函数,用于解决关于版本的问题，“__future__”目的是把下一个版本的特性导入到当前版本
# fig, ax = plt.subplots(figsize=(8, 4))
# x1 = np.linspace(-4, 2, 100)
# x2_1 = (4 - 2 * x1)/3
# x2_2 = (3 - 5 * x1)/4
# ax.plot(x1, x2_1, 'r', lw=2, label=r"$2x_1+3x_2-4=0$")
# ax.plot(x1, x2_2, 'b', lw=2, label=r"$5x_1+4x_2-3=0$")
# A = np.array([[2, 3], [5, 4]])
# b = np.array([4, 3])
# x = la.solve(A, b)      # 求解函数
# ax.plot(x[0], x[1], 'ko', lw=2)    # k为颜色，o为线型
# ax.annotate("The intersection point of\nthe two lines is the solution\nto the equation system",
#             xy=(x[0], x[1]), xycoords='data',   # xycoords：用于设置xy的偏移方式
#             xytext=(-120, -75), textcoords='offset points',  # textcoords：用于设置xytext的偏移方式，要么以点为偏移points,要么以像素为偏移pixels。
#             arrowprops=dict(arrowstyle="->", connectionstyle="arc3, rad=-.3"))
# # arrowprops：箭头参数,参数类型为字典dict。width：箭头的宽度，以点为单位 headwidth：箭头底部的宽度，以点为单位 headlength：箭头的长度，以点为单位
# # shrink：从两端“收缩”的分数 facecolor：箭头颜色 arrowstyle：箭头的样式 connectionstyle：用于设置连接方式，可以设置弧度等
# ax.set_xlabel(r"$x_1$", fontsize=18)
# ax.set_ylabel(r"$x_2$", fontsize=18)
# ax.legend();
# fig.tight_layout()
# fig.savefig('ch5-linear-systems-simple.pdf')
# plt.show(block=True)
# A = sympy.Matrix([[2, 3], [5, 4]])
# b = sympy.Matrix([4, 3])
# # print(A.rank())   # 2，注意这里是调用的属性不是函数，。
# # print(A.condition_number())  # 求条件数，接近于1时候表明方程良定的，否则距离1很大时候视之为病态的。
# '''类似的，使用A.norm()求解其范数,而在Numpy中，使用numpy.linalg.matrix_rank/cond/norm求解'''
# L, U, P = A.LUdecomposition()  # LU 分解，将A化作一个下三角和一个上三角矩阵。
# 这个P是置换矩阵。
# print(P)   # 空阵
# x = A.solve(b)      # 首选方程求解方法。
# print(x)
'''数值方法求解，基于Scipy库的linalg.slove()'''
# A = np.array([[2, 3], [5, 4]])
# b = np.array([4, 3])
# Ran=np.linalg.matrix_rank(A)
# Con=np.linalg.cond(A)
# Nor=np.linalg.norm(A)
# print(Ran,Con,Nor)
# P, L, U = la.lu(A)    # 注意这里参数顺序，不是L，U，P
# print(P)     # [0 1; 1 0]阵，置换矩阵。
# Ma=np.dot(P, np.dot(L, U))
# print(Ma)
'''以下表明数值解numpy/scipy和符号解sympy的特殊之处，前者可能有误差，后者可能会有很长的表达式'''
p = sympy.symbols("p", positive=True)
A = sympy.Matrix([[1, sympy.sqrt(p)], [1, 1/sympy.sqrt(p)]])
b = sympy.Matrix([1, 2])
sympy.simplify(A.solve(b))    #简化求解表达式
# Symbolic problem specification
p = sympy.symbols("p", positive=True)
A = sympy.Matrix([[1, sympy.sqrt(p)], [1, 1/sympy.sqrt(p)]])
b = sympy.Matrix([1, 2])
# Solve symbolically
# x_sym_sol = A.solve(b)
# x_sym_sol.simplify()
# print(x_sym_sol)
# Acond = A.condition_number().simplify()
# # Function for solving numerically
# AA = lambda p: np.array([[1, np.sqrt(p)], [1, 1/np.sqrt(p)]])
# bb = np.array([1, 2])   # 定义lambda表达式，随后带入值进行计算
# x_num_sol = lambda p: np.linalg.solve(AA(p), bb)
# # Graph the difference between the symbolic (exact) and numerical results.
# p_vec = np.linspace(0.9, 1.1, 200)
# fig, axes = plt.subplots(1, 2, figsize=(12, 4))
# for n in range(2):  # range () 函数返回数字序列，默认从 0 开始，默认以 1 递增，即画两张图
#     x_sym = np.array([x_sym_sol[n].subs(p, pp).evalf() for pp in p_vec])  # evalf()函数可以用求出表达式的浮点数。
#     x_num = np.array([x_num_sol(pp)[n] for pp in p_vec])
#     axes[0].plot(p_vec, (x_num - x_sym)/x_sym, 'k')
# axes[0].set_title("Error in solution\n(numerical - symbolic)/symbolic")
# axes[0].set_xlabel(r'$p$', fontsize=18)
# axes[1].plot(p_vec, [Acond.subs(p, pp).evalf() for pp in p_vec])
# axes[1].set_title("Condition number")
# axes[1].set_xlabel(r'$p$', fontsize=18)
# fig.tight_layout()
# fig.savefig('ch5-linear-systems-condition-number.pdf')
# plt.show(block=True)
'''  '''
# unknown = sympy.symbols("x, y, z")   # 求解的未知参数值
# A = sympy.Matrix([[1, 2, 3], [4, 5, 6]])
# x = sympy.Matrix(unknown)
# b = sympy.Matrix([7, 8])
# AA = A * x - b
# c=sympy.solve(A*x - b, unknown)
# print(c)   # 打印出符号解 {x: z - 19/3, y: 20/3 - 2*z}
'''欠定方程组和带有噪声的超定 方程组的函数曲线最小二乘拟合'''
# 在sympy中，使用solve_least_squares求解最小二乘解，在scipy中使用la.lstsq函数进行数值求解
# np.random.seed(1234)
# define true model parameters
# x = np.linspace(-1, 1, 100)
# a, b, c = 1, 2, 3
# y_exact = a + b * x + c * x**2
# # simulate noisy data points
# m = 100
# X = 1 - 2 * np.random.rand(m)
# Y = a + b * X + c * X**2 + np.random.randn(m)
# # fit the data to the model using linear least square
# # vstack的作用： Stack arrays in sequence vertically (row wise).
# A = np.vstack([X**0, X**1, X**2])  # see np.vander for alternative
# sol, r, rank, sv = la.lstsq(A.T, Y)   # 返回参数： 求解向量，误差平方和，矩阵A的秩rank，和奇异值sv
# y_fit = sol[0] + sol[1] * x + sol[2] * x**2
# fig, ax = plt.subplots(figsize=(12, 4))
# ax.plot(X, Y, 'go', alpha=0.5, label='Simulated data')
# ax.plot(x, y_exact, 'k', lw=2, label='True value $y = 1 + 2x + 3x^2$')
# ax.plot(x, y_fit, 'b', lw=2, label='Least square fit')
# ax.set_xlabel(r"$x$", fontsize=18)
# ax.set_ylabel(r"$y$", fontsize=18)
# ax.legend(loc=2);
# fig.savefig('ch5-linear-systems-least-square.pdf')
# plt.show(block=True)
# '''线性最小二乘法'''
# # fit the data to the model using linear least square:
# # 1st order polynomial
# A = np.vstack([X**n for n in range(2)])
# sol, r, rank, sv = la.lstsq(A.T, Y)
# y_fit1 = sum([s * x**n for n, s in enumerate(sol)])
# # 15th order polynomial
# A = np.vstack([X**n for n in range(16)])
# sol, r, rank, sv = la.lstsq(A.T, Y)
# y_fit15 = sum([s * x**n for n, s in enumerate(sol)])
# fig, ax = plt.subplots(figsize=(12, 4))
# ax.plot(X, Y, 'go', alpha=0.5, label='Simulated data')
# ax.plot(x, y_exact, 'k', lw=2, label='True value $y = 1 + 2x + 3x^2$')
# ax.plot(x, y_fit1, 'b', lw=2, label='Least square fit [1st order]')
# ax.plot(x, y_fit15, 'm', lw=2, label='Least square fit [15th order]')
# ax.set_xlabel(r"$x$", fontsize=18)
# ax.set_ylabel(r"$y$", fontsize=18)
# ax.legend(loc=2);
# fig.savefig('ch5-linear-systems-least-square-2.pdf')
# plt.show(block=True)
'''特征值问题，在sympy中，可以使用matrix类的eigenvals和eigenvects方法（主要可以适用于符号矩阵）
 前者返回一个字典，为特征值的值和重数，后者返回一个元组，包含特征值，重数和特征向量'''
# eps, delta = sympy.symbols("epsilon, delta")
# H = sympy.Matrix([[eps, delta], [delta, -eps]])
# print(H)
# eval1, eval2 = H.eigenvals()
# print(eval1, eval2 )
# (eval1, _, evec1), (eval2, _, evec2) = H.eigenvects()
# b=sympy.simplify(evec1[0].T * evec2[0])  # 特征向量之积为0
# print(b)
# 而对于维数很高的数组，sympy构成的解极其复杂，必须使用scipy中的数值解法
# 函数linalg.eig()，或者linalg.eigvals()
# A = np.array([[1, 3, 5], [3, 5, 3], [5, 3, 9]])
# evals, evecs = la.eig(A)
# print(evals,evecs)
# print(la.eigvalsh(A))
# '''非线性方程组的求解，即寻根程序，注意，没有通用方法确保非线性的一个或者多个解，也不能确定得到的解是否唯一'''
# # 在sympy中，使用sympy.slove对单变量求解
# x, a, b, c = sympy.symbols("x, a, b, c")
# e = a + b*x + c*x**2
# sol1, sol2 = sympy.solve(e, x)     # 返回两个符号解
# print(sol1, sol2)
# 非线性函数作图
# x = np.linspace(-2, 2, 1000)
# # four examples of nonlinear functions
# f1 = x ** 2 - x - 1
# f2 = x ** 3 - 3 * np.sin(x)
# f3 = np.exp(x) - 2
# f4 = 1 - x ** 2 + np.sin(50 / (1 + x ** 2))
# # plot each function
# fig, axes = plt.subplots(1, 4, figsize=(12, 3), sharey=True)
# for n, f in enumerate([f1, f2, f3, f4]):
#     axes[n].plot(x, f, lw=1.5)
#     axes[n].axhline(0, ls=':', color='k')
#     axes[n].set_ylim(-5, 5)
#     axes[n].set_xticks([-2, -1, 0, 1, 2])
#     axes[n].set_xlabel(r'$x$', fontsize=18)
# axes[0].set_ylabel(r'$f(x)$', fontsize=18)
# titles = [r'$f(x)=x^2-x-1$', r'$f(x)=x^3-3sin(x)$',  # \left\right
#           r'$f(x)=\exp(x)-2$', r'$f(x)=\sin(50/(1+x^2))+1-x^2$']   # left和right用于控制格式，实际上似乎无作用
# for n, title in enumerate(titles):
#     axes[n].set_title(title)
# fig.tight_layout()
# fig.savefig('ch5-nonlinear-plot-equations.pdf')
# plt.show(block=True)
'''一般在图像下可以看到解的大致位置，随后使用二分法或者牛顿法进行求解'''
# define a function, desired tolerance and starting interval [a, b]
# f = lambda x: np.exp(x) - 2
# tol = 0.1
# a, b = -2, 2
# x = np.linspace(-2.1, 2.1, 1000)
# # graph the function f
# fig, ax = plt.subplots(1, 1, figsize=(12, 4))
# ax.plot(x, f(x), lw=1.5)
# ax.axhline(0, ls=':', color='k')  # matplotlib库的pyplot模块中的axhline ()函数用于在轴上添加一条水平线
# ax.set_xticks([-2, -1, 0, 1, 2])
# ax.set_xlabel(r'$x$', fontsize=18)
# ax.set_ylabel(r'$f(x)$', fontsize=18)
# # find the root using the bisection method and visualize
# # the steps in the method in the graph
# fa, fb = f(a), f(b)
# ax.plot(a, fa, 'ko')
# ax.plot(b, fb, 'ko')
# ax.text(a, fa + 0.5, r"$a$", ha='center', fontsize=18)
# ax.text(b, fb + 0.5, r"$b$", ha='center', fontsize=18)
# n = 1
# while b - a > tol:      # 循环迭代求解过程
#     m = a + (b - a) / 2
#     fm = f(m)
#     ax.plot(m, fm, 'ko')
#     ax.text(m, fm - 0.5, r"$m_%d$" % n, ha='center')
#     n += 1
#     if np.sign(fa) == np.sign(fm):   # 符号函数
#         a, fa = m, fm
#     else:
#         b, fb = m, fm
# ax.plot(m, fm, 'r*', markersize=10)
# ax.annotate("Root approximately at %.3f" % m,
#             fontsize=14, family="serif",
#             xy=(a, fm), xycoords='data',
#             xytext=(-150, +50), textcoords='offset points',
#             arrowprops=dict(arrowstyle="->", connectionstyle="arc3, rad=-.5"))
# ax.set_title("Bisection method")
# fig.tight_layout()
# fig.savefig('ch5-nonlinear-bisection.pdf')
# plt.show(block=True)
''' 牛顿法如下，收敛的速度高于二分法'''
# define a function, desired tolerance and starting point xk
# tol = 0.01
# xk = 2
# s_x = sympy.symbols("x")
# s_f = sympy.exp(s_x) - 2
# f = lambda x: sympy.lambdify(s_x, s_f, 'numpy')(x)
# fp = lambda x: sympy.lambdify(s_x, sympy.diff(s_f, s_x), 'numpy')(x)
# x = np.linspace(-1, 2.1, 1000)
# # setup a graph for visualizing the root finding steps
# fig, ax = plt.subplots(1, 1, figsize=(12,4))
# ax.plot(x, f(x))
# ax.axhline(0, ls=':', color='k')
# # repeat Newton's method until convergence to the desired tolerance has been reached
# n = 0
# while f(xk) > tol:
#     xk_new = xk - f(xk) / fp(xk)
#     ax.plot([xk, xk], [0, f(xk)], color='k', ls=':')
#     ax.plot(xk, f(xk), 'ko')
#     ax.text(xk, -.5, r'$x_%d$' % n, ha='center')
#     ax.plot([xk, xk_new], [f(xk), 0], 'k-')
#     xk = xk_new
#     n += 1
# ax.plot(xk, f(xk), 'r*', markersize=15)
# ax.annotate("Root approximately at %.3f" % xk,
#             fontsize=14, family="serif",
#             xy=(xk, f(xk)), xycoords='data',
#             xytext=(-150, +50), textcoords='offset points',
#             arrowprops=dict(arrowstyle="->", connectionstyle="arc3, rad=-.5"))
# ax.set_title("Newton's method")
# ax.set_xticks([-1, 0, 1, 2])
# fig.tight_layout()
# fig.savefig('ch5-nonlinear-newton.pdf')
# plt.show(block=True)
'''应当注意，牛顿方法需要进行导数的计算，一般必须使用数值方法而不能使用sympy的符号求解。
在一般使用之中，可以使用割线法（用线性函数近似当前函数的值）和高阶插值，'''
#  scipy的optimize模块提供了bisect，newton函数用于二分法和牛顿方法，3个输入依次为函数，自变量的上下限。
# a=optimize.bisect(lambda x: np.exp(x) - 2, -2, 2)
# print(a)
'''在newton的函数参数设定fprime=fprime，那么使用牛顿法，否则使用割线法'''
# 也有其他的函数，brentq()，这个是全能求根函数的首选，brenth(),ridder()函数，都是二分法的变体，同样适用于函数值改变的区间，
a1=optimize.brentq(lambda x: np.exp(x) - 2, -2, 2)
a2=optimize.brenth(lambda x: np.exp(x) - 2, -2, 2)
a3=optimize.ridder(lambda x: np.exp(x) - 2, -2, 2)
print(a1,a2,a3)
def f(x):
    return [x[1] - x[0] ** 3 - 2 * x[0] ** 2 + 1, x[1] + x[0] ** 2 - 1]
so1=optimize.fsolve(f, [1, 1])   # 函数与初始猜测值
print(so1)
def f_jacobian(x):
    return [[-3 * x[0] ** 2 - 4 * x[0], 1], [2 * x[0], 1]]
so2=optimize.fsolve(f, [1, 1], fprime=f_jacobian)    # 人为指定雅可比矩阵
print(so2)
x, y = sympy.symbols("x, y")
f_mat = sympy.Matrix([y - x ** 3 - 2 * x ** 2 + 1, y + x ** 2 - 1])
Joc1=f_mat.jacobian(sympy.Matrix([x, y]))
print(Joc1)   # 求解符号矩阵的雅可比阵
def f(x):
   return [x[1] - x[0]**3 - 2 * x[0]**2 + 1, x[1] + x[0]**2 - 1]
x = np.linspace(-3, 2, 5000)
y1 = x ** 3 + 2 * x ** 2 - 1
y2 = -x ** 2 + 1
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(x, y1, 'b', lw=1.5, label=r'$y = x^3 + 2x^2 - 1$')
ax.plot(x, y2, 'g', lw=1.5, label=r'$y = -x^2 + 1$')
x_guesses = [[-2, 2], [1, -1], [-2, -5]]
for x_guess in x_guesses:
    sol = optimize.fsolve(f, x_guess)
    ax.plot(sol[0], sol[1], 'r*', markersize=15)
    ax.plot(x_guess[0], x_guess[1], 'ko')
    ax.annotate("", xy=(sol[0], sol[1]), xytext=(x_guess[0], x_guess[1]),
                arrowprops=dict(arrowstyle="->", linewidth=2.5))
ax.legend(loc=0)
ax.set_xlabel(r'$x$', fontsize=18)
fig.tight_layout()
fig.savefig('ch5-nonlinear-system.pdf')
plt.show(block=True)

optimize.broyden2(f, x_guesses[1])
def f(x):
    return [x[1] - x[0] ** 3 - 2 * x[0] ** 2 + 1,
            x[1] + x[0] ** 2 - 1]
x = np.linspace(-3, 2, 5000)
y1 = x ** 3 + 2 * x ** 2 - 1
y2 = -x ** 2 + 1
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(x, y1, 'k', lw=1.5, label=r'$y = x^3 + 2x^2 - 1$')
ax.plot(x, y2, 'k', lw=1.5, label=r'$y = -x^2 + 1$')
sol1 = optimize.fsolve(f, [-2, 2])    # 不同初始点的求解
sol2 = optimize.fsolve(f, [1, -1])
sol3 = optimize.fsolve(f, [-2, -5])
sols = [sol1, sol2, sol3]
colors = ['r', 'b', 'g']
for idx, s in enumerate(sols):
    ax.plot(s[0], s[1], colors[idx] + '*', markersize=15)
for m in np.linspace(-4, 3, 80):
    for n in np.linspace(-15, 15, 40):
        x_guess = [m, n]
        sol = optimize.fsolve(f, x_guess)
        idx = (abs(sols - sol) ** 2).sum(axis=1).argmin()
        ax.plot(x_guess[0], x_guess[1], colors[idx] + '.')
ax.set_xlabel(r'$x$', fontsize=18)
ax.set_xlim(-4, 3)
ax.set_ylim(-15, 15)
fig.tight_layout()
fig.savefig('ch5-nonlinear-system-map.pdf')
plt.show(block=True)
def f(x):
    return [x[1] - x[0] ** 3 - 2 * x[0] ** 2 + 1,
            x[1] + x[0] ** 2 - 1]
x = np.linspace(-3, 2, 5000)
y1 = x ** 3 + 2 * x ** 2 - 1
y2 = -x ** 2 + 1
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(x, y1, 'k', lw=1.5, label=r'$y = x^3 + 2x^2 - 1$')
ax.plot(x, y2, 'k', lw=1.5, label=r'$y = -x^2 + 1$')
sol1 = optimize.fsolve(f, [-2, 2])
sol2 = optimize.fsolve(f, [1, -1])
sol3 = optimize.fsolve(f, [-2, -5])
for idx, s in enumerate([sol1, sol2, sol3]):    # 解的绘制
    ax.plot(s[0], s[1], colors[idx] + '*', markersize=15)
colors = ['r', 'b', 'g']
for m in np.linspace(-4, 3, 80):
    for n in np.linspace(-15, 15, 40):
        x_guess = [m, n]
        sol = optimize.fsolve(f, x_guess)      #网格式查看解与初始点的关系
        for idx, s in enumerate([sol1, sol2, sol3]):
            if abs(s - sol).max() < 1e-8:
                # ax.plot(sol[0], sol[1], colors[idx]+'*', markersize=15)
                ax.plot(x_guess[0], x_guess[1], colors[idx] + '.')
ax.set_xlabel(r'$x$', fontsize=18)
fig.tight_layout()
fig.savefig('ch5-nonlinear-system-map.pdf')
plt.show(block=True)







