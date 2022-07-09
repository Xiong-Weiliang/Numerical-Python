#  TODO 符号计算
""" 使用Sympy库 进行符号计算，主要关注于解析计算方法，注意建议将整个库导入，而非from sympy import * 以防止命名冲突 """
import sympy
sympy.init_printing()
from sympy import I,pi,oo     # 注意导入的是符号表示而非数值表示,与numpy之中的pi数值表示区分开。因此调用的时候必须指明其库
# 注意虚数单位是大写的I
# x=sympy.Symbol("x",real=True)     #将x绑定到名称相同的Python变量。
# print(x.is_real)    #返回True
# y=sympy.Symbol('y') #单双引号都可以
# print(y.is_real)    # 返回None,表示符号未知
# ''' 除了real之外，还可以使用其余的关键字参数，类似与加入假设条件以限定符号的范围。其属性都是is_关键词 '''
# # real,imaginary,positive, negative,integer,odd,even,prime(素数),finite ,infinite
# print(sympy.sqrt(x**2)) # 这里复制到Python的控制台，等价于abs(x),|x|相当于对符号进行了简化
# # 同时定义大量符号使用symbols
# a,b,c=sympy.symbols('a,b,c',negative=True)
# # 应当注意在Sympy中的数值变量和Numpy的数值类型不同，如果想在‘表达式树’中使用，必须将其转化为sympy中的类型。
# i=sympy.Integer(5)
# print(type(i))       # <class 'sympy.core.numbers.Integer'>
# ii=sympy.Float(2.545)
# print(type(ii))     # <class 'sympy.core.numbers.Float'>
# i=int(i)   # 转化回来原来的int值
# print(type(i))   # <class 'int'>
# # 类型中is_integer和 is_Integer 不同，后者更加严格，'''一般而言，小写表示类型，大写表示是某满足某个已知类型的值'''
# i=sympy.Integer(i)    # 这里是 True
# print(i.is_Integer)   #由于指定了具体的数值5，所以这里是True
# iii=sympy.Symbol('iii',integer=True)
# print(iii.is_Integer)       # 这里是False,没有指定具体的值时会判错。
# # sympy中没有精度限制，可以表示极其庞大的数值
# ii=sympy.Float(25555)
# print(ii**10)
# print(sympy.factorial(100))  #返回 100! 的值
'''浮点类型的使用，sympy中的浮点类型同样不会限制精度'''
# print(sympy.Float(0.3,45))   #以45位精度打印0.3（误差来源于0.3为python float 0.3，存在误差 ）
# print(sympy.Float('0.2',20))     # 以字符串打印而没有误差，后面全是0值。
# # 一个等价的表述为
# print("%.25f" % 0.3)       # 以25位小数打印出0.3，存在误差。
# # 有理数的使用如下：
# r1=sympy.Rational(11,13)
# r2=sympy.Rational(1,3)
# print(r1*r2)       # 打印出来  11/39
'''几个常用的数学符号如下，pi（圆周率）,E（自然对数）,EulerGamma（欧拉常数）,I（虚数）,oo(正无穷)'''
# 应当注意到，实际中可以调用的函数和symbol.Function的实例之间具有较大的差距。
# x, y, z = sympy.symbols("x, y, z")
# f = sympy.Function("f")            #给出未定义函数，只是指出了抽象映射关系
# print(type(f))   #<class 'sympy.core.function.UndefinedFunction'>
# g = sympy.Function("g")(x, y, z)   # 定义一个和三个变量相关的函数
# print(g.free_symbols)
# n = sympy.Symbol("n", integer=True)
# print(sympy.sin(pi * n))     #非常罕见的情况下可以解出,注意
# h = sympy.Lambda(x, x**2)
# print(h(1+x))
# 表达式 and 表达树
# x = sympy.Symbol("x")
# e = 1 + 2 * x**2 + 3 * x**3
# print(e)
# print(e.args)
# print(e.args[1].args[1].args[0])   # 访问节点，一般地从低阶的独立常数开始，如果x和常数绑定在一起，x在前面，下标从0开始
# print(e.args[0])   # 1
# print(e.args[1].args[1].args[0].args)  # () 表明是空的
'''sympy之中的表达式子被视作不可变对象，因此例如如下的化简操作会新建一个数组，而不是直接操作'''
x,y,z= sympy.symbols("x,y,z")
# expr = 2 * (x**2 - x) - x * (x + 1)
# xx=sympy.simplify(expr)  # 等价调用方法expr.simplify()
# print(xx)
# expr = 2 * sympy.cos(x) * sympy.sin(x)
# print(expr)
# expr2=sympy.trigsimp(expr)    # 三角恒等式化简
# print(expr2)
# expr3 = sympy.exp(x) * sympy.exp(y)
# expr4=sympy.powsimp(expr3)    # 幂花间
# print(expr4)         # 其余的化简常数还有compsimp（化简组合表达式），ratsimp（公分母化简）
# 展开函数expand()，可以在simplify黑箱化简不能给出令人满意的结果时。
# expr = (x + 1) * (x + 2)
# expr1=sympy.expand(expr)
# print(expr1)
# expr2=sympy.sin(x + y).expand(trig=True)  #三角形式展开
# print(expr2)
# # 其余的参数
'''
log=True  用于对数展开
complex=true 用于对实数部分和虚数部分展开
power_based和power_exp分别用于展开幂的低和指数
'''
a, b = sympy.symbols("a, b",)
# a1=sympy.log(a * b).expand()
# print(a1)
# a2=sympy.exp(I*a + b).expand(complex=True)
# print(a2)
# a3=sympy.expand((a * b)**x, power_exp=True)
# print(a3)
# a4=sympy.exp(I*(a-b)*x).expand(power_exp=True)
# print(a4)
# # 因式分解函数factor()函数。对于展开的其余类型，可以使用trigsimp,powsimp,logconbine进行反向操作
# print(sympy.factor(x**2 - 1)) #因式分解
# print(sympy.factor(x * sympy.cos(y) + sympy.sin(z) * x))
# x=sympy.logcombine(sympy.log(x) - sympy.log(y))
# print(x)
# expr = x + y + x * y * z   # collect 用于给定的因子进行合并同类项
# print(expr.factor())
# print(expr.collect(x))
# print(expr.collect(y))
# expr = sympy.cos(x + y) + sympy.sin(x - y)
# a=expr.expand(trig=True).collect([sympy.cos(x), sympy.sin(x)]).collect(sympy.cos(y) - sympy.sin(y))
# print(a)
# a1=sympy.apart(1/(x**2 + 3*x + 2), x)  # 分式拆分
# a2=sympy.together(1 / (y * x + y) + 1 / (1+x))  # 合并
# a3=sympy.cancel(y / (y * x + y))    # 消去公因子
# print(a1,'\n',a2,'\n',a3)  #可以同时打印多个。然后插入换行
# # 符号替换： subs 和replace（更加复杂，例如可以使用通配符进行替换）
# b1=(x + y).subs(x, y)     # 用有替换x
# b2=sympy.sin(x * sympy.exp(x)).subs(x, y)
# b3=sympy.sin(x * z).subs({z: sympy.exp(y), x: y, sympy.sin: sympy.cos})  #同时替换多个的方法。
# expr = x * y + z**2 *x
# values = {x: 1.25, y: 0.4, z: 3.2}   # 赋值带入表达式中进行计算值
# b4=expr.subs(values)   # 计算值
# print(b1,'\n',b2,'\n',b3,'\n',b4,'\n')
 # 数值求值：使用sympy.N 或者evalf方法对表达式进行求值
# a=sympy.N(1 + pi)
# b=sympy.N(pi, 50)
# c=(x + 1/pi).evalf(7)
# expr = sympy.sin(pi * x * sympy.exp(x))
# print([expr.subs(x, xx).evalf(3) for xx in range(0, 10)])    #将具体值赋值给xx，然后用xx代替x
# # 对一系列数值进行操作时候，更好的方法是使用lambdify函数，批量化操作————通过输入一个表达式和自由符号，返回一个函数，
# expr_func = sympy.lambdify(x, expr) #应当注意，lambdify使用的是严格的数值计算，不可以将符号变量给它
# d=expr_func(1.0)
# expr_func = sympy.lambdify(x, expr, 'numpy')   # 将'numpy'作为第三个参数，就可以接受其数组类型。
# print(a,'\n',b,'\n',c,'\n',d)
# import numpy as np
# xvalues = np.arange(0, 10)   #定义一个numpy的类型的数组
# print(expr_func(xvalues)) # 将值带入表达式中进行计算
# 以下考虑微积分操作
# f = sympy.Function('f')(x)   # 定义函数f
# a1=sympy.diff(f, x)
# a2=sympy.diff(f, x, x)
# a3=sympy.diff(f, x, 3)
# print(a1,'\n',a2,'\n',a3,'\n')  #输出一阶导数到三阶导数
# g = sympy.Function('g')(x, y)
# g1=g.diff(x, y)
# g2=g.diff(x, 3, y, 2)         # equivalent to s.diff(g, x, x, x, y, y)
# expr = x**4 + x**3 + x**2 + x + 1
# e1= expr.diff(x)
# e2= expr.diff(x, x)
# expr = (x + 1)**3 * y ** 2 * (z - 1)
# e3= expr.diff(x, y, z)
# expr = sympy.sin(x * y) * sympy.cos(x / 2)
# e4=expr.diff(x)
# print(e1,'\n',e2,'\n',e3,'\n',e4)  #输出一阶导数到三阶导数
# # 以下位特殊的多项式————可能是版本迭代的关系，现在不需要对其显式调用了： .special.polynomials.
# expr = sympy.hermite(x, 0)
# e5= expr.diff(x).doit()
# print(e5)
# d = sympy.Derivative(sympy.exp(sympy.cos(x)), x)   #定义实例，第一输入是表达式，第二元素是需要求导的符号
# print(d)
# print(d.doit())    #延迟计算模式。可以完全实时的控制什么时候对其进行计算。
# integral函数用于计算积分，而Integral函数用于表示积分类似于Derivative，然后用doit()函数进行调用。
# a,b = sympy.symbols("a, b")
# x,y = sympy.symbols('x, y')
# f = sympy.Function('f')(x)
# f1=sympy.integrate(f)
# f2=sympy.integrate(f, (x, a, b))
# f3=sympy.integrate(sympy.sin(x))
# f4=sympy.integrate(sympy.sin(x), (x, a, b))  # 在给定区间上的定积分
# f5=sympy.integrate(sympy.exp(-x**2), (x, 0, oo))  # 从0到无穷的定积分
# print(f1,'\n',f2,'\n',f3,'\n',f4,'\n',f5,'\n')  #输出一阶导数到三阶导数
# '''  一般的积分很难计算, 无法计算时会返回一个Intergrate实例  '''
# a, b, c = sympy.symbols("a, b, c", positive=True)
# g1=sympy.integrate(a * sympy.exp(-((x-b)/c)**2), (x, -oo, oo))
# g2=sympy.integrate(sympy.sin(x * sympy.cos(x)))    # 此处会输出一个Intergrate实例
# expr = sympy.sin(x*sympy.exp(y))
# g3=sympy.integrate(expr, x)
# print(g1,'\n',g2,'\n',g3,'\n')
# # 多元变量，定积分与不定积分
# expr = (x + y)**2
# g4=sympy.integrate(expr, x)  # 仅仅对x进行积分
# g5=sympy.integrate(expr, x, y)  #二元不定积分
# g6=sympy.integrate(expr, (x, 0, 1), (y, 0, 1))    #二元定积分，指定积分区间
# print(g4,'\n',g5,'\n',g6,'\n')
# # 级数展开函数series，其默认展开阶数为6，默认为原点展开
# x = sympy.Symbol("x")
# f = sympy.Function("f")(x)
# print(sympy.series(f, x))    #输出级数
# x0 = sympy.Symbol("{x_0}")
# f.series(x, x0, n=2)   # 二阶展开式子
# f.series(x, x0, n=2).removeO()   #remove0()函数用于移除误差项
# '''对于特定的函数也可以得到'''
# s1=sympy.cos(x).series()
# s2=sympy.sin(x).series()
# s3=sympy.exp(x).series()
# s4=(1/(1+x)).series()
# print(s1,'\n',s2,'\n',s3,'\n',s4,'\n')
# # 二次级数展开加expand()如下：
# expr = sympy.cos(x) / (1 + sympy.sin(x * y))
# q1=expr.series(x, n=4)
# q2=expr.series(y, n=4)
# q3=expr.series(y).removeO().series(x).removeO()
# q4=q3.expand() #expand()对函数进行各种展开
# print(q1,'\n',q2,'\n',q3,'\n',q4,'\n')
# # 极限函数sympy.limit()
# sympy.limit(sympy.sin(x) / x, x, 0)  # 第一个参数为表达式，后续为自变量与极限点的设定
# f = sympy.Function('f')
# x, h = sympy.symbols("x, h")
# diff_limit = (f(x + h) - f(x))/h
# d1=sympy.limit(diff_limit.subs(f, sympy.cos), h, 0)    # 替代函数，将f换为cos
# d2=sympy.limit(diff_limit.subs(f, sympy.sin), h, 0)
# print(d1,'\n',d2,'\n')
# expr = (x**2 - 3*x) / (2*x - 2)
# p = sympy.limit(expr/x, x, oo)     #用于计算其渐近行为
# q = sympy.limit(expr - p*x, x, oo)
# print(p, q)
# '''使用函数sum与product求解区间内的和函数值'''
# n = sympy.symbols("n", integer=True)
# x = sympy.Sum(1/(n**2), (n, 1, oo))
# print(x)
# print(x.doit())    # 具体求值
# x = sympy.Product(n, (n, 1, 7))
# print(x)
# x.doit()
# print(x.doit())
# x = sympy.Symbol("x")
# '''注意下面这个方法叠加使用，显然无穷项叠加是用解析方法计算的，而不是数值方法计算的，因此对于可以积分的函数，sympy可以计算很多这样的函数
# 求解指数函数展开的级数，然后求和，然后求值，然后化简'''
# x1=sympy.Sum((x)**n/(sympy.factorial(n)), (n, 1, oo)).doit().simplify()
# print(x1)
# x = sympy.symbols("x")
# sympy.solve(x**2 + 2*x - 3)
# a, b, c = sympy.symbols("a, b, c")
# x_value=sympy.solve(a * x**2 + b * x + c, x)   # 注意是默认右边为0值的方程
# print(x_value)
# x1=sympy.solve(sympy.sin(x) - sympy.cos(x), x)
# x2=sympy.solve(sympy.exp(x) + 2 * x, x)
# x3=sympy.solve(x**5 - x**2 + 1, x)   # 返回一个形式的解，如果有需要再返回其数值解（即进一步计算）
# print(x1,'\n',x2,'\n',x3)
#sympy.solve(sympy.tan(x) - x, x)  # 无解的方程直接报错:NotImplementedError
#以下为求解方程组
# eq1 = x + 2 * y - 1
# eq2 = x - y + 1
# s=sympy.solve([eq1, eq2], [x, y], dict=True)    #字典形式返回每一个解
# print(s)  #打印求解的函数  # 总体返回一个list，其中任意一个元素为定义的dict类型，表示一个解
# eq1 = x**2 - y
# eq2 = y**2 - x
# sols = sympy.solve([eq1, eq2], [x, y], dict=True)
# print(sols)
# # 解的验证如下
# [eq1.subs(sol).simplify() == 0 and eq2.subs(sol).simplify() == 0 for sol in sols] # 返回值为： [True, True, True, True]
# ## Linear algebra
# sympy.Matrix([1,2])
# sympy.Matrix([[1,2]])
# sympy.Matrix([[1, 2], [3, 4]]) # 注意，sympy中的数组只可以使用二元数组
# sympy.Matrix(3, 4, lambda m,n: 10 * m + n)   # 输入行数和列数，然后直接按照lambda函数生成矩阵
# a, b, c, d = sympy.symbols("a, b, c, d")
# M = sympy.Matrix([[a, b], [c, d]])
# print(M)
# print(M * M)
# x = sympy.Matrix(sympy.symbols("x_1, x_2"))    #  元素可以是符号表达式
# print(M * x)       # 符号矩阵之间的矩阵计算（矩阵乘法）
# p, q = sympy.symbols("p, q")
# M = sympy.Matrix([[1, p], [q, 1]])
# b = sympy.Matrix(sympy.symbols("b_1, b_2"))
# x1 = M.solve(b)     #x1~x3三种线性方程组的求解方法，但是inv求逆比LU分解更加困难，因此不建议使用。
# x2 = M.LUsolve(b)
# x3 = M.inv() * b
# print(x1,'\n',x2,'\n',x3)
# '''一些其余的常见函数操作方法
# adjoint 伴随矩阵
# LU/QRdecomposition 用于求解LU和QR分解
# nullspace  计算矩阵的零空间
# norm 矩阵范数
# singular_values  矩阵奇异值
# diagonalize  对角化矩阵
# '''
