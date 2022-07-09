# TODO Chapter1 科学环境介绍
# 注意事实上，Numpy库等也依赖于更底层的库，MKl，LAPACK，BLAS但是安装的时候会默认加装上，不用管理。
#  IDE Integrated Development Environment 集合成开发环境    REPL(read evaluate pront loop)交互式控制台  模式可以使用ctrl+z退出
#  spyder 用于python的软件包和模块开发
#  终端中使用 python --version 查看版本。 直接输入python可以看更多的信息
#  可以同时使用不同的虚拟环境，然后维护不同的python版本和自己特有的库。
#  ipython模式： 增强的命令行界面，并且底层使用用户--服务器结构，使得可以控制多个不同的用户与同一个服务器通信。使用Anaconda prompt的界面输入ipython进入。
#  可以使用In[3]之类的对之前的python控制台进行重新调用。使用输入的exit推出ipython模式
#  分号;屏蔽输出 。Tab自动补全即对象自省。？查看帮助help，？？ 更详细的帮助。e.g.math ?
#  !后面的内容都是自动理解为Shell w.g. !dir 用于查看目录和文件,
'''Shell是系统的用户界面，提供了用户与内核进行交互操作的一种接口.它接收用户输入的命令并把它送入内核去执行。中文名: 命令行
使用命令 %who 来查看当前的已有变量。
对于给定的一个函数例如x.py，在终端之中使用python x.py进行，或者在命令行中使用 % run x.py进行运行，
 %debug 调试模式，列出错误处的堆栈信息。
 %reset 重设环境
 %timeit和  %times是两个简单的性能测试工具，直接把代码附在文件后面，使用%prun查看细节信息。
 但是只能提供一个循环的总体时间，更复杂的工具分析可以使用代码分析器(Code profiler)
 使用conda inatall jupyter安装对应的版本信息，在激活的环境中使用jupyter qtconsole 激活一个类似于IDLE的命令行窗口，使用jupyter qtconsole 打开对应的网页编辑器。
IPython.display中提供了大量用于展示的方法。以及其他：HTML用于渲染HTML代码，Math类用于渲染Latex公式。'''

