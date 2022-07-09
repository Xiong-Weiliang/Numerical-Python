# TODO 稀疏矩阵和图
#  Warning!!! conda install --channel conda-forge pygraphviz 才可用于安装正确的pygrapgviz；绝不能使用默认的pip(conda) install pygrahpviz
#  否则会出现各种问题————reference website: https://pygraphviz.github.io/documentation/stable/install.html
import matplotlib.pyplot as plt
import matplotlib as mpl
# mpl.rcParams['text.usetex'] = True
# mpl.rcParams['mathtext.fontset'] = 'stix'
# mpl.rcParams['font.family'] = 'serif'
# mpl.rcParams['font.sans-serif'] = 'stix'
import scipy.sparse as sp
import scipy.sparse.linalg
import numpy as np
import time
import scipy.linalg as la
import networkx as nx
import json
import pygraphviz
# import scikit-umfpack   # 无需包含
'''Coordinate list format '''
# values = [1, 2, 3, 4]    # 'list' object is not callable 列表不可访问
# rows = [0, 1, 2, 3]
# cols = [1, 3, 2, 0]
# A = sp.coo_matrix((values, (rows, cols)), shape=[4, 4])  # coo_matrix进行稀疏矩阵的构建
## todense()和toarray()返回转换完毕的numpy 数组和普通的(稠密)矩阵
# print(A.todense())
## A.shape, A.size, A.dtype, A.ndim
# print(A.nnz, A.data ) # 非零元素的数目，非零值
## A.tocsr()   # 用于转化为CSR稀疏矩阵格式，类似的还有tocsc，这两种格式很适合数值运算
# '''并非所有的的稀疏矩阵都支持索引'''
## print(A[1, 2])     #'coo_matrix' object is not subscriptable
##  print(A.tobsr()[1, 2])
##  但是有一些支持索引
## A.tocsr()[1, 2]
## A.tolil()[1:3, 3]
# '''CSR格式细节，很适合数值计算，例如矩阵乘法'''
# A = np.array([[1, 2, 0, 0], [0, 3, 4, 0], [0, 0, 5, 6], [7, 0, 8, 9]]); A
# A = sp.csr_matrix(A)
# print(A.data,A.indices,A.indptr)     # 非零值，每行中非零的列的合成数组, 以及每一行非零开头位置.
# i = 2
# A.indptr[i], A.indptr[i+1]-1   # 可以进行索引操作
# A.indices[A.indptr[i]:A.indptr[i+1]]
# A.data[A.indptr[i]:A.indptr[i+1]]
# CSR和CSC行列对调 '''
''' 尽管可以先生成数据再使用构造，但是可以使用自带函数，
 sp.eye  diags  kron  bmat(稀疏块矩阵排列构成)  vstack  hstack '''
# 以上函数默认得到CSC但是可以用format指定格式. 稀疏矩阵的优势，在大型矩阵时可以表现出来.
# N=10
# A=-2 * sp.eye(N) + sp.eye(N, k=1) + sp.eye(N, k=-1)
# print(A.todense())
# fig, ax = plt.subplots()
# ax.spy(A)
# fig.savefig("ch10-sparse-matrix-1.pdf");
# A = sp.diags([1,-2,1], [1,0,-1], shape=[N, N], format='csc')  # 第一个矩阵稀疏为上面eye的稀疏，第二个是偏移形状，第三个参数为矩阵大下
# # 格式设定为csc.
# fig, ax = plt.subplots()
# ax.spy(A);
# print(A.todense())
# B = sp.diags([1, 1], [-1, 1], shape=[3,3])
# C = sp.kron(A, B, format='csr')      # 克罗内克积（Kronecker product）作用于矩阵（可视为二阶张量），
# print(B,C)
# fig, (ax_A, ax_B, ax_C) = plt.subplots(1, 3, figsize=(12, 4))
# ax_A.spy(A)   # spy() 是对稀疏矩阵进行可视化的方法
# ax_B.spy(B)
# ax_C.spy(C)
# fig.savefig("ch10-sparse-matrix-2.pdf");
# plt.show(block=True)
'''Sparse linear algebra'''
# # 在稀疏矩阵中，仅仅能返回最大和最小特征值对应的那些特征向量而不是全部，否则无法保持稀疏化, 另一方面，例如求逆等操作是无法保证稀疏结构的.
# N = 10
# A = sp.diags([1, -2, 1], [1, 0, -1], shape=[N, N], format='csc')
# b = -np.ones(N)
# x = sp.linalg.spsolve(A, b)       # 稀疏 linear equation 求解
# np.linalg.solve(A.todense(), b)   # 普通 linear equation 求解
# lu = sp.linalg.splu(A)            # LU decomposition of a sparse an square matrix. 也适用于求解 linear equation; 类似的还有spilu()进行不完全LU分解.
# print(lu.L,lu.perm_r,lu.U )       # Lu中会返回置换向量.
def sp_permute(A, perm_r, perm_c):    # 稀疏矩阵的置换操作，防止LU分解导致稀疏性降低.
    # 注意置换使得LU！=A, 恢复时候还需要除去置换矩阵.
    """ permute rows and columns of A """
    M, N = A.shape
    # row permumation matrix
    Pr = sp.coo_matrix((np.ones(M), (perm_r, np.arange(N)))).tocsr()  # perm_r和perm_c用于获取置换向量.
    # column permutation matrix
    Pc = sp.coo_matrix((np.ones(M), (np.arange(M), perm_c))).tocsr()
    return Pr.T * A * Pc.T       # 返回置换后的矩阵.
# print( sp_permute(lu.L * lu.U, lu.perm_r, lu.perm_c) - A)
# fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))
# ax1.spy(lu.L)
# ax2.spy(lu.U)
# ax3.spy(A)
# plt.show(block=True)    # 这个应该被放在最后买了否则会导致图像无法显示
# x = lu.solve(b)
# print(x)
'''(in which case UMFPACK is the default solver)另一个用于计算的库: use_umfpack=True is only effective if scikit-umfpack is installed'''
# x = sp.linalg.spsolve(A, b, use_umfpack=True)     #这里不需要显式调用库scikit-umfpack，只要指定参数就可以了，一般默认的和use_unfpack各有优劣，大矩阵都调试一下。
# print(x)
# """ 一般而言，迭代比直接求解会适用于大规模问题，几个常用的迭代求解方法: 双共轭梯度bicg(); 稳定双共轭bicgstab(); 共轭梯度法cg(); 广义最小残差方法 gmres();
# 松散广义最小残差方法lgmres()——这些方法返回解，以及额外信息info. """
# x, info = sp.linalg.cg(A, b)
# # atol argument is a recent addition
# x, info = sp.linalg.lgmres(A, b, atol=1e-5) # atol收敛公差
# print(x,info)   # info=0表示求解成功, 否则为正值.
'''compare performance of solving Ax=b vs system size N, where A is the sparse matrix for the 1d poisson problem'''
# def setup(N):
#     A = sp.diags([1, -2, 1], [1, 0, -1], shape=[N, N], format='csr')
#     b = -np.ones(N)
#     return A, A.todense(), b
# reps = 10
# N_vec = np.arange(2, 300, 1)
# t_sparse = np.empty(len(N_vec))
# t_dense = np.empty(len(N_vec))
# for idx, N in enumerate(N_vec):
#     A, A_dense, b = setup(N)
#     t = time.time()     # 计时时间开始
#     for r in range(reps):
#         x = np.linalg.solve(A_dense, b)  #普通求解方法
#     t_dense[idx] = (time.time() - t) / reps   # 此时的计算时间和之前的求差.
#     t = time.time()
#     for r in range(reps):
#         x = sp.linalg.spsolve(A, b, use_umfpack=True)  # 稀疏求解方法
#     t_sparse[idx] = (time.time() - t) / reps
# fig, ax = plt.subplots(figsize=(8, 4))
# ax.plot(N_vec, t_dense * 1e3, '.-', label="dense")
# ax.plot(N_vec, t_sparse * 1e3, '.-', label="sparse")
# ax.set_xlabel(r"$N$", fontsize=16)
# ax.set_ylabel("elapsed time (ms)", fontsize=16)
# ax.legend(loc=0)
# fig.tight_layout()
# plt.show(block=True)
'''特征值问题'''
# sp.linalg.eigs和sp.linalg.svds可以用于计算稀疏矩阵特征值；对于实Heritian阵, 也可以使用sp.linalg.eigsh计算, 稀疏阵不会返回所有的特征值和向量，默认6个，参数可调.
# 在eigs中可以设定要求： which参数：LM，SM为最大最小模，LR，SR最大最小实部，LI，SI最大最小虚部.
# N=10  # 矩阵记录方法的一个例子:Reverse Cuthil McKee  (目的:用于重新排列以减小带宽)
# A = sp.diags([1, -2, 1], [8, 0, -8], shape=[N, N], format='csc')
# perm = sp.csgraph.reverse_cuthill_mckee(A)   # 返回按Reverse-Cuthill McKee顺序排列稀疏CSR或CSC矩阵的排列数组。
# print(perm)
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
# ax1.spy(A)
# ax2.spy(sp_permute(A, perm, perm))
# plt.show(block=True)
''' 特征值问题求解 '''
# N = 10
# A = sp.diags([1, -2, 1], [1, 0, -1], shape=[N, N], format='csc')
# evals, evecs = sp.linalg.eigs(A, k=4, which='LM')   # 求解4个最大幅值的特征值.
# print(evals)
# np.allclose(A.dot(evecs[:,0]), evals[0] * evecs[:,0])  # allclose:  Returns True if two arrays are element-wise equal within a tolerance.
# evals,evecs = sp.linalg.eigsh(A, k=4, which='LM')     # 对称时候使用.
# evals,evecs = sp.linalg.eigs(A, k=4, which='SR')
# print(np.real(evals).argsort())   # Returns the indices that would sort this array.
#
# def sp_eigs_sorted(A, k=6, which='SR'):    # compute and return eigenvalues sorted by real value
#     evals, evecs = sp.linalg.eigs(A, k=k, which=which)
#     idx = np.real(evals).argsort()    # 返回从大到小的指标，重排列
#     return evals[idx], evecs[idx]   #idx=3 2 1 0
# evals, evecs = sp_eigs_sorted(A, k=4, which='SM')
# print(evals)
# '''  Random matrix example'''
# N = 100
# x_vec = np.linspace(0, 1, 50)
# # seed sp.rand with random_state to obtain a reproducible result
# M1 = sp.rand(N, N, density=0.2, random_state=112312321)
# # M1 = M1 + M1.conj().T
# M2 = sp.rand(N, N, density=0.2, random_state=984592134)
# # M2 = M2 + M2.conj().T
# evals = np.array([sp_eigs_sorted((1-x)*M1 + x*M2, k=25)[0] for x in x_vec])
# fig, ax = plt.subplots(figsize=(8, 4))
# for idx in range(evals.shape[1]):
#     ax.plot(x_vec, np.real(evals[:,idx]), lw=0.5)  # 绘制实数部分.
# ax.set_xlabel(r"$x$", fontsize=16)
# ax.set_ylabel(r"eig.vals. of $(1-x)M_1+xM_2$", fontsize=16)
# fig.tight_layout()
# plt.show(block=True)
''' 图和连接矩阵 scipy.sparse.scgraph提供了处理稀疏图的连接矩阵的方法. 专业处理库NetworkX(import networkx as nx)用于处理函数库'''
# nx.Graph 无向图；  nx.DiGraph 无向图；  nx.Multi(Di)Graph 多边无(有)向图； add_node和add_node_from用于增加节点(从列表中)，edge类似.
# 返回的所有的节点的迭代器对象NodeView, 或者边的迭代器对象EdgeView. 有权重的边add_weighted_edges_from通过元组传入，包括节点和其元组。
# g = nx.MultiGraph()
# g.add_node(1)
# g.add_nodes_from([3, 4, 5])
# print(g.nodes())
# g.add_edge(1, 2)
# g.add_edges_from([(3, 4), (5, 6)])
# print(g.edges())
# g.add_weighted_edges_from([(1, 3, 1.5), (3, 5, 2.5)])
# g.edges(data=True)
# g.add_weighted_edges_from([(6, 7, 1.5)])
# print(g.nodes())
# print(g.edges())
with open("tokyo-metro.json") as f:   # 打开文件加载.
    data = json.load(f)
print(data.keys() )
data["C"]
# data
g = nx.Graph()
for line in data.values():
    g.add_weighted_edges_from(line["travel_times"])
    g.add_edges_from(line["transfers"])
for n1, n2 in g.edges():
    g[n1][n2]["transfer"] = "weight" not in g[n1][n2]   #
print(g.number_of_nodes() )
list(g.nodes())[:5]
print(g.number_of_edges())
# list(g.edges())[:5]
on_foot = [edge for edge in g.edges() if g.get_edge_data(*edge)["transfer"]]
on_train = [edge for edge in g.edges() if not g.get_edge_data(*edge)["transfer"]]
colors = [data[n[0].upper()]["color"] for n in g.nodes()]
# from networkx.drawing.nx_agraph import graphviz_layout
fig, ax = plt.subplots(1, 1, figsize=(14, 10))
pos = nx.drawing.nx_agraph.graphviz_layout(g, prog="neato")   # 这一行有问题，graphviz包出现问题，替代直接使用draw函数
nx.draw(g, pos, ax=ax, node_size=200, node_color=colors)    # 绘制节点
nx.draw_networkx_labels(g, pos=pos, ax=ax, font_size=6)     # 绘制节点的标记
nx.draw_networkx_edges(g, pos=pos, ax=ax, edgelist=on_train, width=2)
nx.draw_networkx_edges(g, pos=pos, ax=ax, edgelist=on_foot, edge_color="blue")
pos=nx.draw(g, pos=nx.spring_layout(g))
###     removing the default axis on all sides:
for side in ['bottom','right','top','left']:
    ax.spines[side].set_visible(False)
# removing the axis labels and ticks
ax.set_xticks([])
ax.set_yticks([])
ax.xaxis.set_ticks_position('none')
ax.yaxis.set_ticks_position('none')
fig.savefig("ch10-metro-graph.pdf")
fig.savefig("ch10-metro-graph.png")
fig.tight_layout()
plt.show(block=True)

# g.degree()
# d_max = max(d for (n, d) in g.degree())
# [(n, d) for (n, d) in g.degree() if d == d_max]
# p = nx.shortest_path(g, "Y24", "C19")
# np.array(p)
# np.sum([g[p[n]][p[n+1]]["weight"] for n in range(len(p)-1) if "weight" in g[p[n]][p[n+1]]])
# h = g.copy()
# for n1, n2 in h.edges():
#     if "transfer" in h[n1][n2]:
#         h[n1][n2]["weight"] = 5
# p = nx.shortest_path(h, "Y24", "C19")
# np.array(p)
# np.sum([h[p[n]][p[n+1]]["weight"] for n in range(len(p)-1)])
# p = nx.shortest_path(h, "Z1", "H16")
# np.sum([h[p[n]][p[n+1]]["weight"] for n in range(len(p)-1)])
# A = nx.to_scipy_sparse_matrix(g)
# print(A)
# perm = sp.csgraph.reverse_cuthill_mckee(A)
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
# ax1.spy(A, markersize=2)
# ax2.spy(sp_permute(A, perm, perm), markersize=2)
# fig.tight_layout()
# plt.show(block=True)



