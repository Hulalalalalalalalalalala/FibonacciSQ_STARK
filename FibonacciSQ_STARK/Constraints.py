"""
FibonacciSQ Constraints'
1. a[0] = 1
2. a[1022] = 2338775057
3. a[i+2] = a[i+1]^2+ a[i]^2
"""
from FibonacciSq_Trace import FieldElement
from Stark_polynomial import interpolate_poly, Polynomial
from channel import Channel
from merkle import MerkleTree


def part1():
    # 生成迹
    t = [FieldElement(1), FieldElement(3141592)]
    while len(t) < 1023:
        t.append(t[-2] * t[-2] + t[-1] * t[-1])
    # 得到生成元g
    g = FieldElement.generator() ** (3 * 2 ** 20)
    # 得到生成元g的幂，生成对应点x的序列
    points = [g ** i for i in range(1024)]
    # 得到生成元h_gen
    h_gen = FieldElement.generator() ** ((2 ** 30 * 3) // 8192)
    # 得到生成元h的幂，生成对应点x的序列
    h = [h_gen ** i for i in range(8192)]
    # g的幂与h的幂相乘，得到域的生成元
    domain = [FieldElement.generator() * x for x in h]
    # 插值得到p
    p = interpolate_poly(points[:-1], t)
    # 计算p在domain上的值
    ev = [p.eval(d) for d in domain]
    # 构建Merkle树
    mt = MerkleTree(ev)
    # 生成通道
    ch = Channel()
    # 发送根节点
    ch.send(mt.root)
    return t, g, points, h_gen, h, domain, p, ev, mt, ch


def part2():
    # 获取参数
    t, g, points, h_gen, h, domain, p, ev, mt, ch = part1()

    # 分子f-1
    numer0 = p - Polynomial([FieldElement(1)])
    # 分母x-1
    denom0 = Polynomial.gen_linear_term(FieldElement(1))
    # q0为商，r0为余数
    q0, r0 = numer0.qdiv(denom0)

    # 分子f-2338775057
    numer1 = p - Polynomial([FieldElement(2338775057)])
    # 分母x-g^1022
    denom1 = Polynomial.gen_linear_term(points[1022])
    # q1为商，r1为余数
    q1, r1 = numer1.qdiv(denom1)

    # f(g^2x)
    inner_poly0 = Polynomial([FieldElement(0), points[2]])
    final0 = p.compose(inner_poly0)
    # f(gx)^2
    inner_poly1 = Polynomial([FieldElement(0), points[1]])
    composition = p.compose(inner_poly1)
    final1 = composition * composition
    # f(x)^2
    final2 = p * p
    # 分子f(g^2x)-f(gx)^2-f(x)^2
    numer2 = final0 - final1 - final2
    # 分母中的x^1024 - 1
    coef = [FieldElement(1)] + [FieldElement(0)] * 1023 + [FieldElement(-1)]
    numerator_of_denom2 = Polynomial(coef)
    # 分母中的(x - g ^ 1022)(x - g ^ 1021)(x - g ^ 1023)
    factor0 = Polynomial.gen_linear_term(points[1021])
    factor1 = Polynomial.gen_linear_term(points[1022])
    factor2 = Polynomial.gen_linear_term(points[1023])
    denom_of_denom2 = factor0 * factor1 * factor2
    # 分母
    denom2, r_denom2 = numerator_of_denom2.qdiv(denom_of_denom2)
    # q2为商，r2为余数
    q2, r2 = numer2.qdiv(denom2)
    # cp0为q0的点乘
    cp0 = q0.scalar_mul(ch.receive_random_field_element())
    # cp1为q1的点乘
    cp1 = q1.scalar_mul(ch.receive_random_field_element())
    #  cp2为q2的点乘
    cp2 = q2.scalar_mul(ch.receive_random_field_element())
    # cp为cp0+cp1+cp2
    cp = cp0 + cp1 + cp2
    # cp_evaluations求cp在domain上的值
    cp_ev = [cp.eval(d) for d in domain]
    # cp_Merkle_Tree为cp_ev的Merkle树
    cp_mt = MerkleTree(cp_ev)
    # 发送根节点
    ch.send(cp_mt.root)
    return cp, cp_ev, cp_mt, ch, domain

"""
具体来说，next_fri_domain(domain) 函数做了以下几件事：

它接收当前的域 domain 作为输入。这个域是一个包含多项式的根的列表。

然后，它只取域列表的前半部分：domain[:len(domain) // 2]。这意味着如果域有 N 个元素，
只有前 N/2 个元素被使用。

对于所选的每个元素 x，它计算 x ** 2（x 的平方）。

最终，函数返回一个新的域列表，其中包含原始域前半部分元素的平方。
"""
def next_fri_domain(domain):
    return [x ** 2 for x in domain[:len(domain) // 2]]

"""
生成下一层多项式。其步骤如下：

从输入多项式 poly 中提取奇数和偶数位的系数。odd_coefficients = poly.poly[1::2] 
获取奇数位的系数（例如，系数列表中索引为 1, 3, 5, ... 的元素），
而 even_coefficients = poly.poly[::2] 获取偶数位的系数
（例如，系数列表中索引为 0, 2, 4, ... 的元素）。

创建两个新的多项式：一个由奇数位系数构成 (odd)，另一个由偶数位系数构成 (even)。

将奇数位系数的多项式与一个字段元素 alpha 相乘。这是通过调用 
odd = Polynomial(odd_coefficients).scalar_mul(alpha) 实现的。

最后，将这两个多项式相加，得到新的多项式 odd + even，并返回这个结果。
"""
def next_fri_polynomial(poly, alpha):
    odd_coefficients = poly.poly[1::2]
    even_coefficients = poly.poly[::2]
    odd = Polynomial(odd_coefficients).scalar_mul(alpha)
    even = Polynomial(even_coefficients)
    return odd + even

# 生成下一层的FRI层
def next_fri_layer(poly, dom, alpha):
    next_poly = next_fri_polynomial(poly, alpha)
    next_dom = next_fri_domain(dom)
    # 计算新多项式
    next_layer = [next_poly.eval(x) for x in next_dom]
    return next_poly, next_dom, next_layer


def part3():
    # 获取参数，cp（当前多项式）、cp_ev（多项式的点值）、cp_mt（对应的 Merkle 树）、ch（通信通道）和 domain（定义域）
    cp, cp_ev, cp_mt, ch, domain = part2()
    """
    fri_polys = [cp]: 初始化多项式列表，开始时只包含 cp。
    fri_doms = [domain]: 初始化定义域列表，开始时只包含 domain。
    fri_layers = [cp_ev]: 初始化点值层列表，开始时只包含 cp_ev。
    merkles = [cp_mt]: 初始化 Merkle 树列表，开始时只包含 cp_mt。
    """
    fri_polys = [cp]
    fri_doms = [domain]
    fri_layers = [cp_ev]
    merkles = [cp_mt]

    # 当最后一个多项式的度大于 0 时，重复以下步骤：
    while fri_polys[-1].degree() > 0:
        # 从通道中接收一个随机的字段元素 alpha。
        alpha = ch.receive_random_field_element()
        # 生成下一层的多项式、定义域和点值层。
        next_poly, next_dom, next_layer = next_fri_layer(fri_polys[-1], fri_doms[-1], alpha)
        # 将新的多项式添加到列表。
        fri_polys.append(next_poly)
        # 将新的定义域添加到列表。
        fri_doms.append(next_dom)
        # 将新的点值层添加到列表。
        fri_layers.append(next_layer)
        # 为新的点值层创建 Merkle 树，并添加到列表。
        merkles.append(MerkleTree(next_layer))
        # 发送最新 Merkle 树的根到通道。
        ch.send(merkles[-1].root)
    #  发送最后一个多项式的第一个系数（常数项）。
    ch.send(str(fri_polys[-1].poly[0]))
    # 返回所有 FRI 层的多项式、定义域、点值层、Merkle 树和通道。
    return fri_polys, fri_doms, fri_layers, merkles, ch
