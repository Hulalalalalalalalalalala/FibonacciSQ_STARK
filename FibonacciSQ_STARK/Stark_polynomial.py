import operator
import matplotlib
from FibonacciSq_Trace import FieldElement
from list_utils import remove_trailing_elements, scalar_operation, two_lists_tuple_operation
import numpy as np
matplotlib.use('TkAgg')  # 使用TkAgg后端
import matplotlib.pyplot as plt

def simple_iter(iterable):
    """
    一个简单的迭代器，不显示进度条，只返回输入的迭代器本身。
    """
    return iterable
try:
    import progressbar
except ModuleNotFoundError:
    progressbar = simple_iter

def trim_trailing_zeros(p):
    """
    从列表末尾删除零。
    """
    return remove_trailing_elements(p, FieldElement.zero())

def prod(values):
    """
    递归计算乘积。
    """
    len_values = len(values)
    if len_values == 0:
        return 1
    if len_values == 1:
        return values[0]
    return prod(values[:len_values // 2]) * prod(values[len_values // 2:])

class Polynomial:
    """
    表示FieldElement上的多项式
    """

    @classmethod
    def X(cls):
        """
        返回多项式x
        """
        return cls([FieldElement.zero(), FieldElement.one()])

    # 初始化多项式对象。
    def __init__(self, coefficients, var='x'):
        # var 是多项式的变量名，默认为 'x'。
        # coefficients 是一个 FieldElement 对象列表，表示多项式的系数，列表中的位置对应于系数的幂次。
        self.poly = remove_trailing_elements(coefficients, FieldElement.zero())
        """
        在多项式的表示中，列表的每个元素代表多项式的系数，列表中的位置代表对应的幂次。例如，在多项式3x^2+0x+0中，
        系数列表是 [0, 0, 3]。在这种表示法中，列表末尾的零元素（即高幂次的零系数）在数学上是不必要的，因为它们对
        多项式的值没有贡献。
        """
        self.var = var

    # 多项式的字符串表示形式，便于打印和显示。
    def __repr__(self):
        """
        返回多项式的字符串表示形式。
        """
        if not self.poly:
            return '0'
        res = []
        for exponent, coef in enumerate(self.poly):
            if coef == 0:
                continue
            if exponent == 0:
                monomial = str(coef.val)
            elif exponent == 1:
                monomial = f'{coef.val}{self.var}'
            else:
                monomial = f'{coef.val}{self.var}^{exponent}'
            if monomial.startswith('-'):
                res.append(f' - {monomial[1:]}')
            else:
                if res:
                    res.append(f' + {monomial}')
                else:
                    res.append(monomial)
        return ''.join(res)

    # --------------绘图-----------------
    def evaluate(self, x):
        """以给定的x值计算多项式。"""
        result = 0
        for exponent, coef in enumerate(self.poly):
            result += coef.val * (x ** exponent)
        return result

    def plot(self, start=-10, end=10, num_points=1000):
        """使用matplotlib绘制多项式。"""
        x = np.linspace(start, end, num_points)
        y = np.array([self.evaluate(xi) for xi in x])

        plt.figure(figsize=(10, 6))
        plt.plot(x, y, label=str(self))
        plt.title("Polynomial Plot")
        plt.xlabel("x")
        plt.ylabel("Polynomial Value")
        plt.legend()
        plt.grid(True)
        plt.show()
    # --------------结束-----------------

    def __eq__(self, other):
        """
        用于检查两个多项式是否相等。
        other 是与当前多项式进行比较的对象。
        如果 other 不是多项式类型，或者与当前多项式不相等，返回 False，否则返回 True。
        """
        try:
            other = Polynomial.typecast(other)
        except AssertionError:
            return False
        return self.poly == other.poly

    @staticmethod
    def typecast(other):
        """
        用于将 FieldElement 或 int 转换为多项式对象。
        如果 other 是 int 类型，将其转换为 FieldElement 对象，然后再将其转换为多项式。
        如果 other 已经是多项式类型，则直接返回。
        如果 other 既不是 FieldElement 也不是多项式类型，抛出 AssertionError。
        """
        if isinstance(other, int):
            other = FieldElement(other)
        if isinstance(other, FieldElement):
            other = Polynomial([other])
        assert isinstance(other, Polynomial), f'类型错误: Polynomial and {type(other)}.'
        return other

    def __add__(self, other):
        """
        用于多项式的加法运算。
        other 是要与当前多项式相加的多项式对象。
        将当前多项式的系数和 other 多项式的系数逐项相加，得到新的多项式对象，并返回。
        """
        other = Polynomial.typecast(other)
        return Polynomial(two_lists_tuple_operation(self.poly, other.poly, operator.add, FieldElement.zero()))

    __radd__ = __add__

    def __sub__(self, other):
        """
        用于多项式的减法运算。
        other 是要从当前多项式中减去的多项式对象。
        将当前多项式的系数和 other 多项式的系数逐项相减，得到新的多项式对象，并返回
        """
        other = Polynomial.typecast(other)
        return Polynomial(two_lists_tuple_operation(self.poly, other.poly, operator.sub, FieldElement.zero()))

    def __rsub__(self, other):
        return -(self - other)

    def __neg__(self):
        """
        用于多项式的取负运算。
        返回当前多项式的负数（每个系数取负值）。
        """
        return Polynomial([]) - self

    def __mul__(self, other):
        """
        用于多项式的乘法运算。
        other 是要与当前多项式相乘的多项式对象。
        计算两个多项式的乘积，得到新的多项式对象，并返回。
        """
        other = Polynomial.typecast(other)

        # 提取两个多项式的系数
        pol1_coefficients = [x.val for x in self.poly]
        pol2_coefficients = [x.val for x in other.poly]

        # 初始化结果数组
        result_length = self.degree() + other.degree() + 1
        res = [0] * result_length

        # 对于每个多项式的系数进行乘法运算
        for i in range(len(pol1_coefficients)):
            for j in range(len(pol2_coefficients)):
                res[i + j] += pol1_coefficients[i] * pol2_coefficients[j]
        """
        乘积 pol1_coefficients[i] * pol2_coefficients[j] 是两个多项式中对应项的乘积
        其在结果多项式中的位置是 i + j。这是因为当你将两个多项式项相乘时，相乘项的幂次相加
        """

        # 将结果转换回 FieldElement 类型
        res = [FieldElement(x) for x in res]

        return Polynomial(res)

    __rmul__ = __mul__

    # 多项式的复合
    def compose(self, other):
        other = Polynomial.typecast(other)
        res = Polynomial([])
        for coef in self.poly[::-1]:
            res = (res * other) + Polynomial([coef])
        return res

    # 多项式除法,计算两个多项式相除的商和余数。长除法（polynomial long division）的算法
    def qdiv(self, other):
        other = Polynomial.typecast(other)
        pol2 = trim_trailing_zeros(other.poly)
        assert pol2, '除以零多项式。'
        pol1 = trim_trailing_zeros(self.poly)
        if not pol1:
            return [], []
        # 初始化
        rem = pol1
        deg_dif = len(rem) - len(pol2)
        quotient = [FieldElement.zero()] * (deg_dif + 1)
        # 获取被除多项式最高次项的逆
        g_msc_inv = pol2[-1].inverse()
        # 执行长除法
        while deg_dif >= 0:
            tmp = rem[-1] * g_msc_inv
            quotient[deg_dif] = quotient[deg_dif] + tmp
            last_non_zero = deg_dif - 1
            for i, coef in enumerate(pol2, deg_dif):
                rem[i] = rem[i] - (tmp * coef)
                if rem[i] != FieldElement.zero():
                    last_non_zero = i
            # 消除后零（即使r以其最后一个非零系数结束）。
            rem = rem[:last_non_zero + 1]
            deg_dif = len(rem) - len(pol2)
        # 返回商和余数
        return Polynomial(trim_trailing_zeros(quotient)), Polynomial(rem)

    # 定义除法，调用多项式除法
    def __truediv__(self, other):
        div, mod = self.qdiv(other)
        assert mod == 0, '多项式是不可整除的'
        return div

    # 定义取模，调用多项式除法
    def __mod__(self, other):
        return self.qdiv(other)[1]

    @staticmethod
    # 构造单项式
    def monomial(degree, coefficient):
        """
        构造单项系数*x**度。
        """
        return Polynomial([FieldElement.zero()] * degree + [coefficient])

    @staticmethod
    def gen_linear_term(point):
        """
        为给定点p生成多项式（x-p）。用于后续生成多项式整除问题的多项式。
        """
        return Polynomial([FieldElement.zero() - point, FieldElement.one()])
        # 表示：−p+x 或 x−p

    def degree(self):
        """
        多项式由列表表示，因此次数是列表的长度减去
        尾部零的数量（如果存在）减1。
        """
        return len(trim_trailing_zeros(self.poly)) - 1

    def get_nth_degree_coefficient(self, n):
        """
        它返回多项式中 x**n 的系数。
        """
        if n > self.degree():
            return FieldElement.zero()
        else:
            return self.poly[n]
            # 列表的索引对应于幂次，所以返回self.poly[n]是x**n的系数。

    def scalar_mul(self, scalar):
        """
        用于多项式的标量乘法。
        """
        return Polynomial(scalar_operation(self.poly, operator.mul, scalar))

    def eval(self, point):
        """
        使用Horner评估在给定点得到多项式的值。
        """
        point = FieldElement.typecast(point).val
        val = 0
        # 逆序遍历多项式的系数。从多项式的最高次项开始计算。
        for coef in self.poly[::-1]:
            val = (val * point + coef.val) % FieldElement.k_modulus
        return FieldElement(val)

    def __call__(self, other):
        """
        如果“other”是int或FieldElement，则计算多项式的值。

        如果“other”是多项式，则将self与“other’合成self（other（x））。
        """
        if isinstance(other, (int)):
            other = FieldElement(other)
        if isinstance(other, FieldElement):
            return self.eval(other)
        if isinstance(other, Polynomial):
            return self.compose(other)
        raise NotImplementedError()

    def __pow__(self, other):
        assert other >= 0
        res = Polynomial([FieldElement(1)])
        cur = self
        while True:
            if other % 2 != 0:
                res *= cur
            other >>= 1
            if other == 0:
                break
            cur = cur * cur
        return res


X = Polynomial.X()


def calculate_lagrange_polynomials(x_values):
    """
    评估某些多项式的x_值，计算拉格朗日多项式的一部分
    需要在该域上插值多项式。
    """
    lagrange_polynomials = []
    # 创建了一个表示(x-x_i)的多项式。
    monomials = [Polynomial.monomial(1, FieldElement.one()) - Polynomial.monomial(0, x) for x in x_values]
    # (x-x_0)(x-x_1)...(x-x_{len(X)-1})
    numerator = prod(monomials)

    # 创建自定义的 progressbar
    bar = progressbar.ProgressBar(max_value=len(x_values), widgets=[
        '拉格朗日基本多项式计算中: ', progressbar.Bar(), ' ', progressbar.Percentage(),
        ' ', progressbar.Timer(), ' | ETA: ', progressbar.ETA()
    ])

    with bar as progress:
        for j in range(len(x_values)):
            # 计算分母，即(x_j-x_0)(x_j-x_1)...(x_j-x_{len(X)-1})
            denominator = prod([x_values[j] - x for i, x in enumerate(x_values) if i != j])
            # 计算分子并进行除法，得到拉格朗日多项式，在这个除法过程中，(x-x_j) 这一项在 numerator 中被消除，因为除以自己得到 1。
            cur_poly, _ = numerator.qdiv(monomials[j].scalar_mul(denominator))
            # 将计算得到的多项式添加到列表中
            lagrange_polynomials.append(cur_poly)
            # 更新进度条
            progress.update(j)
    return lagrange_polynomials


def interpolate_poly_lagrange(y_values, lagrange_polynomials):
    """
    :param y_values：点的y坐标。

    :param lagrange_polynomials：从calculate_lagrange_polynamils获得的多项式。

    :return：插值多项式
    """
    # 初始化插值多项式
    poly = Polynomial([])
    # 为每个y值添加一个拉格朗日多项式
    for j, y_value in enumerate(y_values):
        # 累加多项式
        poly += lagrange_polynomials[j].scalar_mul(y_value)
    return poly


def interpolate_poly(x_values, y_values):
    """
    返回次数<len（x_values）的多项式，该多项式在x_values[i]上的求值结果为y_values[i]for所有i。
    """
    assert len(x_values) == len(y_values)
    assert all(isinstance(val, FieldElement) for val in x_values), '并非所有的x_值都是FieldElement'
    lp = calculate_lagrange_polynomials(x_values)
    assert all(isinstance(val, FieldElement) for val in y_values), '并非所有y_值都是FieldElement'
    return interpolate_poly_lagrange(y_values, lp)
