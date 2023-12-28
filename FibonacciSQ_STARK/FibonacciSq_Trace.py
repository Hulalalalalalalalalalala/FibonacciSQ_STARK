from random import randint
from gmpy2 import mpz, is_prime, invert

"""
FieldElement 类代表了有限域（finite field）中的元素。有限域是一种代数结构，其中包
含了有限数量的元素，并定义了加法、减法、乘法和除法运算。这个类的关键特点包括：
    初始化：通过给定的整数值 val 和模 k_modulus 初始化一个有限域元素。所有的运算都是在模
           k_modulus 下进行的。
    加法和乘法：实现了有限域上的加法 (__add__) 和乘法 (__mul__)。
    求逆：提供了求有限域元素逆 (inverse) 的方法，这对于实现除法很重要。
    幂运算：允许对有限域元素进行幂运算 (__pow__)。
    特殊元素：静态方法 zero 和 one 分别用于获取域的零元素和单位元素。
    类型转换：typecast 方法用于确保运算中的对象都是 FieldElement 类型。
"""

def find_prime(n):
    """返回第一个大于 n 的素数"""
    while not is_prime(n):
        n += 1
    return n

def prime_factors(n):
    """返回 n 的所有不同素数因子"""
    factors = set()
    i = 2
    while i * i <= n:
        if n % i:
            i += 1
        else:
            n //= i
            factors.add(i)
    if n > 1:
        factors.add(n)
    return factors

def find_generator(modulus):
    if not is_prime(modulus):
        raise ValueError("模数必须是一个素数。")

    phi = modulus - 1
    factors = prime_factors(phi)

    for g in range(2, modulus):
        if all(pow(mpz(g), mpz(phi // f), mpz(modulus)) != 1 for f in factors):
            return FieldElement(g)

    raise ValueError("没有找到生成元")

class FieldElement:

    # def __init__(self, val, mod=None):
    #     if mod is None:
    #         mod = FieldElement.k_modulus
    #     # 如果 val 是 FieldElement 类型，取出其值
    #     if isinstance(val, FieldElement):
    #         val = val.val
    #     self.val = val % mod

    k_modulus = 3 * 2 ** 30 + 1
    generator_val = 5

    def __init__(self, val):
        self.val = val % FieldElement.k_modulus

    @staticmethod
    def zero():
        """
        获取域的零元素。
        """
        return FieldElement(0)

    @staticmethod
    def one():
        """
        获取域的单位元素。
        """
        return FieldElement(1)

    def __repr__(self):
        # 在元素的正值和负值之间选择较短的表示形式。
        return repr((self.val + self.k_modulus//2) % self.k_modulus - self.k_modulus//2)

    def __eq__(self, other):
        if isinstance(other, int):
            other = FieldElement(other)
        return isinstance(other, FieldElement) and self.val == other.val

    def __hash__(self):
        return hash(self.val)

    @staticmethod
    def generator():
        return FieldElement(FieldElement.generator_val)

    @staticmethod
    def typecast(other):
        if isinstance(other, int):
            return FieldElement(other)
        assert isinstance(other, FieldElement), f'类型不匹配: FieldElement 和 {type(other)}。'
        return other

    def __neg__(self):
        return self.zero() - self

    def __add__(self, other):
        try:
            other = FieldElement.typecast(other)
        except AssertionError:
            return NotImplemented
        return FieldElement((self.val + other.val) % FieldElement.k_modulus)

    __radd__ = __add__

    def __sub__(self, other):
        try:
            other = FieldElement.typecast(other)
        except AssertionError:
            return NotImplemented
        return FieldElement((self.val - other.val) % FieldElement.k_modulus)

    def __rsub__(self, other):
        return -(self - other)

    def __mul__(self, other):
        try:
            other = FieldElement.typecast(other)
        except AssertionError:
            return NotImplemented
        return FieldElement((self.val * other.val) % FieldElement.k_modulus)

    __rmul__ = __mul__

    def __truediv__(self, other):
        other = FieldElement.typecast(other)
        return self * other.inverse()

    def __pow__(self, n):
        assert n >= 0
        cur_pow = self
        res = FieldElement(1)
        while n > 0:
            if n % 2 != 0:
                res *= cur_pow
            n = n // 2
            cur_pow *= cur_pow
        return res


    # 计算有限域元素的逆,扩展欧几里得算法（EEA）
    def inverse(self):
        t, new_t = 0, 1
        r, new_r = FieldElement.k_modulus, self.val
        while new_r != 0:
            quotient = r // new_r
            t, new_t = new_t, (t - (quotient * new_t))
            r, new_r = new_r, r - quotient * new_r
        assert r == 1
        #在循环结束时，r 应该等于 1（因为我们要找的是乘法逆元，所以最大公约数应该是 1）。如果不是，则算法失败，抛出断言错误。
        return FieldElement(t)

    def is_order(self, n):
        assert n >= 1
        h = FieldElement(1)
        for _ in range(1, n):
            h *= self
            if h == FieldElement(1):
                return False
        return h * self == FieldElement(1)

    # 序列化
    def _serialize_(self):
        return repr(self.val)

    @staticmethod
    def random_element(exclude_elements=[]):
        fe = FieldElement(randint(0, FieldElement.k_modulus - 1))
        while fe in exclude_elements:
            fe = FieldElement(randint(0, FieldElement.k_modulus - 1))
        return fe


