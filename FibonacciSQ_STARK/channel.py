import inspect # for debugging
from hashlib import sha256

from FibonacciSq_Trace import FieldElement


def serialize(obj):
    """
    序列化为字符串
    """
    if isinstance(obj, (list, tuple)):
        return ','.join(map(serialize, obj))
    return obj._serialize_()


class Channel(object):
    """
    模拟一个交互式证明系统中的通信渠道，同时在内部实际上是非交互式的，并使用 SHA256 散列函数来生成所需的随机性。
    """

    def __init__(self):
        self.state = '0'
        self.proof = []

    def send(self, s):
        self.state = sha256((self.state + s).encode()).hexdigest()
        self.proof.append(f'{inspect.stack()[0][3]}:{s}')
        # inspect.stack()[0][3] 返回当前函数的名称

    def receive_random_int(self, min, max, show_in_proof=True):
        """
        模拟验证器发送的范围为[min，max]的随机整数。
        """
        num = min + (int(self.state, 16) % (max - min + 1))
        self.state = sha256((self.state).encode()).hexdigest()
        if show_in_proof:
            self.proof.append(f'{inspect.stack()[0][3]}:{num}')
        return num

    def receive_random_field_element(self):
        """
        模拟接收一个随机字段元素
        """
        num = self.receive_random_int(0, FieldElement.k_modulus - 1, show_in_proof=False)
        self.proof.append(f'{inspect.stack()[0][3]}:{num}')
        return FieldElement(num)
