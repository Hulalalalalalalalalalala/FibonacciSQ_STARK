from FibonacciSQ_STARK.FibonacciSq_Trace import FieldElement
from FibonacciSQ_STARK.Stark_polynomial import Polynomial, interpolate_poly


if __name__ == '__main__':

    a = [FieldElement(1), FieldElement(3141592)]
    for i in range(1021):
        a.append(a[-2] * a[-2] + a[-1] * a[-1])
    print("a的长度为：", len(a))
    # 断言判断
    assert len(a) == 1023, '迹必须包含确切的 1023 个元素。'
    assert a[0] == FieldElement(1), '迹中的第一个元素必须是单位元素。'
    for i in range(2, 1023):
        assert a[i] == a[i - 1] * a[i - 1] + a[i - 2] * a[i - 2], f'FibonacciSq 递归规则在索引 {i} 处不适用'
    assert a[1022] == FieldElement(2338775057), '最后一个元素不正确！'

    print(a)
    g = FieldElement.generator()
    print(g)
    # 用G的元素填充G，使得G[i]：=G**i
    G = []
    g = FieldElement.generator() ** (3 * 2 ** 20)
    G = [g ** i for i in range(1024)]

    from hashlib import sha256

    # 创建有限域的生成元w
    w = FieldElement.generator()

    # 计算生成元h，用于创建大小为8192的群H
    h = w ** ((2 ** 30 * 3) // 8192)

    # 生成群H，包含从h^0到h^8191的所有元素
    H = [h ** i for i in range(8192)]

    # 通过将群H的每个元素与w相乘来创建陪集eval_domain
    eval_domain = [w * x for x in H]

    # 确保eval_domain中的所有元素都是唯一的，没有重复
    assert len(set(eval_domain)) == len(eval_domain)

    # 重新创建有限域的生成元w
    w = FieldElement.generator()

    # 计算w的逆元素w_inv
    w_inv = w.inverse()

    # 验证群H中的第一个元素H[1]是否正确
    assert '55fe9505f35b6d77660537f6541d441ec1bd919d03901210384c6aa1da2682ce' == sha256(str(H[1]).encode()).hexdigest(), \
        'H list is incorrect. H[1] should be h (i.e., the generator of H).'

    # 验证陪集的正确性，确保eval_domain中的每个元素都可以通过从eval_domain[1]开始，
    # 连续乘以w的逆元素和w本身来得到
    for i in range(8192):
        assert ((w_inv * eval_domain[1]) ** i) * w == eval_domain[i]

    # 打印成功信息
    print('Success!')

    """
    制作测试题（多项式插值）：

    首先，你给学生一些点，比如 (1, 3), (2, 5), (3, 7)。学生的任务是找到一个多项式，比如 f(x) = 2x + 1，这个多项式在这些点上的值与给出的值相匹配。
    在代码中，这对应于 interpolate_poly(G[:-1], a)，这里 G[:-1] 和 a 就像是你给出的点和它们的值。
    做题（在陪集上评估多项式）：

    然后，你让学生使用他们找到的多项式 f(x) 来计算一些新点的值，比如 x = 4, 5, 6。
    在代码中，这相当于遍历 eval_domain，使用 f(d) 计算这些新点上的多项式值。eval_domain 就像是新的点集。
    检查答案（哈希验证）：
    
    为了快速检查答案，你有一个答案的摘要（比如一个特定的数字或代码）。学生通过计算他们答案的摘要，来看它是否与你给的摘要相匹配。
    在代码中，sha256(serialize(f_eval).encode()).hexdigest() 这部分就是在做这个。它检查学生计算出的新点的值（多项式评估的结果）是否正确。
    如果答案正确，表明成功（打印成功信息）：
    
    如果学生的答案摘要与你的摘要匹配，那么他们的答案是正确的。你就会告诉他们他们做对了。
    在代码中，如果哈希值匹配，print('Success!') 这行就会执行，表明整个过程成功完成。
    """
    f = interpolate_poly(G[:-1], a)
    f_eval = [f(d) for d in eval_domain]

    from hashlib import sha256
    from channel import serialize

    assert '1d357f674c27194715d1440f6a166e30855550cb8cb8efeb72827f6a1bf9b5bb' == sha256(
        serialize(f_eval).encode()).hexdigest()
    print('Success!')

    from merkle import MerkleTree

    f_merkle = MerkleTree(f_eval)
    assert f_merkle.root == '6c266a104eeaceae93c14ad799ce595ec8c2764359d7ad1b4b7c57a4da52be04'
    print('Success!')

    from channel import Channel

    channel = Channel()
    channel.send(f_merkle.root)

    print(channel.proof)
