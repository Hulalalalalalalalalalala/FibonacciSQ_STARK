from channel import Channel
from Constraints import part1, part3
import time


"""
这个函数负责在 FRI（快速 Reed-Solomon 交互式证明）的每一层上进行解承诺。
对于 FRI 的每一层，它会发送特定索引处的值和对应的 Merkle 树验证路径。
通过这种方式，验证者可以验证证明者发送的值确实来自于原始数据
"""
def decommit_on_fri_layers(idx, channel):
    for layer, merkle in zip(fri_layers[:-1], fri_merkles[:-1]):
        length = len(layer)
        idx = idx % length
        sib_idx = (idx + length // 2) % length
        channel.send(str(layer[idx]))
        channel.send(str(merkle.get_authentication_path(idx)))
        channel.send(str(layer[sib_idx]))
        channel.send(str(merkle.get_authentication_path(sib_idx)))
    channel.send(str(fri_layers[-1][0]))


"""
这个函数处理针对 STARK 证明的特定查询。
它发送在给定索引处的函数 f 的求值结果以及其 Merkle 树验证路径。
它还会处理 f 的一些变换（如 f(gx) 和 f(g^2x)），这些变换在 STARK 证明的构建中非常重要。
最后，它调用 decommit_on_fri_layers 以处理 FRI 层的解承诺。
"""
def decommit_on_query(idx, channel):
    assert idx + 16 < len(f_eval), f'query index: {idx} is out of range. Length of layer: {len(f_eval)}.'
    channel.send(str(f_eval[idx]))
    channel.send(str(f_merkle.get_authentication_path(idx)))
    channel.send(str(f_eval[idx + 8]))
    channel.send(str(f_merkle.get_authentication_path(idx + 8)))
    channel.send(str(f_eval[idx + 16]))
    channel.send(str(f_merkle.get_authentication_path(idx + 16)))
    decommit_on_fri_layers(idx, channel)

"""
解承诺流程的入口点。
它从验证者处接收随机索引，并对这些索引执行解承诺。
通过这种方式，证明者向验证者展示他们所持有的信息是准确且可靠的，且他们没有试图欺骗系统。
"""
def decommit_fri(channel):
    for query in range(3):
        decommit_on_query(channel.receive_random_int(0, 8191-16), channel)


if __name__ == '__main__':
    try:
        start = time.time()
        start_all = start
        print("生成迹...")
        _, _, _, _, _, _, _, f_eval, f_merkle, _ = part1()
        print(f'迹生成完成。耗时: {time.time() - start}s')

        start = time.time()
        print("生成复合多项式和FRI...")
        fri_polys, fri_domains, fri_layers, fri_merkles, channel = part3()
        print(f'复合多项式和FRI生成完成。耗时: {time.time() - start}s')

        start = time.time()
        print("生成查询和验证承诺...")
        decommit_fri(channel)
        print(f'查询和验证承诺生成完成。耗时: {time.time() - start}s')

        print(f'总耗时: {time.time() - start_all}s')
        print(f'未压缩的校验长度（以字符为单位）: {len(str(channel.proof))}')

        # ___________校验____________

    except Exception as e:
        print(f'执行过程中遇到错误: {str(e)}')