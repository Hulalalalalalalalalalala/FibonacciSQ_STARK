from hashlib import sha256
from math import log2, ceil

from FibonacciSq_Trace import FieldElement


class MerkleTree(object):
    """
    简单实现Merkle树，用于生成和验证证明。
    """

    def __init__(self, data):
        """
        接受一个数据列表 data，构建一个 Merkle 树。首先，确保 data 是列表且不为空。然后，扩展 data
        以确保其长度是 2 的幂次方,用于创建完全二叉树。
        """
        assert isinstance(data, list)
        assert len(data) > 0, '无法创建空树。'

        """
        计算叶节点数
        向上取整。因为在构建完全二叉树时，如果 data 长度不是 2 的幂次方，需要增加额外的节点以保持树的完整性。
        如果 data 的长度是 5，log2(5) 约为 2.32，向上取整为 3， num_leaves 将是 2**3 = 8
        这意味着在构建 Merkle 树时，需要 8 个叶节点来保持树的完整性。
        """
        num_leaves = 2 ** ceil(log2(len(data)))

        # 数据扩展
        self.data = data + [FieldElement(0)] * (num_leaves - len(data))

        # 设置树高和根节点
        self.height = int(log2(num_leaves))
        # 初始化存储节点哈希值的字典
        self.facts = {}
        # 调用 build_tree 方法构建整个 Merkle 树，并设置根节点。
        self.root = self.build_tree()

    def get_authentication_path(self, leaf_id):
        """
        给定叶子节点的 ID，此方法返回从根节点到该叶子节点的路径（称为验证路径或解承诺路径）。
        这个路径对于验证某个数据元素是否在 Merkle 树中。
        在 Merkle 树中，每个叶子节点和内部节点都可以通过一个唯一的二进制路径从根节点访问到。
        这个路径可以通过考虑节点的索引来确定。在 Merkle 树中，通常将叶子节点的索引扩展为包含所有节点
        （叶子节点和内部节点）的索引。
        """
        # 确保传入的叶子节点 ID (leaf_id) 在有效范围内。
        assert 0 <= leaf_id < len(self.data)
        # 计算叶子节点在树中的 ID
        node_id = leaf_id + len(self.data)
        # 从根节点开始，沿着树向下移动，直到叶子节点。
        cur = self.root
        # 用于存储解承诺路径的列表
        decommitment = []
        # 遍历节点 ID 的二进制表示的每一位（除去最高位和 0b 前缀），其中每一位代表向左（'0'）或向右（'1'）的路径选择。
        """
        假设我们有一个简化的 Merkle 树，其结构和节点哈希值如下所示：
        根节点哈希：RootHash
        第一层节点哈希：NodeHash0（左），NodeHash1（右）
        叶子节点哈希：LeafHash0, LeafHash1, LeafHash2, LeafHash3
        我们要找到叶子节点 LeafHash2（索引为 2）的验证路径。

        计算 node_id：叶子节点的索引为 2，叶子节点总数为 4，所以 node_id = 2 + 4 = 6。二进制表示为 110。

        对 bin(node_id)[3:] 进行迭代：忽略最高位和 0b 前缀，只考虑 10。

        遍历二进制位：

        第一位是 1：向右走。记录当前节点（NodeHash0）的兄弟节点 NodeHash1。
        第二位是 0：向左走。当前节点更新为 NodeHash1 的左子节点（LeafHash2），记录其兄弟节点 LeafHash3。
        最终的验证路径是 [NodeHash1, LeafHash3]。这个路径可以用来验证 LeafHash2 是否真实存在于树中。
        """
        for bit in bin(node_id)[3:]:
            cur, auth = self.facts[cur]
            if bit == '1':
                auth, cur = cur, auth
            decommitment.append(auth)
        return decommitment

    def build_tree(self):
        return self.recursive_build_tree(1)

    def recursive_build_tree(self, node_id):
        """
        通过递归调用 recursive_build_tree 方法构建整个树。对于叶节点
        存储其哈希值；对于内部节点，存储由左右子节点哈希值合成的哈希值。
        """
        # 判断当前节点是否是叶子节点。如果节点 ID 大于等于叶子节点的数量，那么它是一个叶子节点。
        if node_id >= len(self.data):
            # 计算叶子节点在data列表中的索引。
            id_in_data = node_id - len(self.data)
            # 获取叶子节点的数据，并转换为字符串。
            leaf_data = str(self.data[id_in_data])
            # 计算叶子节点的哈希值，并将其存储在 facts 字典中。
            h = sha256(leaf_data.encode()).hexdigest()
            self.facts[h] = leaf_data
            # 返回叶子节点的散列值。
            return h
        else:
            # 如果当前节点不是叶子节点，即它是一个内部节点。
            # 递归调用 recursive_build_tree 方法构建左右子树。
            left = self.recursive_build_tree(node_id * 2)
            # 递归地构建右子节点。
            right = self.recursive_build_tree(node_id * 2 + 1)
            # 计算由左右子节点散列值拼接得到的字符串的 SHA256 散列值。
            h = sha256((left + right).encode()).hexdigest()
            # 将内部节点的散列值与其左右子节点的散列值关联存储在 self.facts 字典中。
            self.facts[h] = (left, right)
            # 返回内部节点的散列值。
            return h


def verify_decommitment(leaf_id, leaf_data, decommitment, root):
    """
    用于验证给定的叶子数据和解承诺路径是否与 Merkle 树的根节点一致，从而证明该叶子数据确实是树的一部分。
    """
    # 计算验证路径 decommitment 的长度对应的叶子节点总数（2 的幂次方）。
    leaf_num = 2 ** len(decommitment)
    # 计算在包含所有节点的树中的节点 ID
    node_id = leaf_id + leaf_num
    # 计算叶子节点的哈希值。
    cur = sha256(str(leaf_data).encode()).hexdigest()
    # 遍历节点 ID 的二进制表示（反转并去除前两位）和反转的验证路径。
    for bit, auth in zip(bin(node_id)[3:][::-1], decommitment[::-1]):
        # 如果当前节点 ID 的二进制表示的某一位是 1，那么向右走，否则向左走。
        if bit == '0':
            h = cur + auth
        else:
            h = auth + cur
        cur = sha256(h.encode()).hexdigest()
    # 如果最终得到的哈希值与根节点的哈希值相同，则验证成功。
    return cur == root
