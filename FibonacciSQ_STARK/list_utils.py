from itertools import dropwhile, starmap, zip_longest

# 从列表 list_of_elements 中移除尾部的连续元素，这些元素等于 element_to_remove。
def remove_trailing_elements(list_of_elements, element_to_remove):
    return list(dropwhile(lambda x: x == element_to_remove, list_of_elements[::-1]))[::-1]
"""
首先，使用 [::-1] 将列表反转。
接着，使用 dropwhile 函数。dropwhile 会从列表的开始（现在是反转后的开始）丢弃所有满足条件 lambda x: x == element_to_remove 的元素，
直到遇到第一个不满足条件的元素。最后，再次反转列表，得到原始顺序的列表，但已经移除了尾部的特定元素。
"""


# 对两个列表 f 和 g 中的元素执行操作 operation，并返回结果列表。如果两个列表长度不一致，fill_value 会被用作较短列表的填充值。
def two_lists_tuple_operation(f, g, operation, fill_value):
    return list(starmap(operation, zip_longest(f, g, fillvalue=fill_value)))
"""
zip_longest(f, g, fillvalue=fill_value)合并两个列表,如果 f 和 g 的长度不同，较短的列表会用 fill_value 填充至与较长列表相同的长度。
starmap(operation, zip_longest(f, g, fillvalue=fill_value)) 对 zip_longest(f, g, fillvalue=fill_value) 的结果执行 operation 操作。
"""

# 对列表 list_of_elements 中的每个元素执行操作 operation，并返回结果列表。
def scalar_operation(list_of_elements, operation, scalar):
    return [operation(c, scalar) for c in list_of_elements]
"""
函数遍历 list_of_elements 中的每个元素 c。
对于每个元素，它执行 operation(c, scalar)，其中 operation 是一个二元函数，c 是当前元素，scalar 是给定的标量值。
将所有的结果收集成一个新列表并返回。
"""