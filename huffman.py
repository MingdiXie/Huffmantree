"""
Code for compressing and decompressing using Huffman compression.
"""

from nodes import HuffmanNode, ReadNode


# ====================
# Helper functions for manipulating bytes


def get_bit(byte, bit_num):
    """ Return bit number bit_num from right in byte.

    @param int byte: a given byte
    @param int bit_num: a specific bit number within the byte
    @rtype: int

    >>> get_bit(0b00000101, 2)
    1
    >>> get_bit(0b00000101, 1)
    0
    """
    return (byte & (1 << bit_num)) >> bit_num


def byte_to_bits(byte):
    """ Return the representation of a byte as a string of bits.

    @param int byte: a given byte
    @rtype: str

    >>> byte_to_bits(14)
    '00001110'
    """
    return "".join([str(get_bit(byte, bit_num))
                    for bit_num in range(7, -1, -1)])


def bits_to_byte(bits):
    """ Return int represented by bits, padded on right.

    @param str bits: a string representation of some bits
    @rtype: int

    >>> bits_to_byte("00000101")
    5
    >>> bits_to_byte("101") == 0b10100000
    True
    """
    return sum([int(bits[pos]) << (7 - pos)
                for pos in range(len(bits))])


# ====================
# Functions for compression


def make_freq_dict(text):
    """ Return a dictionary that maps each byte in text to its frequency.

    @param bytes text: a bytes object
    @rtype: dict(int,int)

    >>> d = make_freq_dict(bytes([65, 66, 67, 66]))
    >>> d == {65: 1, 66: 2, 67: 1}
    True
    """
    result = {}
    for i in text:
        if i not in result.keys():
            result.setdefault(i, text.count(i))
    return result
    # todo


def huffman_tree(freq_dict):
    """ Return the root HuffmanNode of a Huffman tree corresponding
    to frequency dictionary freq_dict.

    @param dict(int,int) freq_dict: a frequency dictionary
    @rtype: HuffmanNode

    >>> freq = {2: 6, 3: 4}
    >>> t = huffman_tree(freq)
    >>> result1 = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> result2 = HuffmanNode(None, HuffmanNode(2), HuffmanNode(3))
    >>> t == result1 or t == result2
    True
    """
    if not freq_dict:
        return HuffmanNode(None,None,None)
    reverse_dict = {}
    for key,value in freq_dict.items():
        reverse_dict[value] = reverse_dict.get(value, []) + [HuffmanNode(key)]
    index_list = sorted(list(reverse_dict.keys()))
    
    if len(index_list) == 1 and len(reverse_dict[index_list[0]]) == 1:
        return HuffmanNode(None, None, reverse_dict[index_list[0]][0])
    
    while len(index_list) > 1 or len(reverse_dict[index_list[0]]) > 1:
        helper_build_tree(reverse_dict, index_list)
    
    return reverse_dict[index_list[0]][0]
        



def helper_build_tree(reverse_dict, index_list):
    '''build tree one step'''
    
    first = index_list[0]
    if len(reverse_dict[first]) >= 2:
        second = first
        f_byte = reverse_dict[first].pop(0)
        s_byte = reverse_dict[first].pop(0)
        if len(reverse_dict[first]) == 0:
            reverse_dict.pop(first)
            index_list.pop(0)
    else:
        f_byte = reverse_dict.pop(first)[0]
        index_list.pop(0)
        second = index_list[0]
        if len(reverse_dict[second]) >= 2:
            s_byte = reverse_dict[second].pop(0)
        else:
            s_byte = reverse_dict.pop(second)[0]
            index_list.pop(0)
    
    
    new_key = first + second
    new_byte = HuffmanNode(None, f_byte, s_byte)
    if reverse_dict.get(new_key, -1) == -1:
        reverse_dict[new_key] = [new_byte]
        index_list.append(new_key)
        index_list.sort()
    else:
        reverse_dict[new_key].append(new_byte)
        index_list.sort()
        
    
def get_codes(tree):
    """ Return a dict mapping symbols from Huffman tree to codes.

    @param HuffmanNode tree: a Huffman tree rooted at node 'tree'
    @rtype: dict(int,str)

    >>> tree = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> d = get_codes(tree)
    >>> d == {3: "0", 2: "1"}
    True
    """
    result_dict = {}
    s =''
    helper_get_code(tree, s, result_dict)
    return result_dict

def helper_get_code(tree, s, result_dict):
    '''a nice helper'''
    if not tree:
        return None
    if tree.is_leaf():
        result_dict[tree.symbol] = s
        return 
    helper_get_code(tree.left, s +'0', result_dict)
    helper_get_code(tree.right, s+'1', result_dict)
    




def number_nodes(tree):
    """ Number internal nodes in tree according to postorder traversal;
    start numbering at 0.

    @param HuffmanNode tree:  a Huffman tree rooted at node 'tree'
    @rtype: NoneType

    >>> left = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> right = HuffmanNode(None, HuffmanNode(9), HuffmanNode(10))
    >>> tree = HuffmanNode(None, left, right)
    >>> number_nodes(tree)
    >>> tree.left.number
    0
    >>> tree.right.number
    1
    >>> tree.number
    2
    """
    l = [0]
    helper_number_nodes(tree, l)
    
def helper_number_nodes(tree, l):
    '''a magic function'''
    if tree.is_leaf():
        return None
    helper_number_nodes(tree.left, l)
    helper_number_nodes(tree.right, l)
    tree.number = l[0]
    l[0] += 1


def avg_length(tree, freq_dict):
    """ Return the number of bits per symbol required to compress text
    made of the symbols and frequencies in freq_dict, using the Huffman tree.

    @param HuffmanNode tree: a Huffman tree rooted at node 'tree'
    @param dict(int,int) freq_dict: frequency dictionary
    @rtype: float

    >>> freq = {3: 2, 2: 7, 9: 1}
    >>> left = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> right = HuffmanNode(9)
    >>> tree = HuffmanNode(None, left, right)
    >>> avg_length(tree, freq)
    1.9
    """
    dic = get_codes(tree)
    result = 0
    total = 0
    for key, value in dic.items():
        result += len(value)*freq_dict[key]
        total += freq_dict[key]
    return result / total
    # todo


def generate_compressed(text, codes):
    """ Return compressed form of text, using mapping in codes for each symbol.

    @param bytes text: a bytes object
    @param dict(int,str) codes: mapping from symbols to codes
    @rtype: bytes

    >>> d = {0: "0", 1: "10", 2: "11"}
    >>> text = bytes([1, 2, 1, 0])
    >>> result = generate_compressed(text, d)
    >>> [byte_to_bits(byte) for byte in result]
    ['10111000']
    >>> text = bytes([1, 2, 1, 0, 2])
    >>> result = generate_compressed(text, d)
    >>> [byte_to_bits(byte) for byte in result]
    ['10111001', '10000000']
    """
    s_result = []
    l_result = []
    l = 0
    for i in text:
        l += len(codes[i])
        s_result.append(codes[i])
        while l >= 8:
            s = ''.join(s_result)
            l_result.append(bits_to_byte(s[0:8]))
            s_result = [s[8:]]
            l -= 8
    if s_result:        
        s = ''.join(s_result)
        l_result.append(bits_to_byte(s))              
    return bytes(l_result)


def tree_to_bytes(tree):
    """ Return a bytes representation of the Huffman tree rooted at tree.

    @param HuffmanNode tree: a Huffman tree rooted at node 'tree'
    @rtype: bytes

    The representation should be based on the postorder traversal of tree
    internal nodes, starting from 0.
    Precondition: tree has its nodes numbered.

    >>> tree = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> number_nodes(tree)
    >>> list(tree_to_bytes(tree))
    [0, 3, 0, 2]
    >>> left = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> right = HuffmanNode(5)
    >>> tree = HuffmanNode(None, left, right)
    >>> number_nodes(tree)
    >>> list()
    [0, 3, 0, 2, 1, 0, 0, 5]
    """
    return bytes(wrapper_tree_to_bytes(tree))

def wrapper_tree_to_bytes(tree):
    if tree.is_leaf():
        return []
    return (wrapper_tree_to_bytes(tree.left) + wrapper_tree_to_bytes(tree.right)\
            + helper_tree_to_byte(tree))

def helper_tree_to_byte(tree):
    '''a great helper'''
    res = []
    if tree.left.is_leaf():
        res.append(0)
        res.append(tree.left.symbol)
    else:
        res.append(1)
        res.append(tree.left.number)
    if tree.right.is_leaf():
        res.append(0)
        res.append(tree.right.symbol)
    else:
        res.append(1)
        res.append(tree.right.number)
    return res


def num_nodes_to_bytes(tree):
    """ Return number of nodes required to represent tree (the root of a
    numbered Huffman tree).

    @param HuffmanNode tree: a Huffman tree rooted at node 'tree'
    @rtype: bytes
    """
    return bytes([tree.number + 1])


def size_to_bytes(size):
    """ Return the size as a bytes object.

    @param int size: a 32-bit integer to convert to bytes
    @rtype: bytes

    >>> list(size_to_bytes(300))
    [44, 1, 0, 0]
    """
    # little-endian representation of 32-bit (4-byte)
    # int size
    return size.to_bytes(4, "little")


def compress(in_file, out_file):
    """ Compress contents of in_file and store results in out_file.

    @param str in_file: input file to compress
    @param str out_file: output file to store compressed result
    @rtype: NoneType
    """
    with open(in_file, "rb") as f1:
        text = f1.read()
    freq = make_freq_dict(text)
    tree = huffman_tree(freq)
    codes = get_codes(tree)
    number_nodes(tree)
    print("Bits per symbol:", avg_length(tree, freq))
    result = (num_nodes_to_bytes(tree) + tree_to_bytes(tree) +
              size_to_bytes(len(text)))
    result += generate_compressed(text, codes)
    with open(out_file, "wb") as f2:
        f2.write(result)


# ====================
# Functions for decompression


def generate_tree_general(node_lst, root_index):
    """ Return the root of the Huffman tree corresponding
    to node_lst[root_index].

    The function assumes nothing about the order of the nodes in node_lst.

    @param list[ReadNode] node_lst: a list of ReadNode objects
    @param int root_index: index in 'node_lst'
    @rtype: HuffmanNode

    >>> lst = [ReadNode(0, 5, 0, 7), ReadNode(0, 10, 0, 12), \
    ReadNode(1, 1, 1, 0)]
    >>> generate_tree_general(lst, 2)
    HuffmanNode(None, HuffmanNode(None, HuffmanNode(10, None, None), \
HuffmanNode(12, None, None)), \
HuffmanNode(None, HuffmanNode(5, None, None), HuffmanNode(7, None, None)))
    """
    tree_lst = []
    for i in node_lst:
        tree_node = HuffmanNode(None, None, None)
        if i.l_type == 0:
            tree_node.left = HuffmanNode(i.l_data, None, None)
        if i.r_type == 0: 
            tree_node.right = HuffmanNode(i.r_data, None, None)
        tree_lst.append(tree_node)
    
    root = tree_lst[root_index]
    for i in range(len(tree_lst)):
        if node_lst[i].l_type == 1:
            index = node_lst[i].l_data
            tree_lst[i].left = tree_lst[index]
        if node_lst[i].r_type == 1:
            index = node_lst[i].r_data
            tree_lst[i].right = tree_lst[index] 
    return root
    # todo


def generate_tree_postorder(node_lst, root_index):
    """ Return the root of the Huffman tree corresponding
    to node_lst[root_index].

    The function assumes that node_lst represents a tree in postorder.

    @param list[ReadNode] node_lst: a list of ReadNode objects
    @param int root_index: index in 'node_lst'
    @rtype: HuffmanNode

    >>> lst = [ReadNode(0, 5, 0, 7), ReadNode(0, 10, 0, 12), \
    ReadNode(1, 0, 1, 0)]
    >>> generate_tree_postorder(lst, 2)
    HuffmanNode(None, HuffmanNode(None, HuffmanNode(5, None, None), \
HuffmanNode(7, None, None)), \
HuffmanNode(None, HuffmanNode(10, None, None), HuffmanNode(12, None, None)))
    """
    tree_lst = []
    for i in node_lst:
        tree_node = HuffmanNode(None, None, None)
        if i.l_type == 0:
            tree_node.left = HuffmanNode(i.l_data, None, None)
        if i.r_type == 0: 
            tree_node.right = HuffmanNode(i.r_data, None, None)
        tree_lst.append(tree_node)
    tree_lst.reverse()
    helper_generate_tree_postorder(tree_lst)
    return tree_lst[0]


def helper_generate_tree_postorder(tree_lst):
    ''' a nice function'''
    size = len(tree_lst)
    current = tree_lst[0]
    i = 1
    right_nodes = [tree_lst[0]]
    while current.right == None and i < size:
        current.right = tree_lst[i]
        right_nodes.append(tree_lst[i])
        current = tree_lst[i]
        i += 1
    while right_nodes and i <size:
        current = right_nodes.pop(-1)
        if current.left == None:
            current.left = tree_lst[i]
            current = tree_lst[i]
            right_nodes.append(current)
            i += 1
            while current.right == None and i < size:
                current.right = tree_lst[i]
                right_nodes.append(tree_lst[i])
                current = tree_lst[i]
                i += 1
        
    


def generate_uncompressed(tree, text, size):
    """ Use Huffman tree to decompress size bytes from text.

    @param HuffmanNode tree: a HuffmanNode tree rooted at 'tree'
    @param bytes text: text to decompress
    @param int size: number of bytes to decompress from text.
    @rtype: bytes
    """
    dic = generate_dic(tree)
    lst = [byte_to_bits(byte) for byte in text]
    res = []
    s = ''.join(lst)
    i = 0
    j = 0
    while size > 0:
        if s[i:j] in dic.keys():
            res.append(dic[s[i:j]])
            size -= 1
            i = j
        j += 1
    return bytes(res)
    
            

def generate_dic(tree):
    dic = get_codes(tree)
    res = dict()
    for key,value in dic.items():
        res[value] = key
    return res


def bytes_to_nodes(buf):
    """ Return a list of ReadNodes corresponding to the bytes in buf.

    @param bytes buf: a bytes object
    @rtype: list[ReadNode]

    >>> bytes_to_nodes(bytes([0, 1, 0, 2]))
    [ReadNode(0, 1, 0, 2)]
    """
    lst = []
    for i in range(0, len(buf), 4):
        l_type = buf[i]
        l_data = buf[i+1]
        r_type = buf[i+2]
        r_data = buf[i+3]
        lst.append(ReadNode(l_type, l_data, r_type, r_data))
    return lst


def bytes_to_size(buf):
    """ Return the size corresponding to the
    given 4-byte little-endian representation.

    @param bytes buf: a bytes object
    @rtype: int

    >>> bytes_to_size(bytes([44, 1, 0, 0]))
    300
    """
    return int.from_bytes(buf, "little")


def uncompress(in_file, out_file):
    """ Uncompress contents of in_file and store results in out_file.

    @param str in_file: input file to uncompress
    @param str out_file: output file that will hold the uncompressed results
    @rtype: NoneType
    """
    with open(in_file, "rb") as f:
        num_nodes = f.read(1)[0]
        buf = f.read(num_nodes * 4)
        node_lst = bytes_to_nodes(buf)
        # use generate_tree_general or generate_tree_postorder here
        tree = generate_tree_general(node_lst, num_nodes - 1)
        size = bytes_to_size(f.read(4))
        with open(out_file, "wb") as g:
            text = f.read()
            g.write(generate_uncompressed(tree, text, size))


# ====================
# Other functions

def sorting(dic):
    value = []
    key_lst = []
    for keys,values in dic.items():
        value.append(values)
        key_lst.append(keys)
    value.sort()
    
    key = []
    for size in value:
        index = 0
        while dic[key_lst[index]] != size:
            index += 1
        key.insert(0,key_lst[index])
        key_lst.pop(index)
    return key          
    

def len_sort(lst):
    dic = {}
    res = []
    for i in lst:
        dic.setdefault(len(i),[]).append(i)
    length = sorted(list(dic.keys()))
    for j in length:
        res.extend(dic[j])
    return res

def change_the_tree(tree, dic, s):
    if not tree:
        return None
    if tree.is_leaf():
        tree.symbol = dic[s]
        return None
    change_the_tree(tree.left, dic, s+'0')
    change_the_tree(tree.right, dic, s+'1')

def improve_tree(tree, freq_dict):
    """ Improve the tree as much as possible, without changing its shape,
    by swapping nodes. The improvements are with respect to freq_dict.

    @param HuffmanNode tree: Huffman tree rooted at 'tree'
    @param dict(int,int) freq_dict: frequency dictionary
    @rtype: NoneType

    >>> left = HuffmanNode(None, HuffmanNode(99), HuffmanNode(100))
    >>> right = HuffmanNode(None, HuffmanNode(101), \
    HuffmanNode(None, HuffmanNode(97), HuffmanNode(98)))
    >>> tree = HuffmanNode(None, left, right)
    >>> freq = {97: 26, 98: 23, 99: 20, 100: 16, 101: 15}
    >>> improve_tree(tree, freq)
    >>> avg_length(tree, freq)
    2.31
    """
    s = ''
    d = get_codes(tree)
    sorted_key = sorting(freq_dict) #key ranged from higest freq to lowest freq
    mapping = list(d.values())
    sorted_mapping = len_sort(mapping)
    dic = {}
    for i in range(len(mapping)):
        dic[sorted_mapping[i]] = sorted_key[i]
    change_the_tree(tree, dic, s)
        

if __name__ == "__main__":
    # TODO: Uncomment these when you have implemented all the functions
    import doctest
    doctest.testmod()

    import time

    mode = input("Press c to compress or u to uncompress: ")
    if mode == "c":
        fname = input("File to compress: ")
        start = time.time()
        compress(fname, fname + ".huf")
        print("compressed {} in {} seconds."
              .format(fname, time.time() - start))
    elif mode == "u":
        fname = input("File to uncompress: ")
        start = time.time()
        uncompress(fname, fname + ".orig")
        print("uncompressed {} in {} seconds."
              .format(fname, time.time() - start))