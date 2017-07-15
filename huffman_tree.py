__author__ = 'PC-LiNing'


class TreeNode:
    def __init__(self, parent=-1,left=-1,right=-1,count=1e15,binary=False):
        self.parent = parent
        self.left = left
        self.right = right
        self.count = count
        self.binary = binary


# counts = [0_count,1_count,....]
def build_tree(counts):
    label_num = len(counts)
    huffman_tree = []
    for i in range(0,2*label_num - 1):
        huffman_tree.append(TreeNode())
    for i in range(0,label_num):
        huffman_tree[i].count = counts[i]
    leaf = label_num -1
    node = label_num
    for i in range(label_num,2*label_num - 1):
        mini = [-1,-1]
        for j in range(0,2):
            if leaf>=0 and huffman_tree[leaf].count < huffman_tree[node].count:
                mini[j] = leaf
                leaf -= 1
            else:
                mini[j] = node
                node += 1
        huffman_tree[i].left = mini[0]
        huffman_tree[i].right = mini[1]
        huffman_tree[i].count = huffman_tree[mini[0]].count + huffman_tree[mini[1]].count
        huffman_tree[mini[0]].parent = i
        huffman_tree[mini[1]].parent = i
        huffman_tree[mini[1]].binary = True

    # record path and code
    paths = []
    codes = []
    for i in range(0,label_num):
        path = []
        code = []
        j = i
        while huffman_tree[j].parent != -1:
            path.append(huffman_tree[j].parent)
            code.append(huffman_tree[j].binary)
            j = huffman_tree[j].parent
        paths.append(path)
        codes.append(code)
    return huffman_tree, paths, codes


def find2node(huffman_tree,flag):
    mini = []
    for i in range(0,len(huffman_tree)):
        if i in flag:
            continue
        if len(mini) < 2:
            mini.append(i)
        else:
            if huffman_tree[i].count < max([huffman_tree[mini[0]].count,huffman_tree[mini[1]].count]):
                pos = 0 if huffman_tree[mini[0]].count > huffman_tree[mini[1]].count else 1
                mini[pos] = i
    return sorted(mini)


# counts = [0_count,1_count,....]
def build_huffman_tree(counts):
    max_path_length = 15
    label_num = len(counts)
    huffman_tree = []
    for i in range(0,label_num):
        huffman_tree.append(TreeNode(count=counts[i]))

    flag = []
    for i in range(label_num,2*label_num - 1):
        mini = find2node(huffman_tree, flag)
        flag.append(mini[0])
        flag.append(mini[1])
        count = huffman_tree[mini[0]].count + huffman_tree[mini[1]].count
        node = TreeNode(left=mini[0],right=mini[1],count=count)
        huffman_tree.append(node)
        huffman_tree[mini[0]].parent = len(huffman_tree) - 1
        huffman_tree[mini[1]].parent = len(huffman_tree) - 1
        huffman_tree[mini[1]].binary = True

    # record path and code
    paths = []
    codes = []
    # record path length
    path_length = []
    label_size = 736
    for i in range(0,label_num):
        path = []
        code = []
        j = i
        while huffman_tree[j].parent != -1:
            path.append(huffman_tree[j].parent - label_size)
            code.append(int(huffman_tree[j].binary))
            j = huffman_tree[j].parent
        path_length.append(len(path))
        # padding = 0
        paths.append(path+(max_path_length - len(path))*[0])
        codes.append(code+(max_path_length - len(code))*[0])

    return huffman_tree, paths, codes, path_length

