#！/user/bin/env python3
# -*- coding:utf-8 -*-

import numpy as np


#TODO define types
float16 = np.float16
float32 = np.float32
float = float64 = np.float64
float128 = np.float128

int8 = np.int8
int16 = np.int16
int32 = np.int32
int = int64 = np.int64
#int128 = np.int128

uint8 = np.uint8
uint16 = np.uint16
uint32 = np.uint32
uint = uint64 = np.uint64
#uint128 = np.uint128




#TODO finish Session class
class Session:

    def __init__(self,
                 target = '',
                 graph = None,
                 config = None):
        self.target = target
        self.graph = graph
        self.config = config

    def __enter__(self):
        pass

    def __exit__(self, exec_type, exec_val, exec_tb):
        print(exec_type)
        print(exec_val)
        print(exec_tb)

    def _run(self, output, node_value):
        if type(output) == Op:
            return None
        topo_order = find_topo_sort([output])
        for node in topo_order:
            if type(node.op) in [myVariable, myplaceholder, myConstant]:
                continue
            val = []
            for i in node.input:
                if type(i.op) in [myConstant, myVariable]:
                    val.append(i.value)
                else:
                    val.append(node_value[i])
            #print( "%s : %s" % (node, val))
            node_value[node] = node.op.compute(node, val)
        #node_val_results = [node_to_val_map[node] for node in self.eval_node_list]
        return node_value[output]


    def run(self,
            fetches,
            feed_dict = None,
            options = None,
            run_metadata = None):
        node_value = {}
        if feed_dict:
            for i, j in feed_dict.items():
                if not i in placeholder.node_list:
                    raise NameError
                if isinstance(j, list) :
                    node_value[i] = np.array(j, dtype = i.dtype)
                else:
                    node_value[i] = i.dtype(j)

        if isinstance(fetches, (list, tuple)):
            result = []
            for node in fetches:
                result.append(self._run(node, node_value))
        elif isinstance(fetches, dict):
            result = {}
            for node in fetches:
                result[node] = self._run(node, node_value)
        else:
            result = self._run(fetches, node_value)

        return result


class Node(object):

    def __init__(self):
        self.input = []
        self.op = None
        self.const_attr = None
        self.value = None
        self.dtype = None
        self.shape = ()
        self.name = ""

    def __add__(self, other):
        if isinstance(other, Node):
            return add_op(self, other)
        else:
            return add_byconst_op(self, other)

    def __mul__(self, other):
        if isinstance(other, Node):
            return mul_op(self, other)
        else:
            return mul_byconst_op(self, other)

    def __sub__(self, other):
        if isinstance(other, Node):
            return sub_op(self, other)
        else:
            return sub_byconst_op(self, other)

    def __rsub__(self, other):
        if isinstance(other, Node):
            return sub_op(self, other, True)
        else:
            return sub_byconst_op(self, other, True)

    __radd__ = __add__
    __rmul__ = __mul__

    def __str__(self):
        """Allow print to display node name."""
        return "%s = %s" % (self.name, self.value)

    __repr__ = __str__


class Op(object):

    def __call__(self):
        new_node = Node()
        new_node.op = self
        return new_node

    def compute(self, node, input_vals):
        raise NotImplementedError

    def gradient(self, node, output_grad):
        raise NotImplementedError


class myplaceholder(Op):
    node_list = []
    def __call__(self, dtype, shape = None, name = "placeholder"):
        newNode = Node()
        newNode.dtype = dtype
        newNode.shape = shape
        newNode.name = name
        newNode.op = self
        newNode.const_attr = None
        newNode.value = None
        self.node_list.append(newNode)
        return newNode


class myVariable(Op):
    node_list = []
    def __call__(self,
                 initial_value=None,
                 trainable=True,
                 collections=None,
                 validate_shape=True,
                 caching_device=None,
                 name=None,
                 variable_def=None,
                 dtype=None,
                 expected_shape=None,
                 import_scope=None,
                 constraint=None):
        newNode = Node()
        if isinstance(initial_value, list):
            newNode.value = np.array(initial_value)
        else:
            newNode.value = initial_value
        newNode.dtype = dtype
        newNode.name = name
        newNode.input = [newNode]
        newNode.op = self
        newNode.const_attr = None
        newNode.shape = newNode.value.shape
        self.node_list.append(newNode)
        return newNode


class myConstant(Op):
    node_list = []
    def __call__(self,
                 value,
                 dtype = None,
                 shape = None,
                 name = "Const",
                 verify_shape = False):
        newNode = Node()
        if isinstance(value, list):
            newNode.value = np.array(value)
        else:
            newNode.value = value
        newNode.dtype = dtype
        newNode.name = name
        newNode.input = [newNode]
        newNode.op = self
        newNode.const_attr = None
        if shape:
            assert shape == newNode.value.shape
        newNode.shape = newNode.value.shape
        self.node_list.append(newNode)
        return newNode


class AddOp(Op):
    def __call__(self, nodeA, nodeB):
        newNode = Node()
        newNode.op = self
        newNode.input = [nodeA, nodeB]
        newNode.name = "%s+%s" % (nodeA.name, nodeB.name)
        return newNode

    def compute(self, node, input_vals):
        assert len(input_vals) == 2
        return input_vals[0] + input_vals[1]

    def gradient(self, node, output_grad):
        return [output_grad, output_grad]


class Add_byConstant_Op(Op):

    def __call__(self, nodeA, const_val):
        newNode = Node()
        newNode.op = self
        newNode.input = [nodeA]
        newNode.const_attr = const_val
        newNode.name = "%s+%s" % (nodeA.name, str(const_val))
        return newNode

    def compute(self, node, input_vals):
        assert len(input_vals) == 1
        return input_vals[0] + node.const_attr

    def gradient(self, node, output_grad):
        return [output_grad]


class SubOp(Op):
    def __call__(self, nodeA, nodeB, trans = False):
        newNode = Node()
        newNode.op = self
        newNode.input = [nodeA, nodeB]
        newNode.trans = trans
        if trans:
            newNode.name = "%s-%s" % (nodeB.name, nodeA.name)
        else:
            newNode.name = "%s-%s" % (nodeA.name, nodeB.name)
        return newNode

    def compute(self, node, input_vals):
        assert len(input_vals) == 2
        if node.trans:
            return input_vals[1] - input_vals[0]
        else:
            return input_vals[0] - input_vals[1]

    def gradient(self, node, output_grad):
        if node.trans:
            return [-output_grad, output_grad]
        else:
            return [output_grad, -output_grad]


class Sub_byConstant_Op(Op):

    def __call__(self, nodeA, const_val, trans = False):
        newNode = Node()
        newNode.op = self
        newNode.input = [nodeA]
        newNode.const_attr = const_val
        newNode.trans = trans
        if trans:
            newNode.name = "%s-%s" % (str(const_val), nodeA.name)
        else:
            newNode.name = "%s-%s" % (nodeA.name, str(const_val))
        return newNode

    def compute(self, node, input_vals):
        assert len(input_vals) == 1
        if node.trans:
            return node.const_attr - input_vals[0]
        else:
            return input_vals[0] - node.const_attr

    def gradient(self, node, output_grad):
        if node.trans:
            return [-output_grad]
        else:
            return [output_grad]


class MulOp(Op):

    def __call__(self, node_A, node_B):
        newNode = Node()
        newNode.op = self
        newNode.input = [node_A, node_B]
        newNode.name = "(%s*%s)" % (node_A.name, node_B.name)
        return newNode

    def compute(self, node, input_vals):
        assert len(input_vals) == 2
        return input_vals[0] * input_vals[1]

    def gradient(self, node, output_grad):
        return [node.input[1] * output_grad, node.input[0] * output_grad]


class Mul_byConstant_Op(Op):

    def __call__(self, node_A, const_val):
        newNode = Node()
        newNode.op = self
        newNode.const_attr = const_val
        newNode.input = [node_A]
        newNode.name = "(%s*%s)" % (node_A.name, str(const_val))
        return newNode

    def compute(self, node, input_vals):
        assert len(input_vals) == 1
        return input_vals[0] * node.const_attr

    def gradient(self, node, output_grad):
        return [node.const_attr * output_grad]


class Reduce_Sum(Op):

    def __call__(self, ):
        newNode = Node()
        newNode.op = self
        newNode.input =

        return newNode


def reduce_sum(input_tensor,
               axis=None,keepdims=None,
               name=None,
               reduction_indices=None,
               keep_dims=None):
    if input

def global_variables_initializer():
    return Op()


def find_topo_sort(node_list):
    visited = set()
    topo_order = []
    for node in node_list:
        topo_sort_dfs(node, visited, topo_order)
    return topo_order

def topo_sort_dfs(node, visited, topo_order):
    if node in visited:
        return
    visited.add(node)
    for n in node.input:
        topo_sort_dfs(n, visited, topo_order)
    topo_order.append(node)

def sum_node_list(node_list):
    from operator import add
    from functools import reduce
    return reduce(add, node_list)



constant = myConstant()
Variable = myVariable()
placeholder = myplaceholder()
add_op = AddOp()
add_byconst_op = Add_byConstant_Op()
mul_op = MulOp()
mul_byconst_op = Mul_byConstant_Op()
sub_op = SubOp()
sub_byconst_op = Sub_byConstant_Op()


if __name__ == "__main__":
    sess = Session()

    W = Variable([.5], dtype = float32)
    b = Variable([1.5], dtype = float32)
    x = placeholder(float32)

    linear_model = W * x + b

    # define error
    y = placeholder(float32)
    error = reduce_sum(linear_model - y)

    # run init
    init = global_variables_initializer()
    sess.run(init)

    # calc error
    feed = {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}

    # assign
    fixW = assign(W, [-1.0])
    fixb = assign(b, [1.])
    sess.run([fixW, fixb])
    ans = sess.run(error, feed)

    assert np.equal(ans, 0)

    # # linear model
    # W = Variable([.5], dtype = float32, name = 'W')
    # b = Variable([1.5], dtype = float32, name = 'b')
    # x = placeholder(float32, name = 'x')
    #
    # linear_model = W * x + b
    #
    # init = global_variables_initializer()
    # sess.run(init)
    #
    # ans = sess.run(linear_model, {x: [1, 2, 3, 4]})
    # assert np.array_equal(ans, [2, 2.5, 3, 3.5])
    #
    # a = placeholder(float64)
    # b = placeholder(float64)
    # adder_node = a + b
    #
    #
    # ans = sess.run(adder_node, {a: 3, b: 4.5})
    # assert np.equal(ans, 7.5)
    #
    # ans = sess.run(adder_node, {a: [1, 3], b: [2, 3]})
    # assert np.array_equal(ans, [3, 6])
    # ans = sess.run(adder_node, {a: [[1, 3],[1, 3]], b: [[2, 3]]})

    # x1 = constant([10, 10], name = "x1")
    # print(x1.shape)
    # print(x1.name)
    # print(x1)