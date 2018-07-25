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


class Session:

    def __init__(self,
                 target = '',
                 graph = None,
                 config = None):
        self.target = target
        self.graph = graph
        self.config = config

    def __enter__(self):
        return self

    def __exit__(self, exec_type, exec_val, exec_tb):
        if(exec_type):
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
                if not i in placeholder.placeholder_list:
                    raise NameError
                if i.shape:
                    shapei = i.shape
                    shapej = np.shape(j)
                    assert len(shapei) == len(shapej)
                    for x in range(len(shapei)):
                        if(shapei[x] and shapei[x] != shapej[x]):
                            raise TypeError
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

    def __neg__(self):
        return sub_byconst_op(self, 0, True)

    def __truediv__(self, other):
        if isinstance(other, Node):
            return div_op(self, other)
        else:
            return div_byconst_op(self, other)

    def __rtruediv__(self, other):
        if isinstance(other, Node):
            return div_op(self, other, True)
        else:
            return div_byconst_op(self, other, True)

    __radd__ = __add__
    __rmul__ = __mul__

    def __str__(self):
        """Allow print to display node name."""
        return "%s = %s" % (self.name, self.value)

    __repr__ = __str__


class Op(object):

    def __call__(self):
        newNode = Node()
        newNode.op = self
        return newNode

    def compute(self, node, input_vals):
        raise NotImplementedError

    def gradient(self, node, output_grad):
        raise NotImplementedError


class myplaceholder(Op):
    placeholder_list = []
    def __call__(self, dtype, shape = None, name = "plh"):
        newNode = Node()
        newNode.dtype = dtype
        newNode.shape = shape
        newNode.name = name
        newNode.op = self
        newNode.const_attr = None
        newNode.value = None
        self.placeholder_list.append(newNode)
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
        newNode.input = []
        newNode.op = self
        newNode.const_attr = None
        newNode.shape = newNode.value.shape
        self.node_list.append(newNode)
        return newNode


class Assign(Op):

    def __call__(self,
               ref,
               value,
               validate_shape=True,
               use_locking=None,
               name="Assign"):
        assert type(ref.op) == myVariable
        newNode = Node()
        newNode.op = self
        newNode.name = name
        newNode.value = value
        newNode.ref = ref
        newNode.validate_shape = validate_shape
        return newNode

    def compute(self, node, input_vals):
        #assert input_vals == None
        ref = node.ref
        if isinstance(node.value, list):
            ref.value = np.array(node.value)
        else:
            ref.value = node.value
        if node.validate_shape:
            ref.value = np.reshape(ref.value, ref.shape)
        else:
            ref.shape = ref.value.shape
        return ref


class myConstant(Op):
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
            np.reshape(newNode.value.shape, shape)
            newNode.shape = shape
        else:
            newNode.shape = newNode.value.shape
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
        result = input_vals[0] + input_vals[1]
        node.shape = np.shape(result)
        return result

    def gradient(self, node, output_grad):
        return [output_grad, output_grad]


class Add_byConstant_Op(Op):

    def __call__(self, nodeA, const_val):
        newNode = Node()
        newNode.op = self
        newNode.input = [nodeA]
        newNode.shape = nodeA.shape
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
        newNode.shape = nodeA.shape
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


class DivOp(Op):
    def __call__(self, nodeA, nodeB, trans = False):
        newNode = Node()
        newNode.op = self
        newNode.input = [nodeA, nodeB]
        newNode.trans = trans
        if trans:
            newNode.name = "%s/%s" % (nodeB.name, nodeA.name)
        else:
            newNode.name = "%s/%s" % (nodeA.name, nodeB.name)
        return newNode

    def compute(self, node, input_vals):
        assert len(input_vals) == 2
        if node.trans:
            return input_vals[1] / input_vals[0]
        else:
            return input_vals[0] / input_vals[1]

    def gradient(self, node, output_grad):
        if node.trans:
            return [-output_grad * node.input[1] / (node.input[0] * node.input[0]), output_grad / node.input[0]]
        else:
            return [output_grad / node.input[1], -output_grad * node.input[0] / (node.input[1] * node.input[1])]


class Div_byConstant_Op(Op):

    def __call__(self, nodeA, const_val, trans = False):
        newNode = Node()
        newNode.op = self
        newNode.input = [nodeA]
        newNode.shape = nodeA.shape
        newNode.const_attr = const_val
        newNode.trans = trans
        if trans:
            newNode.name = "%s/%s" % (str(const_val), nodeA.name)
        else:
            newNode.name = "%s/%s" % (nodeA.name, str(const_val))
        return newNode

    def compute(self, node, input_vals):
        assert len(input_vals) == 1
        if node.trans:
            return node.const_attr / input_vals[0]
        else:
            return input_vals[0] / node.const_attr

    def gradient(self, node, output_grad):
        if node.trans:
            return [output_grad / node.input[0]]
        else:
            return [-output_grad * node.const_attr / (node.input[0] * node.input[0])]


class MulOp(Op):

    def __call__(self, node_A, node_B):
        newNode = Node()
        newNode.op = self
        newNode.input = [node_A, node_B]
        newNode.name = "(%s*%s)" % (node_A.name, node_B.name)
        return newNode

    def compute(self, node, input_vals):
        assert len(input_vals) == 2
        result = input_vals[0] * input_vals[1]
        node.shape = np.shape(result)
        return result

    def gradient(self, node, output_grad):
        return [node.input[1] * output_grad, node.input[0] * output_grad]


class Mul_byConstant_Op(Op):

    def __call__(self, nodeA, const_val):
        newNode = Node()
        newNode.op = self
        newNode.const_attr = const_val
        newNode.input = [nodeA]
        newNode.shape = nodeA.shape
        newNode.name = "(%s*%s)" % (nodeA.name, str(const_val))
        return newNode

    def compute(self, node, input_vals):
        assert len(input_vals) == 1
        return input_vals[0] * node.const_attr

    def gradient(self, node, output_grad):
        return [node.const_attr * output_grad]


class ExpOp(Op):
    def __call__(self, node_A):
        newNode = Node()
        newNode.op = self
        newNode.input = [node_A]
        newNode.name = "Exp(%s)" % node_A.name
        return newNode

    def compute(self, node, input_vals):
        assert len(input_vals) == 1
        return np.exp(input_vals[0])

    def gradient(self, node, output_grad):
        return [output_grad * np.exp(node.input[0])]


class LogOp(Op):
    def __call__(self, node_A):
        newNode = Node()
        newNode.op = self
        newNode.input = [node_A]
        newNode.name = "Log(%s)" % node_A.name;
        return newNode

    def compute(self, node, input_vals):
        assert len(input_vals) == 1
        return np.log(input_vals[0])

    def gradient(self, node, output_grad):
        return [1 / node.input[0] * output_grad]


class MatMulOp(Op):
    """Op to matrix multiply two nodes."""

    def __call__(self, node_A, node_B, trans_A = False, trans_B = False):
        """Create a new node that is the result a matrix multiple of two input nodes.

        Parameters
        ----------
        node_A: lhs of matrix multiply
        node_B: rhs of matrix multiply
        trans_A: whether to transpose node_A
        trans_B: whether to transpose node_B

        Returns
        -------
        Returns a node that is the result a matrix multiple of two input nodes.
        """
        newNode = Op.__call__(self)
        newNode.matmul_attr_trans_A = trans_A
        newNode.matmul_attr_trans_B = trans_B
        newNode.input = [node_A, node_B]
        newNode.name = "MatMul(%s,%s,%s,%s)" % (node_A.name, node_B.name, str(trans_A), str(trans_B))
        return newNode

    def compute(self, node, input_vals):
        assert len(input_vals) == 2
        if node.matmul_attr_trans_A:
            if node.matmul_attr_trans_B:
                result = np.dot(input_vals[0].T, input_vals[1].T)
            else:
                result = np.dot(input_vals[0].T, input_vals[1])
        else:
            if node.matmul_attr_trans_B:
                result = np.dot(input_vals[0], input_vals[1].T)
            else:
                result = np.dot(input_vals[0], input_vals[1])
        node.shape = np.shape(result)
        return result

    def gradient(self, node, output_grad):
        if node.matmul_attr_trans_A:
            if node.matmul_attr_trans_B:
                return [matmul(output_grad, node.input[1], False, False),
                        matmul(node.input[0], output_grad, False, False)]
            else:
                return [matmul(output_grad, node.input[1], False, True),
                        matmul(node.input[0], output_grad, False, False)]
        else:
            if node.matmul_attr_trans_B:
                return [matmul(output_grad, node.input[1], False, False),
                        matmul(node.input[0], output_grad, True, False)]
            else:
                return [matmul(output_grad, node.input[1], False, True),
                        matmul(node.input[0], output_grad, True, False)]


class ExtentionOp(Op):
    def __call__(self, nodeA, grad, muled = False):
        newNode = Node()
        newNode.op = self
        newNode.input = [nodeA, grad]
        newNode.name = "extened(%s, %s)" % (nodeA.name, grad.name)
        newNode.const_attr = muled
        return newNode

    def compute(self, node, input_vals):
        assert len(input_vals) == 2
        output_grad = np.array(input_vals[1])
        mul = 1
        if node.const_attr:
            mul = node.input[0].input[0].shape
        node = node.input[0]
        shape = node.input[0].shape
        if len(shape) == 1:
            return output_grad * ones(shape)
        else:
            if node.axis == 1:
                if node.keepdims:
                    return np.dot(output_grad, ones(shape[0]))
                else:
                    return np.dot(output_grad.T, ones(shape[1]))
            elif node.axis == 0:
                return np.dot(ones(n).T, output_grad)
            else:
                return output_grad * ones(shape)

    def gradient(self, node, output_grad):
        pass


class Reduce_SumOp(Op):

    def __call__(self,
                 input_tensor,
                 axis=None,
                 keepdims=None,
                 name=None,
                 reduction_indices=None,
                 keep_dims=None):
        newNode = Node()
        newNode.op = self
        newNode.input = [input_tensor]
        newNode.name = name
        if reduction_indices:
            newNode.axis = reduction_indices[0]
        else:
            newNode.axis = axis
        if keep_dims:
            newNode.keepdims = keep_dims
        else:
            newNode.keepdims = keepdims
        return newNode

    def compute(self, node, input_vals):
        assert len(input_vals) ==  1
        if node.keepdims:
            return np.sum(input_vals[0], axis = node.axis, keepdims = node.keepdims)
        else:
            return np.sum(input_vals[0], axis = node.axis)

    def gradient(self, node, output_grad):
        n = node.input[0].shape[0]
        m = node.input[0].shape[1]
        if node.axis == 1:
            if node.keepdims:
                return [matmul(output_grad, Variable(ones(m)), False, False)]
            else:
                return [matmul(output_grad, Variable(ones(m)), True, False)]
        elif node.axis == 0:
            return [matmul(Variable(ones(n)), output_grad), True, False]
        else:
            return [output_grad * Variable(ones([n, m]))]


class Reduce_MeanOp(Op):

    def __call__(self,
                 input_tensor,
                 axis=None,
                 keepdims=None,
                 name=None,
                 reduction_indices=None,
                 keep_dims=None):
        newNode = Node()
        newNode.op = self
        newNode.input = [input_tensor]
        newNode.name = name
        if reduction_indices:
            newNode.axis = reduction_indices[0]
        else:
            newNode.axis = axis
        if keep_dims:
            newNode.keepdims = keep_dims
        else:
            newNode.keepdims = keepdims
        return newNode

    def compute(self, node, input_vals):
        assert len(input_vals) ==  1
        if node.keepdims:
            return np.mean(input_vals[0], axis = node.axis, keepdims = node.keepdims)
        else:
            return np.mean(input_vals[0], axis = node.axis)

    def gradient(self, node, output_grad):
        n = node.input[0].shape[0]
        m = node.input[0].shape[1]
        if node.axis == 1:
            if node.keepdims:
                return [matmul(output_grad, Variable(ones(m) / m), False, False)]
            else:
                return [matmul(output_grad, Variable(ones(m) / m), True, False)]
        elif node.axis == 0:
            return [matmul(Variable(ones(n) / n), output_grad), True, False]
        else:
            return [output_grad * Variable(ones([n, m]) / (n * m))]


class OnesOp(Op):

    def __call__(self,
                 shape,
                 dtype=float32,
                 name=None):
        return np.ones(shape, dtype)


class ZeroOp(Op):

    def __call__(self,
                 shape,
                 dtype=float32,
                 name=None):
        return np.zeros(shape, dtype)


class Gradients(Op):

    def __call__(self):
        pass

class CNN:

    def softmax(self, node):
        tmpNode = exp(-node)
        return tmpNode / reduce_sum(tmpNode, axis = 1, keepdims = True)


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
div_op = DivOp()
div_byconst_op = Div_byConstant_Op()
log = LogOp()
exp = ExpOp()
reduce_sum = Reduce_SumOp()
reduce_mean = Reduce_MeanOp()
#gradients = Graidents()
matmul = MatMulOp()
assign = Assign()
ones = OnesOp()
zeros = ZeroOp()
nn = CNN()



if __name__ == "__main__":

    a = np.array([[1, 2], [3, 4], [4, 5]])
    b = np.array([[10, 10, 10]])
    x = np.zeros([5, 3])
    print(x)
    print(x + b)
    exit(0)
    sess = Session()

    x = placeholder(float64, [None, 784])
    W = Variable(zeros([784, 10], dtype = float64))
    b = Variable(zeros([10], dtype = float64))
    y = nn.softmax(matmul(x, W) + b)
    t = reduce_sum(x, axis = 1)
    print(t.op.gradient(t, [1]))

    # define loss and optimizer
    y_ = placeholder(float64, [None, 10])

    cross_entropy = reduce_mean(-reduce_sum(y_ * log(y), reduction_indices=[1]))
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)
    with Session() as sess:
        n = 1
        data_x = np.array([mnist.train.images[i] for i in range(n)])
        data_y = np.array([mnist.train.labels[i] for i in range(n)])
        print("start")
        print(sess.run(gradients(y, [W])[0], feed_dict = {x: data_x, y_: data_y}))

    # #NOTE:assign
    # W = Variable([.5], dtype = float32)
    # b = Variable([1.5], dtype = float32)
    # x = placeholder(float32)
    #
    # linear_model = W * x + b
    #
    # y = placeholder(float32)
    # error = reduce_sum(linear_model - y)
    #
    # init = global_variables_initializer()
    # sess.run(init)
    #
    # feed = {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}
    #
    # fixW = assign(W, [-1.0])
    # fixb = assign(b, [1.])
    # sess.run([fixW, fixb])
    # ans = sess.run(error, feed)
    #
    # assert np.equal(ans, 0)
    #
    # #init
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
    # #add_node
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
    #
    #
    # #context
    # a = placeholder(float32)
    # b = placeholder(float32)
    # adder_node = a + b
    #
    # with Session() as sess:
    #     ans = sess.run(adder_node, {a: 3, b: 4.5})
    #     assert np.equal(ans, 7.5)
    #
    #     ans = sess.run(adder_node, {a: [1, 3], b: [2, 3]})
    #     assert np.array_equal(ans, [3, 6])