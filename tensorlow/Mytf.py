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

	def __exit__(self, exc_type, exc_val, exc_tb):
		if(exc_type):
			print(exc_type)
			print(exc_val)
			print(exc_tb)

	@staticmethod
	def _run(self, output, node_value):
		if type(output) == Op:
			return None
		topo_order = find_topo_sort([output])
		for node in topo_order:
			if type(node.op) == myplaceholder:
				continue
			val = []
			for i in node.input:
				if type(i.op) in [myConstant, myVariable]:
					val.append(i.value)
				else:
					val.append(node_value[i])
			node_value[node] = node.op.compute(node, val)
		return node_value[output]


	def run(self,
			fetches,
			feed_dict = None,
			options = None,
			run_metadata = None):
		if feed_dict:
			for i, j in feed_dict.items():
				if not i in placeholder.placeholder_list:
					raise NameError
				if i.shape:
					shapei = i.shape
					shapej = np.shape(j)
					assert len(shapei) == len(shapej)
					for x in range(len(shapei)):
						if shapei[x] and shapei[x] != shapej[x]:
							raise TypeError
				if isinstance(j, list):
					placeholder.value_list[i] = np.array(j, dtype = i.dtype)
				else:
					placeholder.value_list[i] = i.dtype(j)

		if isinstance(fetches, (list, tuple)):
			result = []
			for node in fetches:
				result.append(self._run(node, placeholder.value_list))
		elif isinstance(fetches, dict):
			result = {}
			for node in fetches:
				result[node] = self._run(node, placeholder.value_list)
		else:
			result = self._run(fetches, placeholder.value_list)
		if feed_dict:
			for i in feed_dict:
				placeholder.value_list[i] = None
		return result


class Node(object):

	def __init__(self):
		self.input = []
		self.op = None
		self.const_attr = None
		self.dtype = None
		self.shape = ()
		self.isVariable = False
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
		return self.name

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
	placeholder_list = []
	value_list = {}
	def __call__(self, dtype, shape = None, name = "plh"):
		new_node = Node()
		new_node.dtype = dtype
		new_node.shape = shape
		new_node.name = name
		new_node.op = self
		self.placeholder_list.append(new_node)
		return new_node

	def gradient(self, node, output_grad):
		return None


class myVariable(Op):
	def __call__(self,
				 initial_value=None,
				 trainable=True,
				 collections=None,
				 validate_shape=True,
				 caching_device=None,
				 name="var",
				 variable_def=None,
				 dtype=float64,
				 expected_shape=None,
				 import_scope=None,
				 constraint=None):
		new_node = placeholder(dtype, name = name)
		if isinstance(initial_value, list):
			new_node.const_attr = np.array(initial_value, dtype = dtype)
		else:
			new_node.const_attr = dtype(initial_value)

		new_node.shape = new_node.const_attr.shape
		new_node.isVariable = True
		return new_node


class myConstant(Op):
	def __call__(self,
				 value,
				 dtype = None,
				 shape = None,
				 name = "Const",
				 verify_shape = False):
		new_node = placeholder(dtype, name = name)
		if isinstance(value, list):
			new_node.const_attr = np.array(value, dtype = dtype)
		else:
			new_node.const_attr = np.cast(value, dtype = dtype)
		if shape:
			np.reshape(new_node.const_attr.shape, shape)
			new_node.shape = shape
		else:
			new_node.shape = new_node.const_attr.shape
		placeholder.value_list[new_node] = new_node.const_attr
		return new_node


class Assign(Op):

	def __call__(self,
			   ref,
			   value,
			   validate_shape=True,
			   use_locking=None,
			   name="Assign"):
		assert ref.changeAble, "Only Variable can be assigned!"
		new_node = Node()
		new_node.op = self
		new_node.name = name
		new_node.const_attr = value
		new_node.ref = ref
		new_node.validate_shape = validate_shape
		return new_node

	def compute(self, node, input_vals):
		assert len(input_vals) == 0
		ref = node.ref
		if isinstance(node.const_attr, list):
			ref.const_attr = np.array(node.const_attr)
		else:
			ref.const_attr = node.const_attr
		if node.validate_shape:
			ref.const_attr = np.reshape(ref.const_attr, ref.shape)
		else:
			ref.shape = ref.const_attr.shape
		placeholder.value_list[ref] = ref.const_attr
		return ref


class AddOp(Op):
	def __call__(self, node1, node2):
		new_node = Node()
		new_node.op = self
		new_node.input = [node1, node2]
		new_node.name = "%s+%s" % (node1.name, node2.name)
		return new_node

	def compute(self, node, input_vals):
		assert len(input_vals) == 2
		return input_vals[0] + input_vals[1]

	def gradient(self, node, output_grad):
		return [output_grad, output_grad]


class Add_byConstant_Op(Op):

	def __call__(self, node1, const_val):
		new_node = Node()
		new_node.op = self
		new_node.input = [node1]
		new_node.shape = node1.shape
		new_node.const_attr = const_val
		new_node.name = "%s+%s" % (node1.name, str(const_val))
		return new_node

	def compute(self, node, input_vals):
		assert len(input_vals) == 1
		return input_vals[0] + node.const_attr

	def gradient(self, node, output_grad):
		return [output_grad]


class SubOp(Op):
	def __call__(self, node1, node2, trans = False):
		new_node = Node()
		new_node.op = self
		new_node.input = [node1, node2]
		new_node.trans = trans
		if trans:
			new_node.name = "%s-%s" % (node2.name, node1.name)
		else:
			new_node.name = "%s-%s" % (node1.name, node2.name)
		return new_node

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

	def __call__(self, node1, const_val, trans = False):
		new_node = Node()
		new_node.op = self
		new_node.input = [node1]
		new_node.shape = node1.shape
		new_node.const_attr = const_val
		new_node.trans = trans
		if trans:
			new_node.name = "%s-%s" % (str(const_val), node1.name)
		else:
			new_node.name = "%s-%s" % (node1.name, str(const_val))
		return new_node

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
	def __call__(self, node1, node2, trans = False):
		new_node = Node()
		new_node.op = self
		new_node.input = [node1, node2]
		new_node.trans = trans
		if trans:
			new_node.name = "%s/%s" % (node2.name, node1.name)
		else:
			new_node.name = "%s/%s" % (node1.name, node2.name)
		return new_node

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

	def __call__(self, node1, const_val, trans = False):
		new_node = Node()
		new_node.op = self
		new_node.input = [node1]
		new_node.shape = node1.shape
		new_node.const_attr = const_val
		new_node.trans = trans
		if trans:
			new_node.name = "%s/%s" % (str(const_val), node1.name)
		else:
			new_node.name = "%s/%s" % (node1.name, str(const_val))
		return new_node

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

	def __call__(self, node1, node2):
		new_node = Node()
		new_node.op = self
		new_node.input = [node1, node2]
		new_node.name = "(%s*%s)" % (node1.name, node2.name)
		return new_node

	def compute(self, node, input_vals):
		assert len(input_vals) == 2
		result = input_vals[0] * input_vals[1]
		node.shape = np.shape(result)
		return result

	def gradient(self, node, output_grad):
		return [node.input[1] * output_grad, node.input[0] * output_grad]


class Mul_byConstant_Op(Op):

	def __call__(self, node1, const_val):
		new_node = Node()
		new_node.op = self
		new_node.const_attr = const_val
		new_node.input = [node1]
		new_node.shape = node1.shape
		new_node.name = "(%s*%s)" % (node1.name, str(const_val))
		return new_node

	def compute(self, node, input_vals):
		assert len(input_vals) == 1
		return input_vals[0] * node.const_attr

	def gradient(self, node, output_grad):
		return [node.const_attr * output_grad]


class ExpOp(Op):
	def __call__(self, node1):
		new_node = Node()
		new_node.op = self
		new_node.input = [node1]
		new_node.name = "Exp(%s)" % node1.name
		return new_node

	def compute(self, node, input_vals):
		assert len(input_vals) == 1
		return np.exp(input_vals[0])

	def gradient(self, node, output_grad):
		return [output_grad * np.exp(node.input[0])]


class LogOp(Op):
	def __call__(self, node1):
		new_node = Node()
		new_node.op = self
		new_node.input = [node1]
		new_node.name = "Log(%s)" % node1.name
		return new_node

	def compute(self, node, input_vals):
		assert len(input_vals) == 1
		return np.log(input_vals[0])

	def gradient(self, node, output_grad):
		return [1 / node.input[0] * output_grad]


class MatMulOp(Op):

	def __call__(self, node1, node2, trans_A = False, trans_B = False):
		new_node = Node()
		new_node.op = self
		new_node.matmul_attr_trans_A = trans_A
		new_node.matmul_attr_trans_B = trans_B
		new_node.input = [node1, node2]
		new_node.name = "MatMul(%s,%s,%s,%s)" % (node1.name, node2.name, str(trans_A), str(trans_B))
		return new_node

	def compute(self, node, input_vals):
		assert len(input_vals) == 2
		if node.matmul_attr_trans_A:
			if node.matmul_attr_trans_B:
				result = np.matmul(input_vals[0].T, input_vals[1].T)
			else:
				result = np.matmul(input_vals[0].T, input_vals[1])
		else:
			if node.matmul_attr_trans_B:
				result = np.matmul(input_vals[0], input_vals[1].T)
			else:
				result = np.matmul(input_vals[0], input_vals[1])
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


class Expand_SumOp(Op):

	def __call__(self, node1, input_node, grad):
		new_node = Node()
		new_node.op = self
		new_node.input = [input_node, grad]
		new_node.name = "extened(%s, %s)" % (node1.name, grad.name)
		new_node.axis = node1.axis
		new_node.keepdims = node1.keepdims
		return new_node

	def compute(self, node, input_vals):
		assert len(input_vals) == 2
		output_grad = input_vals[1]
		shape = np.shape(input_vals[0])
		if len(shape) == 1:
			result =  output_grad + np.zeros(shape)
		else:
			if node.axis == 1:
				if node.keepdims:
					result = output_grad + np.zeros(shape)
				else:
					result = output_grad.T + np.zeros(shape[1])
				if node.muled:
					result *= 1 / shape[1]
			elif node.axis == 0:
				result = output_grad + np.zeros(shape)
			else:
				result =  output_grad + np.zeros(shape)
		return result

	def gradient(self, node, output_grad):
		return [output_grad]


class Expand_MeanOp(Op):

	def __call__(self, node1, input_node, grad):
		new_node = Node()
		new_node.op = self
		new_node.input = [input_node, grad]
		new_node.name = "extened(%s, %s)" % (node1.name, grad.name)
		new_node.axis = node1.axis
		new_node.keepdims = node1.keepdims
		new_node.muled = type(node1.op) == Reduce_MeanOp
		return new_node

	def compute(self, node, input_vals):
		assert len(input_vals) == 2
		output_grad = input_vals[1]
		shape = np.shape(input_vals[0])
		if len(shape) == 1:
			result = output_grad * (1 / shape[0]) + np.zeros(shape)
		else:
			if node.axis == 1:
				if node.keepdims:
					result = output_grad * (1 / shape[1]) + np.zeros(shape)
				else:
					result = output_grad.T * (1 / shape[1]) + np.zeros(shape[1])
			elif node.axis == 0:
				result = output_grad * (1 / shape[0]) + np.zeros(shape)
			else:
				result = output_grad * (1 / (shape[1] * shape[0])) + np.zeros(shape)
		return result

	def gradient(self, node, output_grad):
		return [output_grad]


class Reduce_SumOp(Op):

	def __call__(self,
				 input_tensor,
				 axis=None,
				 keepdims=None,
				 name=None,
				 reduction_indices=None,
				 keep_dims=None):
		new_node = Node()
		new_node.op = self
		new_node.input = [input_tensor]
		new_node.name = name
		if reduction_indices:
			new_node.axis = reduction_indices[0]
		else:
			new_node.axis = axis
		if keep_dims:
			new_node.keepdims = keep_dims
		else:
			new_node.keepdims = keepdims
		return new_node

	def compute(self, node, input_vals):
		assert len(input_vals) ==  1
		if node.keepdims:
			return np.sum(input_vals[0], axis = node.axis, keepdims = node.keepdims)
		else:
			return np.sum(input_vals[0], axis = node.axis)

	def gradient(self, node, output_grad):
		return [extension(node, node.input[0], output_grad)]


class Reduce_MeanOp(Op):

	def __call__(self,
				 input_tensor,
				 axis=None,
				 keepdims=None,
				 name=None,
				 reduction_indices=None,
				 keep_dims=None):
		new_node = Node()
		new_node.op = self
		new_node.input = [input_tensor]
		new_node.name = name
		if reduction_indices:
			new_node.axis = reduction_indices[0]
		else:
			new_node.axis = axis
		if keep_dims:
			new_node.keepdims = keep_dims
		else:
			new_node.keepdims = keepdims
		return new_node

	def compute(self, node, input_vals):
		assert len(input_vals) ==  1
		if node.keepdims:
			return np.mean(input_vals[0], axis = node.axis, keepdims = node.keepdims)
		else:
			return np.mean(input_vals[0], axis = node.axis)

	def gradient(self, node, output_grad):
		return [extension(node, node.input[0], output_grad)]


class ZerosLikeOp(Op):
	def __call__(self, node1):
		new_node = Node()
		new_node.input = [node1]
		new_node.op = self
		new_node.name = "Zeroslike(%s)" % node1.name
		return new_node

	def compute(self, node, input_vals):
		assert(isinstance(input_vals[0], np.ndarray))
		return np.zeros(input_vals[0].shape)

	def gradient(self, node, output_grad):
		return [zeroslike_op(node.input[0])]


class OnesLikeOp(Op):
	def __call__(self, node1):
		new_node = Node()
		new_node.input = [node1]
		new_node.op = self
		new_node.name = "Oneslike(%s)" % node1.name
		return new_node

	def compute(self, node, input_vals):
		assert(isinstance(input_vals[0], np.ndarray))
		return np.ones(input_vals[0].shape)

	def gradient(self, node, output_grad):
		return [zeroslike_op(node.input[0])]


class Gradients(Op):

	def __call__(self, output_node, node_list):
		node_to_output_grads_list = {}
		node_to_output_grads_list[output_node] = [oneslike_op(output_node)]
		node_to_output_grad = {}
		reverse_topo_order = reversed(find_topo_sort([output_node]))

		for node in reverse_topo_order:
			grad = sum_node_list(node_to_output_grads_list[node])
			node_to_output_grad[node] = grad
			input_grads = node.op.gradient(node, grad)
			if input_grads == None:
				continue
			for i in range(len(node.input)):
				if node_to_output_grads_list.get(node.input[i]) == None:
					node_to_output_grads_list[node.input[i]] = [input_grads[i]]
				else:
					node_to_output_grads_list[node.input[i]].append(input_grads[i])

		grad_node_list = [node_to_output_grad[node] for node in node_list]
		return grad_node_list


class Global_Variables_InitializerOp(Op):

	def __call__(self):
		new_node = Node()
		new_node.op = self
		return new_node

	def compute(self, node, input_vals):
		for i in placeholder.placeholder_list:
			if i.isVariable:
				placeholder.value_list[i] = i.const_attr


class CNN:

	def softmax(self, node):
		tmpNode = exp(-node)
		return tmpNode / reduce_sum(tmpNode, axis = 1, keepdims = True)


def ones(shape, dtype=float32, name=None):
	return np.ones(shape, dtype)

def zeros(shape, dtype=float32, name=None):
	return np.zeros(shape, dtype)


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
gradients = Gradients()
extension = ExtensionOp()
matmul = MatMulOp()
assign = Assign()
global_variables_initializer = Global_Variables_InitializerOp()
oneslike_op = OnesLikeOp()
zeroslike_op = ZerosLikeOp()
nn = CNN()



if __name__ == "__main__":


	x = placeholder(float64, [None, 784], name = "x")
	W = Variable(zeros([784, 10], dtype = float64), name = "W")
	b = Variable(zeros([10], dtype = float64), name = "b")
	y = nn.softmax(matmul(x, W) + b)

	init = global_variables_initializer()

	t = matmul(x, W) + b
	db = gradients(t, [b])[0]
	tmp = gradients(t, [W, b])
	tmp1 = gradients(tmp[0], [W, b])

	# define loss and optimizer
	y_ = placeholder(float64, [None, 10])

	cross_entropy = reduce_mean(-reduce_sum(y_ * log(y), reduction_indices=[1]))
	from tensorflow.examples.tutorials.mnist import input_data
	mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)
	with Session() as sess:
		sess.run(init)
		n = 3
		data_x = np.array([mnist.train.images[i] for i in range(n)])
		data_y = np.array([mnist.train.labels[i] for i in range(n)])
		print("start")
		print(np.shape(sess.run(db, feed_dict = {x: data_x, y_: data_y})))
		print(np.shape(sess.run(tmp[0], feed_dict = {x: data_x, y_: data_y})))
		print(np.shape(sess.run(tmp[1], feed_dict = {x: data_x, y_: data_y})))
		print(np.shape(sess.run(tmp1[0], feed_dict = {x: data_x, y_: data_y})))
		print(np.shape(sess.run(tmp1[1], feed_dict = {x: data_x, y_: data_y})))


	# #context
	# a = placeholder(float32)
	# b = placeholder(float32)
	# adder_node = a + b
	#
	# with Session() as sess:
	#	 ans = sess.run(adder_node, {a: 3, b: 4.5})
	#	 assert np.equal(ans, 7.5)
	#
	#	 ans = sess.run(adder_node, {a: [1, 3], b: [2, 3]})
	#	 assert np.array_equal(ans, [3, 6])
	#
	# #NOTE:assign
	# sess = Session()
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
