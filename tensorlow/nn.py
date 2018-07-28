#！/user/bin/env python3
# -*- coding:utf-8 -*-

from tensorlow.ops import *


class SoftmaxOp(Op):

	def __call__(self, node, axis = 1):
		tmpNode = exp(node)
		return tmpNode / reduce_sum(tmpNode, axis = axis, keepdims = True)

	def compute(self, node, input_vals):
		assert False, "\033[1;31mSofrmaxOp can't compute!\033[0m"

	def gradient(self, node, grad):
		assert False, "\033[1;31mSofrmaxOp don't have gradient!\033[0m"


class Softmax_Cross_Entropy_With_LogitsOp(Op):
	def __call__(self, labels, logits):
		#tmp = reduce_sum(labels * logits, axis = 1) + reduce_sum(labels, axis = 1) * log(reduce_sum(exp(-logits), axis = 1, keepdims = True))
		tmp = -reduce_sum(labels * log(nn.softmax(logits)), axis = 1)
		return tmp

	def compute(self, node, input_vals):
		assert False, "\033[1;31mSoftmax_Cross_Entropy_With_LogitsOp can't compute!\033[0m"

	def gradient(self, node, grad):
		assert False, "\033[1;31mSoftmax_Cross_Entropy_With_LogitsOp don't have gradient!\033[0m"


class SingOp(Op):

	def __call__(self, node1):
		new_node = Node()
		new_node.op = self
		new_node.input = [node1]
		new_node.name = "sing(%s)" % node1.name
		return new_node

	def compute(self, node, input_vals):
		assert len(input_vals) == 1, "\033[1;31mNode number not suit at sign!\033[0m"
		#print(np.maximum(np.sign(input_vals[0]), 0))
		return np.maximum(np.sign(input_vals[0]), 0)

	def gradient(self, node, grad):
		assert False, "\033[1;31mSignOp don't have gradient!\033[0m"


class ReluOp(Op):
	cnt = 0
	def __call__(self, features):
		new_node = Node()
		new_node.op = self
		new_node.input = [features]
		new_node.name = "relu(%s)" % features.name
		return new_node

	def compute(self, node, input_vals):
		assert len(input_vals) == 1, "\033[1;31mNode number not suit at relu!\033[0m"
		#print(np.maximum(input_vals[0], 0))
		return np.maximum(input_vals[0], 0)

	def gradient(self, node, grad):
		return [sign(node.input[0]) * grad]


class Conv2dOp(Op):
	def __call__(self):
		new_node = Node()
		new_node.op = self
		#TODO
		return new_node

	def compute(self, node, input_vals):
		assert len(input_vals) == len(node.input), "Node number not suit!"
		#TODO

	def gradient(self, node, grad):
		# TODO
		pass


class MaxpoolOp(Op):
	def __call__(self):
		new_node = Node()
		new_node.op = self
		#TODO
		return new_node

	def compute(self, node, input_vals):
		assert len(input_vals) == len(node.input), "Node number not suit!"
		#TODO

	def gradient(self, node, grad):
		# TODO
		pass


class nn(object):
	softmax = SoftmaxOp()
	softmax_cross_entropy_with_logits = Softmax_Cross_Entropy_With_LogitsOp()
	conv2d = Conv2dOp()
	max_pool = MaxpoolOp()
	relu = ReluOp()


sign = SingOp()