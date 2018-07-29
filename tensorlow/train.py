#！/user/bin/env python3
# -*- coding:utf-8 -*-

from tensorlow.ops import *


class GradientDescentOptimizer(Op):

	def __init__(self, learning_rate):
		self.learning_rate = learning_rate

	def minimize(self, loss, global_step=None, var_list=None):
		new_node = Node()
		new_node.op = self
		new_node.rate = self.learning_rate
		if not var_list:
			var_list = Variable.node_list
		new_node.var_list = var_list
		new_node.input = gradients(loss, var_list)
		new_node.name = "GD(%s)" % loss.name
		return new_node

	def compute(self, node, input_vals):
		grad = input_vals
		for i, it in enumerate(node.var_list):
			placeholder.value_list[it] -= node.rate * grad[i]
		return None

	def gradient(self, node, grad):
		assert False, "\033[1;31mSGD don't have gradient!\033[0m"


class AdamOptimizer(Op):
	#TODO
	def __init__(self, learning_rate = 0.001, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-08):
		self.learning_rate = learning_rate
		self.beta1 = beta1
		self.beta2 = beta2
		self.epsilon = epsilon


	def minimize(self, loss, global_step=None, var_list=None):
		new_node = Node()
		new_node.op = self
		if not var_list:
			var_list = Variable.node_list
		new_node.input = gradients(loss, var_list)
		new_node.var_list = var_list
		new_node.t = 0
		new_node.m = np.zeros(np.shape(var_list))
		new_node.v = np.zeros(np.shape(var_list))
		new_node.b1 = self.beta1
		new_node.b2 = self.beta2
		new_node.eps = self.epsilon
		new_node.rate = self.learning_rate
		new_node.name = "Adma(%s)" % loss.name
		return new_node

	def compute(self, node, input_vals):
		grad = np.array(input_vals)
		node.t = node.t + 1
		node.m = node.m * node.b1 + (1 - node.b1) * grad
		node.v = node.v * node.b2 + (1 - node.b2) * np.square(grad)
		m = node.m / (1 - np.power(node.b1, node.t))
		v = node.v / (1 - np.power(node.b2, node.t))
		for i, it in enumerate(node.var_list):
			placeholder.value_list[it] -= node.rate * m[i] / np.sqrt(v[i] + node.eps)
		return None

	def gradient(self, node, grad):
		assert False, "\033[1;31mAdam don't have gradient!\033[0m"



