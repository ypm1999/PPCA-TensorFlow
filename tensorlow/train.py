#！/user/bin/env python3
# -*- coding:utf-8 -*-

from tensorlow.ops import *


class GradientDescentOptimizer(Op):

	def __init__(self, learning_rate):
		self.learning_rate = learning_rate

	def minimize(self, loss, global_step=None, var_list=None):
		new_node = Node()
		new_node.op = self
		if not var_list:
			var_list = Variable.node_list
		grad = gradients(loss, var_list)
		for i in range(len(var_list)):
			new_node.input.append(assign(var_list[i], var_list[i] - self.learning_rate * grad[i]))
		new_node.name = "SGD(%s)" % loss.name
		return new_node

	def compute(self, node, input_vals):
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


	def minimize(self):
		new_node = Node()
		new_node.op = self
		#TODO
		return new_node


