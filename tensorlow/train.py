#！/user/bin/env python3
# -*- coding:utf-8 -*-

from tensorlow.ops import *


class GradientDescentOptimizer(Op):

	def __init__(self, learning_rate):
		self.learning_rate = constant(learning_rate)

	def minimize(self, loss, global_step=None, var_list=None):
		new_node = Node()
		new_node.op = self
		new_node.rate = self.learning_rate
		new_node.var_list = var_list
		new_node.input = [gradients(loss, var_list)]
		new_node.name = "SGD(%s)" % loss.name
		return new_node

	def compute(self, node, input_vals):
		grad = input_vals[1]
		for i, it in enumerate(node.var_list):
			placeholder.value_list[it] -= node.rate * grad[i]
		return None

	def gradient(self, node, grad):
		assert False, "\033[1;31mSGD don't have gradient!\033[0m"


class AdamOptimizer(Op):
	#TODO
	def __init__(self, learning_rate = 0.001, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-08):
		self.learning_rate = constant(learning_rate)
		self.beta1 = constant(beta1)
		self.beta2 = constant(beta2)
		self.epsilon = constant(epsilon)


	def minimize(self, loss, global_step=None, var_list=None):
		new_node = Node()
		new_node.op = self
		if not var_list:
			var_list = Variable.node_list
		new_node.input = [loss ,gradients(loss, var_list)]
		new_node.t = 0
		new_node.w = zeros(np.shape(var_list))
		new_node.v = zeros(np.shape(var_list))
		new_node.b1 = self.beta1
		new_node.b2 = self.beta2
		new_node.rate = self.learning_rate
		return new_node

	def compute(self, node, input_vals):
		loss = input_vals[0]
		grad = input_vals[1]
		for i, it in enumerate(node.var_list):
			loss_now = loss[i]
			grad_now = grad[i]


	def gradient(self, node, grad):
		assert False, "\033[1;31mAdam don't have gradient!\033[0m"



