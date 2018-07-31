#！/user/bin/env python3
# -*- coding:utf-8 -*-

from tensorlow.ops import *
from scipy import signal

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
	def __call__(self, image, filter, strides, padding):
		new_node = Node()
		new_node.op = self
		new_node.input = [image, filter]
		new_node.strides = strides
		new_node.padding = padding
		new_node.name = "conv2d(%s,%s)" % (image.name, filter.name)
		return new_node

	def compute(self, node, input_vals):
		assert len(input_vals) == 2, "\033[1;31mNode number not suit at nn.conv2d!\033[0m"
		img = input_vals[0]
		flt = input_vals[1]
		num, n, m, ins = np.shape(img)
		fn, fm, fin, fout = np.shape(flt)
		assert ins == fin, "\033[1;31mThe number of channels is not same for img and filter!\033[0m"
		if(node.padding == 'SAME'):
			result = np.zeros(shape = (num, n, m, fout))
		else:
			result = np.zeros(shape = (num, n - fn + 1, m - fm + 1, fout))

		filter = np.zeros(shape = (fin, fout, fn, fm))
		for i in range(fin):
			for j in range(fout):
				filter[i][j] = np.rot90(flt[:, :, i, j], 2)

		pad_up = (fn - 1) // 2
		pad_down = fn // 2
		pad_left = (fm - 1) // 2
		pad_right = fm // 2
		for i in range(num):
			for k in range(fin):
				image = img[i, :, :, k].copy()
				if (node.padding == 'SAME'):
					image = np.pad(image, ((pad_up, pad_down), (pad_left, pad_right)), 'constant')
				for j in range(fout):
					result[i, :, :, j] += signal.convolve2d(image, filter[k][j], 'valid')

		return result


	def gradient(self, node, grad):
		return [grad_of_conv2d(node.input[0], node.input[1], grad, node.strides, node.padding),
		        grad_toW_ofconv2d(node.input[0], node.input[1], grad, node.strides, node.padding)]


class Grad_Of_conv2dOp(Op):
	def __call__(self, img, filter, grad, strides, padding):
		new_node = Node()
		new_node.op = self
		new_node.input = [img, filter, grad]
		new_node.strides = strides
		new_node.padding = padding
		return new_node

	def compute(self, node, input_vals):
		assert len(input_vals) == 3, "\033[1;31mNode number not suit at nn.conv2d!\033[0m"
		img = input_vals[0]
		flt = input_vals[1]
		grad = input_vals[2]
		fn, fm, fin, fout = np.shape(flt)
		num, n, m, ins = np.shape(img)

		if node.padding == 'VALID':
			grad = np.pad(grad, ((0, 0), (fn - 1, fn - 1), (fm - 1, fm - 1), (0, 0)), 'constant')
		else:
			grad = np.pad(grad, ((0, 0), (fn // 2, (fn - 1) // 2), (fm // 2, (fm - 1) // 2), (0, 0)), 'constant')

		filter = np.zeros(shape = (fin, fout, fn, fm))
		for i in range(fin):
			for j in range(fout):
				filter[i][j] = flt[:, :, i, j]

		result = np.zeros(shape = (num, n, m, fin))
		for i in range(num):
			for j in range(fout):
				image = grad[i, :, :, j].copy()
				for k in range(fin):
					result[i, :, :, k] += signal.convolve2d(image, filter[k, j], 'valid')

		return result

	def gradient(self, node, grad):
		assert False, "\033[1;31mgradient of conv2d don't have gradient!\033[0m"


class Grad_toW_Of_conv2dOp():
	def __call__(self, img, filter, grad, strides, padding):
		new_node = Node()
		new_node.op = self
		new_node.input = [img, filter, grad]
		new_node.strides = strides
		new_node.padding = padding
		return new_node

	def compute(self, node, input_vals):
		assert len(input_vals) == 3, "\033[1;31mNode number not suit at nn.conv2d!\033[0m"
		img = input_vals[0]
		shape = input_vals[1].shape
		flt  = input_vals[2]
		num, fn, fm, fout = np.shape(flt)
		num, n, m, fin = np.shape(img)

		if node.padding == 'SAME':
			img = np.pad(img, ((0, 0), (shape[0] // 2, (shape[0] - 1) // 2), (shape[1] // 2, (shape[1] - 1) // 2), (0, 0)), 'constant')

		filter = np.zeros(shape = (num, fout, fn, fm))
		for i in range(num):
			for j in range(fout):
				filter[i][j] = np.rot90(flt[i, :, :, j], 2)

		result = np.zeros(shape = (shape[0], shape[1], fin, fout))
		for i in range(num):
			for k in range(fin):
				image = img[i, :, :, k].copy()
				for j in range(fout):
					result[:, :, k, j] += signal.convolve2d(image, filter[i][k], 'valid')

		return result

	def gradient(self, node, grad):
		assert False, "\033[1;31mgradient of conv2d don't have gradient!\033[0m"


class MaxpoolOp(Op):
	def __call__(self, value, ksize, strides, padding):
		assert ksize == strides, "\033[1;31mksize != strides at max_pool, not support!\033[0m"
		assert strides[0] == 1 and strides[3] == 1, "\033[1;31mstrides are not be as [1, x, x, 1] at max_pool, notsupport!\033[0m"
		new_node = Node()
		new_node.op = self
		new_node.input = [value]
		new_node.ksize = ksize
		new_node.strides = strides
		new_node.padding = padding.lower()
		new_node.name = "maxpool(%s)" % value.name
		return new_node

	def compute(self, node, input_vals):
		assert len(input_vals) == 1, "\033[1;31mNode number not suit at nn.max_pool!\033[0m"
		img = input_vals[0]
		shape = np.shape(img)
		num, n, m, ins = [shape[i] // node.strides[i] for i in range(4)]
		result = np.zeros(shape = (num, n, m, ins), dtype = float64) - 1e100

		for i in range(num):
			for j in range(shape[1]):
				idj = j // node.strides[1]
				for k in range(shape[2]):
					idk = k // node.strides[2]
					res = result[i][idj][idk]
					tmp = img[i][j][k]
					res[:] = np.maximum(res, tmp)

		return result

	def gradient(self, node, grad):
		return [grad_of_maxpool(node.input[0], node, grad, node.strides)]


class Grad_Of_MaxpoolOp(Op):
	def __call__(self, node1, node2, node3, strides):
		new_node = Node()
		new_node.op = self
		new_node.input = [node1, node2, node3]
		new_node.strides = strides
		return new_node

	def compute(self, node, input_vals):
		assert len(input_vals) == 3, "\033[1;31mNode number not suit at max_pool's gradient!\033[0m"
		img = input_vals[0]
		mx = input_vals[1]
		grad = input_vals[2]
		num, n, m, ins = np.shape(img)
		result = np.zeros(shape = (num, n, m, ins), dtype = float64)

		for i in range(num):
			for j in range(n):
				idj = j // node.strides[1]
				for k in range(m):
					idk = k // node.strides[2]
					mx_now = mx[i][idj][idk]
					img_now =  img[i][j][k]
					res = result[i][j][k]
					grad_now = grad[i][idj][idk]
					for p in range(ins):
						if mx_now[p] == img_now[p]:
							res[p] = grad_now[p]
							mx_now[p] += 1

		return result

	def gradient(self, node, grad):
		assert False, "\033[1;31mgradient of max_pool don't have gradient!\033[0m"


class DropoutOp(Op):
	def __call__(self, x, keep_prob):
		new_node = Node()
		new_node.op = self
		new_node.input = [x, keep_prob]
		new_node.data = None
		new_node.name = "dropout(%s,%s)" % (x.name, keep_prob.name)
		return new_node

	def compute(self, node, input_vals):
		assert len(input_vals) == 2, "\033[1;31mNode number not suit at nn.max_pool!\033[0m"
		x = input_vals[0]
		keep_prob = input_vals[1]
		shape = np.shape(x)
		node.data = np.random.rand(*shape)
		node.data = node.data < keep_prob
		return x * node.data

	def gradient(self, node, grad):
		return [grad_of_dropout(node, grad), 0]


class Grad_Of_DropoutOp(Op):
	def __call__(self, node1, node2):
		new_node = Node()
		new_node.op = self
		new_node.input = [node1, node2]
		new_node.name = "grad_of_dropout(%s,%s)" % (node1.name, node2.name)
		return new_node

	def compute(self, node, input_vals):
		assert len(input_vals) == 2, "\033[1;31mNode number not suit at nn.max_pool!\033[0m"
		return input_vals[1] * node.input[0].data

	def gradient(self, node, grad):
		assert False, "\033[1;31mgradient of dropout don't have gradient!\033[0m"


class nn(object):
	softmax = SoftmaxOp()
	softmax_cross_entropy_with_logits = Softmax_Cross_Entropy_With_LogitsOp()
	conv2d = Conv2dOp()
	max_pool = MaxpoolOp()
	relu = ReluOp()
	dropout = DropoutOp()


sign = SingOp()
grad_of_maxpool = Grad_Of_MaxpoolOp()
grad_of_conv2d = Grad_Of_conv2dOp()
grad_toW_ofconv2d = Grad_toW_Of_conv2dOp()
grad_of_dropout = Grad_Of_DropoutOp()


if __name__  == "__main__":
	img = np.array([[[[1], [3], [8], [0]],
#			        [[0], [1], [5], [1], [0]],
			        [[9], [0], [1], [7]],
			        [[0], [8], [1], [4]],
			        [[8], [1], [1], [0]]]]*1, dtype = float64)

	flt = np.array([ [ [[1] * 1], [[0] * 1], [[1] * 1] ],
			        [ [[0] * 1], [[1] * 1], [[0] * 1] ],
			        [ [[1] * 1], [[0] * 1], [[1] * 1] ]], dtype = float64)

	fflt = flt[0:2, 0:2]
	a = nn.conv2d(None, None, [1, 1, 1, 1], 'SAME')
	res = a.op.compute(a, [img, flt])
	print(res.shape)
	print(res)
	b = nn.max_pool(None, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')
	mx = b.op.compute(b, [res])
	print(mx.shape)
	print(mx)
	c = grad_of_maxpool(b, None, None, [1, 2, 2, 1])
	grad = c.op.compute(c, [res, mx, ones(mx.shape)])
	print(grad.shape)
	print(grad)
	d = grad_of_conv2d(a.input[0], a.input[1], None, [1, 1, 1, 1], a.padding)
	e = grad_toW_ofconv2d(a.input[0], a.input[1], None, [1, 1, 1, 1], a.padding)
	grad1 = d.op.compute(d, [img, flt, grad])
	grad2 = e.op.compute(e, [img, flt, grad])
	print("grad:\n", grad1.shape)
	print(np.reshape(grad1, (-1, 4)))
	print("grad:\n", grad2.shape)
	print(np.reshape(grad2, (-1, 3)))
