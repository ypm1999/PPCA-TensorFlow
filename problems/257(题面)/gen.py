#!/usr/bin/env python3
#使用前请先安装cyaron 库（pip install cyaron）
from cyaron import *
import os
import sys

def F(x, num):
	x = x / num
	return x * x

def main():
	std = "std"
	num = 10
	name = 'path'
	if(len(sys.argv) >= 2):
		std = sys.argv[1]
	std = './' + std
	name = './data/' + name
	os.system("rm -rf ./data && mkdir data")
	maxn = 100000
	for i in range(1, num + 1):
		print("gen %d" % i)
		path = name + str(i)
		io = IO(file_prefix = name, data_id = i)
		base_n = maxn * F(i, num)
		n = randint(int(base_n * 0.95), int(base_n))
		print(n)
		io.input_writeln(n)
		val = Sequence(lambda i, f: randint(1, 10000))
		io.input_writeln(val.get(1, n))
		tree = Graph.tree(n, 0.2, 0.5, weight_limit = (1, 10000))
		io.input_writeln(tree.to_str(shuffle = True))
		os.system(std + " < " + path + ".in > " + path + ".out")
		# os.system("./std1" + " < " + path + ".in > " + path + ".out1")
		# os.system("diff -w -b " + path + ".out " + path + ".out1 >> ./data/log")
		print("finished %d" % i)





if __name__ == '__main__':
	main()
