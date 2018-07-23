#！/user/bin/env python3
# -*- coding:utf-8 -*-

import numpy as np


#TODO define types
float16 = np.float16
float32 = np.float32
float64 = np.float64
float80 = np.float80
float96 = np.float96
float128 = np.float128
float256 = np.float256


#TODO finish Session class
class Session:
    def __init__(self, target = '', graph = None, config = None):
        self._target = target
        self._graph = graph
        self._config = config

    def __enter__(self):
        pass

    def __exit__(self, exec_type, exec_val, exec_tb):
        print(exec_type)
        print(exec_val)
        print(exec_tb)

    def run(self, fetches, feed_dict = None, options = None, run_metadata = None):
        pass




class Node(object):

    def __init__(self):
        self.inputs = []
        self.op = None
        self.const_attr = None
        self.name = ""

    def __add__(self, other):
        if isinstance(other, Node):
            new_node = add_op(self, other)
        else:
            new_node = add_byconst_op(self, other)
        return new_node


    __radd__ = __add__
    __rmul__ = __mul__

    def __str__(self):
        """Allow print to display node name."""
        return self.name

    __repr__ = __str__

def Variable(name):
    #TODO
    pass

class Op(object):

    def __call__(self):
        new_node = Node()
        new_node.op = self
        return new_node

    def compute(self, node, input_vals):
        raise NotImplementedError

    def gradient(self, node, output_grad):
        raise NotImplementedError







class Executor:
    """Executor computes values for a given subset of nodes in a computation graph."""
    def __init__(self, eval_node_list):
        """
        Parameters
        ----------
        eval_node_list: list of nodes whose values need to be computed.
        """
        self.eval_node_list = eval_node_list

    def run(self, feed_dict):
        """Computes values of nodes in eval_node_list given computation graph.
        Parameters
        ----------
        feed_dict: list of variable nodes whose values are supplied by user.

        Returns
        -------
        A list of values for nodes in eval_node_list.
        """
        node_to_val_map = dict(feed_dict)
        # Traverse graph in topological sort order and compute values for all nodes.
        topo_order = find_topo_sort(self.eval_node_list)
        #TODO1: Your code here
        for node in topo_order:
            if node in node_to_val_map:
                continue
            val = []
            for input in node.inputs:
                val.append(node_to_val_map[input])
            node_to_val_map[node] = node.op.compute(node, val)
        # Collect node values.
        node_val_results = [node_to_val_map[node] for node in self.eval_node_list]
        return node_val_results

def gradients(output_node, node_list):
    """Take gradient of output node with respect to each node in node_list.

    Parameters
    ----------
    output_node: output node that we are taking derivative of.
    node_list: list of nodes that we are taking derivative wrt.

    Returns
    -------
    A list of gradient values, one for each node in node_list respectively.

    """

    # a map from node to a list of gradient contributions from each output node
    node_to_output_grads_list = {}
    # Special note on initializing gradient of output_node as oneslike_op(output_node):
    # We are really taking a derivative of the scalar reduce_sum(output_node)
    # instead of the vector output_node. But this is the common case for loss function.
    node_to_output_grads_list[output_node] = [oneslike_op(output_node)]
    # a map from node to the gradient of that node
    node_to_output_grad = {}
    # Traverse graph in reverse topological order given the output_node that we are taking gradient wrt.
    reverse_topo_order = reversed(find_topo_sort([output_node]))

    for node in reverse_topo_order:
        #print(node)
        grad_node = node_to_output_grads_list[node]
        grad = grad_node[0]
        for i in range(1, len(grad_node)):
            grad = add_op(grad, grad_node[i])
        node_to_output_grad[node] = grad
        input_grads = node.op.gradient(node, grad)
        if input_grads == None:
            continue
        for i in range(len(node.inputs)):
            if node_to_output_grads_list.get(node.inputs[i]) == None:
                node_to_output_grads_list[node.inputs[i]] = [input_grads[i]]
            else:
                node_to_output_grads_list[node.inputs[i]].append(input_grads[i])
            #print("%s, %s" % (node.inputs[i], input_grads[i]))

    # Collect results for gradients requested.
    grad_node_list = [node_to_output_grad[node] for node in node_list]
    return grad_node_list

##############################
####### Helper Methods #######
##############################

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
    for n in node.inputs:
        topo_sort_dfs(n, visited, topo_order)
    topo_order.append(node)

def sum_node_list(node_list):
    from operator import add
    from functools import reduce
    return reduce(add, node_list)
