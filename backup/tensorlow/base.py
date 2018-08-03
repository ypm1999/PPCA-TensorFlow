#！/user/bin/env python3
# -*- coding:utf-8 -*-

import numpy as np

float16 = np.float16
float32 = np.float32
float64 = np.float64
float128 = np.float128

int8 = np.int8
int16 = np.int16
int32 = np.int32
int64 = np.int64

uint8 = np.uint8
uint16 = np.uint16
uint32 = np.uint32
uint64 = np.uint64

zeros = np.zeros
ones = np.ones

def random_normal(shape,mean=0.0,stddev=1.0,dtype=float32):
	return np.random.normal(mean, stddev, shape).astype(dtype)





