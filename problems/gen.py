#!/usr/bin/env python3
#使用前请先安装cyaron 库（pip install cyaron）
#from cyaron import *
import os
import sys

def main():
    std = "std"
    maker = "maker"
    num = 10
    name = '';
    if(len(sys.argv) >= 2):
        std = sys.argv[1];
    if(len(sys.argv) >= 3):
        maker = sys.argv[2];
    if(len(sys.argv) >= 4):
        num = int(sys.argv[3]);
    if(len(sys.argv) >= 5):
        name = sys.argv[4];
    if(len(sys.argv) > 4):
        printf("ERROR")
        return
    std = './' + std
    maker = './' + maker
    name = './data/' + name
    print(std)
    print(maker)
    print(name)
    os.system("rm -rf ./data && mkdir data")
    for i in range(1, num + 1):
        fileIn = name + str(i) + '.in'
        fileOut = name + str(i) + '.out'
        os.system(maker + ' > ' + fileIn)
        os.system(std + ' < ' + fileIn + ' > ' + fileOut)

if __name__ == '__main__':
    main()
