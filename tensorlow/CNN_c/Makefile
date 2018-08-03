TARGET=nn
SRCS = library.cpp

OBJS = $(SRCS:.c=.o)

$(TARGET):
	g++ -std=c++14 -shared -O2 -fPIC -o nn_c.so library.cpp  libopenblas.a -lpthread
	mv -f ./nn_c.so ../
