NVCC = /usr/local/cuda-11.1/bin/nvcc
CC = g++

#Optimization flags. Don't use this for debugging.
#NVCCFLAGS = -c -m64 -O2 --compiler-options -Wall -Xptxas -O2,-v

#No optimizations. Debugging flags. Use this for debugging.
NVCCFLAGS = -c -g -G -m64 --compiler-options -Wall

OBJS = histo.o wrappers.o h_histo.o d_histo.o
.SUFFIXES: .cu .o .h 
.cu.o:
	$(NVCC) $(CC_FLAGS) $(NVCCFLAGS) $(GENCODE_FLAGS) $< -o $@

all: histo generate

histo: $(OBJS)
	$(CC) $(OBJS) -L/usr/local/cuda/lib64 -lcuda -lcudart -ljpeg -o histo

histo.o: histo.cu wrappers.h h_histo.h d_histo.h config.h histogram.h

h_histo.o: h_histo.cu h_histo.h CHECK.h config.h histogram.h

d_histo.o: d_histo.cu d_histo.h CHECK.h config.h histogram.h 

wrappers.o: wrappers.cu wrappers.h

generate: generate.c
	gcc -O2 generate.c -o generate -ljpeg

clean:
	rm generate histo *.o
