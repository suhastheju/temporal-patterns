MAKE = make
CC = gcc
CFLAGS = -O3 -Wall -march=native -std=c99 -fopenmp

all: gen-count

gen-count: gen-count.c ffprng.h
	$(CC) $(CFLAGS) -o gen-count gen-count.c -lm

.PHONY: gen-count
clean:
	rm -f *~ gen-count
	rm -f *.o
