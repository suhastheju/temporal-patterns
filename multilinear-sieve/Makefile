MAKE = make
CC = gcc
CFLAGS = -O5 -Wall -march=native -std=c99 -fopenmp

all: LISTER_SIN LISTER_PAR

LISTER_PAR: lister.c builds.h gf.h ffprng.h
	$(CC) $(CFLAGS) -DLISTER_DEFAULT -o LISTER_PAR lister.c

LISTER_SIN: lister.c builds.h gf.h ffprng.h
	$(CC) $(CFLAGS) -DLISTER_SINGLE -o LISTER_SIN lister.c

clean:	
	rm -f *.o *.a *~ gen-count gen-unique LISTER_SIN LISTER_PAR
