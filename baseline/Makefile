MAKE = make
CC = gcc
CFLAGS = -O5 -Wall -march=native -std=c99 -fopenmp
BIN = BASE_SIN BASE_PAR

all: $(BIN)

BASE_PAR: base.c ffprng.h
	$(CC) $(CFLAGS) -DLISTER_DEFAULT -o BASE_PAR base.c

BASE_SIN: base.c ffprng.h
	$(CC) $(CFLAGS) -DLISTER_SINGLE -o BASE_SIN base.c

clean:	
	rm -f *.o *.a *~ $(BIN)
	rm -rf BASE_SIN.dSYM BASE_PAR.dSYM
	rm -f core*
