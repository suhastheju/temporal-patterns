MAKE = make
CC = g++
CFLAGS = -O5 -Wall -march=native -std=c++11 -fopenmp
BIN = BASE_SIN BASE_PAR

all: $(BIN)

BASE_PAR: base.c builds.h gf.h ffprng.h
	$(CC) $(CFLAGS) -DLISTER_DEFAULT -o BASE_PAR base.c

BASE_SIN: base.c builds.h gf.h ffprng.h
	$(CC) $(CFLAGS) -DLISTER_SINGLE -o BASE_SIN base.c

clean:	
	rm -f *.o *.a *~ $(BIN)
	rm -rf BASE_SIN.dSYM BASE_PAR.dSYM
	rm -f core*
