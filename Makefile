CC       = gcc
MPICC    = mpicc
CFLAGS   = -O2 -Wall -Wextra -std=c11
LDFLAGS  = -lm

BIN_DIR  = bin

.PHONY: all clean

all: $(BIN_DIR)/matmul_serial $(BIN_DIR)/matmul_mpi

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

$(BIN_DIR)/matmul_serial: src/matmul_serial.c | $(BIN_DIR)
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

$(BIN_DIR)/matmul_mpi: src/matmul_mpi.c | $(BIN_DIR)
	$(MPICC) $(CFLAGS) -o $@ $< $(LDFLAGS)

clean:
	rm -rf $(BIN_DIR)
