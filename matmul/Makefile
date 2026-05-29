CC      = gcc
NVC     = nvc
NVCC    = nvcc

CFLAGS    = -O2 -Wall -Wextra
ACCFLAGS  = -O2 -acc=gpu -mp=multicore -Minfo=accel,mp
CUDAFLAGS = -O2 -Xcompiler "-fopenmp -O2 -Wall"
CUDALIBS  = -lcublas -lgomp

N        ?= 1024
NSET     ?= 4
NSET_GPU ?= 2
NITER    ?= 10

DEFS = -DN=$(N) -DNSET=$(NSET) -DNSET_GPU=$(NSET_GPU) -DNITER=$(NITER)

TARGET      = matmul
TARGET_ACC  = matmul_acc
TARGET_ACC2 = matmul_acc2
TARGET_ACC3 = matmul_acc3
TARGET_CU   = matmul_cuda

all: $(TARGET)

$(TARGET): matmul.c
	$(CC) $(CFLAGS) -DN=$(N) -DNSET=$(NSET) -DNITER=$(NITER) -o $@ $< $(LDFLAGS)

acc: $(TARGET_ACC)
$(TARGET_ACC): matmul_acc.c
	$(NVC) $(ACCFLAGS) $(DEFS) -o $@ $<

acc2: $(TARGET_ACC2)
$(TARGET_ACC2): matmul_acc2.c
	$(NVC) $(ACCFLAGS) $(DEFS) -o $@ $<

acc3: $(TARGET_ACC3)
$(TARGET_ACC3): matmul_acc3.c
	$(NVC) $(ACCFLAGS) $(DEFS) -o $@ $<

cuda: $(TARGET_CU)
$(TARGET_CU): matmul_cuda.cu
	$(NVCC) $(CUDAFLAGS) $(DEFS) -o $@ $< $(CUDALIBS)

run: $(TARGET)
	./$(TARGET)

run-acc: $(TARGET_ACC)
	./$(TARGET_ACC)

run-cuda: $(TARGET_CU)
	./$(TARGET_CU)

clean:
	rm -f $(TARGET) $(TARGET_ACC) $(TARGET_ACC2) $(TARGET_ACC3) $(TARGET_CU)

.PHONY: all acc acc2 acc3 cuda run run-acc run-cuda clean
