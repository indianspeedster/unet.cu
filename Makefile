

# Compiler flags
CFLAGS = -O3 --use_fast_math
NVCCFLAGS = -lcublas -lcublasLt
# Conditional debug flags
DEBUG_FLAGS = -g -G
PROFILE_FLAGS = -g -lineinfo

# Check for debug mode
ifeq ($(DEBUG),1)
	NVCCFLAGS += $(DEBUG_FLAGS)
endif

ifeq ($(PROFILE),1)
	NVCCFLAGS += $(PROFILE_FLAGS)
endif

# AMD SUPPORT
BUILD_DIR = build

# AMD flags
ROCM_PATH ?= /opt/rocm
AMDGPU_TARGETS ?= $(shell $(ROCM_PATH)/llvm/bin/amdgpu-offload-arch)
HIPCC := $(shell which hipcc 2>/dev/null)
HIPIFY := $(shell which hipify-perl 2>/dev/null)
HIPCC_FLAGS = -O3 -march=native -I$(BUILD_DIR)
HIPCC_FLAGS += $(addprefix --offload-arch=,$(AMDGPU_TARGETS))
HIPCC_LDFLAGS = -lhipblas -lhipblaslt -lamdhip64

HIPCC_FLAGS += -I./build/hip
HIPCC_FLAGS += -DBUILD_AMD

HIPCC_FLAGS += -I/opt/rocm/include/hipblas
ifneq ($(filter gfx1100,$(AMDGPU_TARGETS)),)
  HIPCC_LDFLAGS += -ldevice_gemm_operations -lutility -ldevice_other_operations
else
  HIPCC_FLAGS += -DDISABLE_CK
endif
ifdef DISABLE_CK
  HIPCC_FLAGS += -DDISABLE_CK
endif
ifdef WAVEFRONTSIZE64
  HIPCC_FLAGS += -DWAVEFRONTSIZE64 -mwavefrontsize64
endif
ifdef CUMODE
  HIPCC_FLAGS += -mcumode
endif
ifneq ($(NO_MULTI_GPU), 1)
  ifeq ($(shell [ -d /usr/lib/x86_64-linux-gnu/openmpi/lib/ ] && [ -d /usr/lib/x86_64-linux-gnu/openmpi/include/ ] && echo "exists"), exists)
    HIPCC_FLAGS += -I/usr/lib/x86_64-linux-gnu/openmpi/include -DMULTI_GPU
    HIPCC_LDFLAGS += -L/usr/lib/x86_64-linux-gnu/openmpi/lib/ -lmpi -lrccl
  endif
endif

train_unet: train_unet.cu
	$(NVCC) $(CFLAGS) $(NVCCFLAGS) $^ -o $@

clean: rm -f train_unet

$(BUILD_DIR)/hip/%h: %h
	@mkdir -p $(dir $@)
	$(HIPIFY) -quiet-warnings $< -o $@

AMD_HEADERS = $(addprefix $(BUILD_DIR)/hip/,$(wildcard *h))

amd_headers: $(AMD_HEADERS)

$(BUILD_DIR)/hip/%.cu: %.cu
	@mkdir -p $(dir $@)
	$(HIPIFY) -quiet-warnings $< -o $@

%amd: $(BUILD_DIR)/hip/%.cu amd_headers
	$(HIPCC) $(HIPCC_FLAGS) $(PFLAGS) $< $(HIPCC_LDFLAGS) -o $@