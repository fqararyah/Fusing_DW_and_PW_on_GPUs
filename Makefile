DATA_TYPE=int8
MODEL_ID=mob_v1
$(DATA_TYPE)_exes/$(MODEL_ID)_$(DATA_TYPE).out: ./*.cpp \
	./fcm_and_lbl_kernels/$(DATA_TYPE)/*.cu \
	./utils/*.cpp \
	./cpu_imps/*.cpp \
	./model_specs/*.cpp
	nvcc -O3 -gencode=arch=compute_75,code=sm_75 -o ./$(DATA_TYPE)_exes/$(MODEL_ID)_$(DATA_TYPE).out \
	./*.cpp \
	./fcm_and_lbl_kernels/$(DATA_TYPE)/*.cu \
	./utils/*.cpp \
	./cpu_imps/*.cpp \
	./model_specs/*.cpp

