
model = CreateModelGPU(2^20);save data/pcloud.model.20.mat model;
model = CreateModelGPU(2^18);save data/pcloud.model.18.mat model;
model = CreateModelGPU(2^16);save data/pcloud.model.16.mat model;
drrModel = CreateModelGPU_DRR(2^20); save data/drrModel.mat drrModel;

