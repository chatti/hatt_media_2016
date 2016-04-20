cmake_minimum_required(VERSION 2.6)
CMAKE_MINIMUM_REQUIRED(VERSION 2.6)
IF(COMMAND CMAKE_POLICY)
  CMAKE_POLICY(SET CMP0003 NEW)
ENDIF(COMMAND CMAKE_POLICY)

Project(XUS)

set(VXL_DIR /usr/local/vxl/bin)
FIND_PACKAGE(VXL REQUIRED)
INCLUDE(${VXL_CMAKE_DIR}/UseVXL.cmake)

FIND_PACKAGE(CUDA REQUIRED)
INCLUDE(FindCUDA)

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} --gpu-architecture sm_21)


CUDA_ADD_EXECUTABLE(raycast mainRayCast.cxx RayCast.cxx kernelRayCast.cu)
CUDA_ADD_EXECUTABLE(patchraycast mainPatchRayCast.cxx PatchRayCast.cxx PatchRayCastCostFunction.cxx kernelPatchRayCast.cu)
CUDA_ADD_EXECUTABLE(dsc  mainDSC.cxx DSC.cxx kernelDSC.cu)
CUDA_ADD_EXECUTABLE(raycastDRR mainDRR.cxx RayCast.cxx kernelRayCast.cu)
CUDA_ADD_EXECUTABLE(hybrid mainHybrid.cxx PatchRayCast.cxx PatchRayCastCostFunction.cxx kernelPatchRayCast.cu DSC.cxx kernelDSC.cu)

TARGET_LINK_LIBRARIES(dsc vnl rrel)
TARGET_LINK_LIBRARIES(raycast vnl rrel)
TARGET_LINK_LIBRARIES(patchraycast vnl rrel)
TARGET_LINK_LIBRARIES(hybrid vnl rrel)


