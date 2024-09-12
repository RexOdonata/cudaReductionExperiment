#include "cuda_runtime.h"
#include "device_launch_parameters.h"


#include <vector>
#include <algorithm>
#include <numeric>
#include <iostream>
#include <chrono>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__global__ void BinaryReductionStep(int ** sections, const size_t increment, const size_t width)
{
    int * sectionA = sections[blockIdx.x];
    int * sectionB = sections[blockIdx.x + increment];

    for (int i=threadIdx.x; i<width; i+=blockDim.x)
    {
        sectionA[i] += sectionB[i];
    }
}

__global__ void RotatingReductionStep(int ** sections, const size_t increment, int * output)
{
    size_t srcSectionIndex = (blockIdx.x + increment) % gridDim.x;
    int * inputSection = sections[srcSectionIndex];

    size_t index = blockDim.x * blockIdx.x + threadIdx.x;

    output[index] += inputSection[index];
}

std::vector<int*> createDevData(std::vector<int>& srcData, const size_t sections)
{
    std::vector<int*> dev_data(sections);

    for (int i = 0; i < sections; i++)
    {
        gpuErrchk( cudaMalloc((void**)&dev_data[i], srcData.size() * sizeof(int)));
        gpuErrchk( cudaMemcpy(dev_data[i], srcData.data(), srcData.size() * sizeof(int), cudaMemcpyDefault));
    }

    return dev_data;
}

int** getDevDataSections(std::vector<int*> src)
{
    int ** dev_data;
    gpuErrchk( cudaMalloc((void**)&dev_data, src.size() * sizeof(int*)));
    gpuErrchk( cudaMemcpy(dev_data, src.data(), src.size() * sizeof(int*),cudaMemcpyDefault));

    return dev_data;
}

int main()
{
    const size_t N = 8192;

    const size_t sections = 16;

    const size_t blockwidth = 512;



    std::vector<int> srcData(N);
    std::iota(srcData.begin(), srcData.end(),1);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    auto dataB = createDevData(srcData, sections);
    auto dataB_sections = getDevDataSections(dataB);
    
    std::vector<int> resultB(N);
    float msB;
    {
        cudaEventRecord(start, stream);

        BinaryReductionStep<<<8, blockwidth,0, stream>>>(dataB_sections, 8, N);
        BinaryReductionStep<<<4, blockwidth,0, stream>>>(dataB_sections, 4, N);
        BinaryReductionStep<<<2, blockwidth,0, stream>>>(dataB_sections, 2, N);
        BinaryReductionStep<<<1, blockwidth,0, stream>>>(dataB_sections, 1, N);

        cudaEventRecord(end, stream);

        cudaEventSynchronize(end);

        cudaEventElapsedTime(&msB, start, end);

        cudaMemcpy(resultB.data(), dataB[0], srcData.size() * sizeof(int), cudaMemcpyDefault);
    }

    auto dataR = createDevData(srcData, sections);
    auto dataR_sections = getDevDataSections(dataR);

    std::vector<int> resultR(N);
    float msR;
    {
        int * dev_RO;
        cudaMalloc((void**)&dev_RO, srcData.size() * sizeof(int));
        cudaMemset(dev_RO, 0, srcData.size() * sizeof(int));

        cudaEventRecord(start, stream);

        for (int i= 0; i < sections; i++)
        {
            RotatingReductionStep<<<16, blockwidth,0, stream>>>(dataR_sections, i, dev_RO);
        }    

        cudaEventRecord(end, stream);

        cudaEventSynchronize(end);

        cudaEventElapsedTime(&msR, start, end);

        cudaMemcpy(resultR.data(), dev_RO, srcData.size() * sizeof(int), cudaMemcpyDefault);
    }

    if ( std::equal(resultR.begin(), resultR.end(), resultB.begin(), resultB.end() ) )
    {
        std::cout<<"Binary: "<<msR<<" - Rotating: "<<msB<<"\n";
    }
    else
    {
        std::cout<<"Fail\n";
        for (int i = 0; i < 12; i++)
        {
            std::cout<<"B"<<resultB.at(i)<<" - R"<<resultR.at(i)<<"\n";
        }
    }


    return 0;
}

