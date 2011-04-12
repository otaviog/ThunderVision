#include "dim.hpp"
#include "dsimem.hpp"
#include "dsimemutil.h"
#include "cudaconstraits.hpp"

__global__ void dynamicprog(const DSIDim dim, const float *costDSI,
                            float *sumCostDSI, int *pathDSI)
{
  uint z = threadIdx.x + blockIdx.x*blockDim.x;
  uint y = threadIdx.y + blockIdx.y*blockDim.y;
  if ( dim.z <= z || dim.y <= y )
    return ;
  
  const uint initialOff = dsiOffset(dim, 0, y, z);
  sumCostDSI[initialOff] = costDSI[initialOff];
  
  __syncthreads();

  for (uint x=1; x<dim.x; x++) {    
    const uint c0Offset = dsiOffset(dim, x, y, z);  
    const float c0 = costDSI[c0Offset];
      
    /**
     * c1\
     * c2-c0
     * c3/
     */  
    const float c1 = dsiIntensityClamped(dim, x - 1, y, z - 1, sumCostDSI);
    const float c2 = dsiIntensityClamped(dim, x - 1, y, z, sumCostDSI);
    const float c3 = dsiIntensityClamped(dim, x - 1, y, z + 1, sumCostDSI);      
      
    float m;      
    char p;  
    if ( c1 < c2 && c1 < c3 ) {
      m = c1;
      p = 1;
    } else if ( c2 < c3 ) {
      m = c2;
      p = 0;
    } else {
      m = c3;
      p = -1;
    }
      
    sumCostDSI[c0Offset] = c0 + m;
    pathDSI[c0Offset] = p;
    
    __syncthreads();
  }
}

__global__ void reduceImage(const DSIDim dim, const float *sumCostDSI, 
                            const int *pathDSI, float *dispImg)
{
  const uint y = threadIdx.x + blockIdx.x*blockDim.x; // slice

  if ( y >= dim.y )
    return ;
  
  uint lastMinZ = 0;
  float min = dsiIntensityClamped(dim, dim.x - 1, y, 0, sumCostDSI);
  
  for (uint z=1; z < dim.z; z++) {
    const float sc = dsiIntensityClamped(dim, dim.x - 1, y, z, sumCostDSI);    
    if ( sc < min ) {
      lastMinZ = z;
      min = sc;
    }          
  }
  
  uint imgOffset = y*dim.x + dim.x - 1;
  dispImg[imgOffset] = float(lastMinZ)/float(dim.z);
    
  for (int x=dim.x - 2; x >= 0; x--) {
    const uint offset = dsiOffset(dim, x, y, lastMinZ);
    const uint nz = lastMinZ + pathDSI[offset];
    
    if ( nz < dim.maxOffset )
      lastMinZ = nz;
    
    imgOffset = y*dim.x + x;
    dispImg[imgOffset] = float(lastMinZ)/float(dim.z);    
  }
}

void RunDynamicProgDev(const tdv::Dim &tdv_dsiDim, float *dsi, float *dispImg)
{  
  tdv::CUerrExp cuerr;
  
  DSIDim dsiDim(DSIDimCreate(tdv_dsiDim));  
  tdv::DSIMem sumCostDSI = tdv::DSIMem::Create(tdv_dsiDim);
 
  int *pathDSI;
  cuerr << cudaMalloc((void**) &pathDSI, tdv_dsiDim.size()*sizeof(int));
  
  tdv::CudaConstraits constraits;  
  tdv::WorkSize ws = constraits.imageWorkSize(tdv::Dim(tdv_dsiDim.depth(), 
                                                       tdv_dsiDim.width()));  
  
  dynamicprog<<<tdv_dsiDim.height(), 512>>>(dsiDim, dsi, sumCostDSI.mem(), pathDSI);  
  reduceImage<<<tdv_dsiDim.height(), 1>>>(dsiDim, sumCostDSI.mem(), pathDSI, dispImg);
  
  cudaFree(pathDSI);  
}