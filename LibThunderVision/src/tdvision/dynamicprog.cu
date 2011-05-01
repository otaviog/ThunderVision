#include "dim.hpp"
#include "dsimem.hpp"
#include "dsimemutil.h"
#include "benchmark.hpp"
#include "cudaconstraits.hpp"

#define SINGLE_KERNEL 1

#ifdef SINGLE_KERNEL
__global__ void dynamicprog(const DSIDim dim, const float *costDSI,
                            float *sumCostDSI, int *pathDSI)
{
  //uint z = threadIdx.x + blockIdx.x*blockDim.x;
  //uint y = threadIdx.y + blockIdx.y*blockDim.y;
  const uint z = threadIdx.x;
  const uint y = blockIdx.x;

  if ( z >= dim.z || y >= dim.y  )
    return ;
  
  const uint initialOff = dsiOffset(dim, 0, y, z);
  sumCostDSI[initialOff] = costDSI[initialOff];
  sumCostDSI[initialOff] = 0;
  __threadfence();
  pathDSI[initialOff] = 0;    
  __syncthreads();
  
  __shared__ float sharedCost[32];    
  for (uint x=0; x<dim.x; x++) {    
    
    const uint c0Offset = dsiOffset(dim, x, y, z);  
    const float c0 = costDSI[c0Offset];
    float c2;
    if ( x > 0 )
      c2 = sumCostDSI[dsiOffset(dim, x - 1, y, z)];
    else
      c2 = c0;
    
    sharedCost[z] = c2;
    __syncthreads();
    
    float c1, c3;
    if ( z > 0 )
      c1 = sharedCost[z - 1];
    else 
      c1 = DSI_HIGHDSI_VALUE;
    
    if ( z < dim.z - 1 ) 
      c3 = sharedCost[z + 1];
    else
      c3 = DSI_HIGHDSI_VALUE;
      
    float m;      
    int p;  
    if ( c1 < c2 && c1 < c3 ) {
      m = c1;
      p = 1;
    } else if ( c2 < c3 ) {
      m = c2;
      p = 0;
    } else { //if ( c3 < c2 && c3 < c1 ){
      m = c3;
      p = -1;
    } 
            
    sumCostDSI[c0Offset] = c0 + m;
    pathDSI[c0Offset] = p;
    
    __syncthreads();
  }
}
#else
__global__ void dynamicprog(int x, const DSIDim dim, const float *costDSI,
                            float *sumCostDSI, int *pathDSI)
{
  //uint z = threadIdx.x + blockIdx.x*blockDim.x;
  //uint y = threadIdx.y + blockIdx.y*blockDim.y;
  const uint z = threadIdx.x;
  const uint y = blockIdx.x;

  if ( z >= dim.z || y >= dim.y  )
    return ;
  
  if ( x == 0 )
    {
  const uint initialOff = dsiOffset(dim, x, y, z);
  sumCostDSI[initialOff] = costDSI[initialOff];
  pathDSI[initialOff] = 0;
    }
  else
    {    
    const uint c0Offset = dsiOffset(dim, x, y, z);  
    const float c0 = costDSI[c0Offset];
      
    /**
     * c1\
     * c2-c0
     * c3/
     */  
    const float c1 = dsiIntensityClamped(dim, x - 1, y, z + 1, sumCostDSI);
    const float c2 = dsiIntensityClamped(dim, x - 1, y, z, sumCostDSI);
    const float c3 = dsiIntensityClamped(dim, x - 1, y, z - 1, sumCostDSI);
    
    float m;      
    int p;  
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
    }

}
#endif

__global__ void reduceImage(const DSIDim dim, const float *sumCostDSI, 
                            const int *pathDSI, float *dispImg)
{
  //const uint y = threadIdx.x + blockIdx.x*blockDim.x; // slice
  const uint y = blockIdx.x;
  if ( y >= dim.y )
    return ;
  
  int lastMinZ = 0;
  float min = dsiIntensityClamped(dim, dim.x - 1, y, 0, sumCostDSI);
  
  for (uint z=1; z < dim.z; z++) {
    const float sc = dsiIntensityClamped(dim, dim.x - 1, y, z, sumCostDSI);    
    if ( sc < min ) {
      lastMinZ = z;
      min = sc;
    }          
  }
  
  uint imgOffset = y*dim.x + (dim.x - 1);
  dispImg[imgOffset] = float(lastMinZ)/float(dim.z);
  
  for (uint _x=0; _x < dim.x - 1; _x++) {
    const uint x = dim.x - 2 - _x;
    const uint offset = dsiOffset(dim, x, y, lastMinZ);
    const int p = pathDSI[offset];
    const uint nz = lastMinZ + p;
    
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
  tdv::CudaBenchmarker bm;
  bm.begin();
#if SINGLE_KERNEL
  dynamicprog<<<tdv_dsiDim.height(), tdv_dsiDim.depth()>>>(dsiDim, dsi, sumCostDSI.mem(), pathDSI);  
#else
  for (size_t x=0; x<tdv_dsiDim.width(); x++) {
    dynamicprog<<<tdv_dsiDim.height(), tdv_dsiDim.depth()>>>(x, dsiDim, dsi, sumCostDSI.mem(), pathDSI);  
    cuerr << cudaThreadSynchronize();
  }
#endif
  bm.end();
  cuerr << cudaThreadSynchronize();
  
  reduceImage<<<tdv_dsiDim.height(), 1>>>(dsiDim, sumCostDSI.mem(), pathDSI, dispImg);
  
  cuerr << cudaFree(pathDSI);  
}