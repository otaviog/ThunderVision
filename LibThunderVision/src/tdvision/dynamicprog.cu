#include <math_constants.h>
#include "cuerr.hpp"
#include "dim.hpp"
#include "dsimemutil.h"
#include "cudaconstraits.hpp"

#define MAX_DISP 2048

__global__ void dynamicprog(const dim3 dsiDim, cudaPitchedPtr costDSI,
                            cudaPitchedPtr pathDSI, float *lastSumCost)
{
  const ushort z = threadIdx.x;
  const ushort y = blockIdx.x;
  const ushort dz = z + 1;
  
  __shared__ float sharedCost[MAX_DISP + 2];
  __shared__ float *costDsiRow, *pathDsiRow;
      
  sharedCost[z] = dsiGetRow(costDSI, dsiDim.x, 0, y)[z];
  dsiGetRow(pathDSI, dsiDim.x, 0, y)[z] = 0;

  if ( z == 0 ) {
    costDsiRow = dsiGetRow(costDSI, dsiDim.x, y, z);
    pathDsiRow = dsiGetRow(pathDSI, dsiDim.x, y, z);
    
    sharedCost[0] = CUDART_INF_F;
    sharedCost[dsiDim.z + 1] = CUDART_INF_F;
  }

  __syncthreads();  
  
  for (ushort x=1; x<dsiDim.x; x++) {                
    const float c0 = costDsiRow[z];
    
    const float c2 = sharedCost[dz];            
    const float c1 = sharedCost[dz - 1];    
    const float c3 = sharedCost[dz + 1];
      
    float m;      
    char p;  
    if ( c1 < c2 && c1 < c3 ) {
      m = c1;
      p = -1;
    } else if ( c2 < c3 ) {
      m = c2;
      p = 0;
    } else {
      m = c3;
      p = 1;
    } 
    
    pathDsiRow[z] = p;
    sharedCost[dz] = c0 + m;

    __syncthreads();
  }
  
  lastSumCost[dsiDim.z*y + z] = sharedCost[dz];
}

__global__ void reduceImage(const dim3 dsiDim, 
                            const cudaPitchedPtr pathDSI, 
                            const float *lastSumCost,
                            float *dispImg)
{
  const uint y = blockIdx.x;
    
  int lastMinZ = 0;
  float min = lastSumCost[0];
  
  const uint lscBaseOff = y*dsiDim.z;
  for (uint z=1; z < dsiDim.z; z++) {
    const float sc = lastSumCost[lscBaseOff*y + z];
    
    if ( sc < min ) {
      lastMinZ = z;
      min = sc;
    }          
  }
  
  float *imgRow = &dispImg[y*dsiDim.x];  
  imgRow[dsiDim.x - 1] = float(lastMinZ)/float(dsiDim.z);  
  
  for (int x = dsiDim.x - 1; x >= 0; x--) {            
    const char p = dsiGetValueB(pathDSI, dsiDim.x, x, y, lastMinZ);        
    const int nz = lastMinZ + p;
    
    if ( nz >= 0 && nz < dsiDim.z ) {
      lastMinZ = nz;  
    }        
    
    imgRow[x] = float(lastMinZ)/float(dsiDim.z);
  }
  
}

TDV_NAMESPACE_BEGIN

void DynamicProgDevRun(const tdv::Dim &dsiDim, 
                       const cudaPitchedPtr costDSI,
                       cudaPitchedPtr pathDSI,
                       float *lastSumCosts,
                       float *dispImg)
{  
  CUerrExp cuerr;  
                      
  dynamicprog<<<dsiDim.height(), 
    dsiDim.depth()>>>(tdvDimTo(dsiDim), costDSI, pathDSI, lastSumCosts);
  
  cuerr = cudaThreadSynchronize();
    
  reduceImage<<<dsiDim.height(), 1>>>(tdvDimTo(dsiDim), 
                                      pathDSI, lastSumCosts, 
                                      dispImg);      
}

TDV_NAMESPACE_END