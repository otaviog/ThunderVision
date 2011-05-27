#include <math_constants.h>
#include "dim.hpp"
#include "dsimemutil.h"
#include "benchmark.hpp"
#include "cudaconstraits.hpp"

const int Nm = 12;
const float kOcc = 5;
const float kR = 25;

__global__ void dynamicprog(const DSIDim dim, const float *costDSI,
                            float *lastSumCost, char *pathDSI)
{
  const uint z = threadIdx.x;
  const uint y = blockIdx.x;

  if ( z >= dim.z || y >= dim.y  )
    return ;
  
  __shared__ float sharedCost[128];
  __shared__ int occs[128];
  
  const uint initialOff = dsiOffset(dim, 0, y, z);
  sharedCost[z] = costDSI[initialOff];
  occs[z] = 0;
  pathDSI[initialOff] = 0;      

  __syncthreads();  
  
  for (uint x=1; x<dim.x; x++) {        
    const uint c0Offset = dsiOffset(dim, x, y, z);  
    
    const float c0 = costDSI[c0Offset];    
    const float c2 = sharedCost[z];    
        
    const float c1 = (z > 0)
      ? sharedCost[z - 1]
      : CUDART_INF_F;
    
    const float c3 = ( z < (dim.z - 1) ) 
      ? sharedCost[z + 1]
      : CUDART_INF_F;
      
    float m;      
    char p;  
    if ( c1 < c2 && c1 < c3 ) {
      m = c1;
      p = -1;
      occs[z] += 1;
    } else if ( c2 < c3 ) {
      m = c2;
      p = 0;
    } else {
      m = c3;
      p = 1;
      occs[z] += 1;
    } 
          
    pathDSI[c0Offset] = p;
    
    //sharedCost[z] = c0 + m + occs[z]*kOcc + Nm*kR;

    if ( x % 24 == 0 ) {      
        occs[z] = 0;
      }
    
    sharedCost[z] = c0 + m;

    __syncthreads();
  }
  
  lastSumCost[dim.z*y + z] = sharedCost[z];
}

__global__ void reduceImage(const DSIDim dim, const float *lastSumCost, 
                            const char *pathDSI, float *dispImg)
{
  const uint y = blockIdx.x;
  if ( y >= dim.y )
    return ;
    
  int lastMinZ = 0;
  float min = lastSumCost[0];
  
  const uint lscBaseOff = y*dim.z;
  for (uint z=1; z < dim.z; z++) {
    const float sc = lastSumCost[lscBaseOff*y + z];
    if ( sc < min ) {
      lastMinZ = z;
      min = sc;
    }          
  }
  
  uint imgOffset = y*dim.x + (dim.x - 1);
  dispImg[imgOffset] = float(lastMinZ)/float(dim.z);
  
  for (int x = dim.x - 1; x >= 0; x--) {    
    const uint offset = dsiOffset(dim, x, y, lastMinZ);
    const char p = pathDSI[offset];
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
  float *lastSumCost = NULL;
  
  cuerr << cudaMalloc((void**) &lastSumCost, 
                      sizeof(float)
                      *tdv_dsiDim.depth()
                      *tdv_dsiDim.height());
                      
  char *pathDSI = NULL;
  size_t pathSize = tdv_dsiDim.size();  
  
  cuerr = cudaMalloc((void**) &pathDSI, pathSize*sizeof(int));  
  if ( !cuerr.good() ) {
    cudaFree(lastSumCost);
    cuerr.checkErr();
    return ;
  }

  tdv::CudaBenchmarker bm;
  bm.begin();
  
  dynamicprog<<<tdv_dsiDim.height(), tdv_dsiDim.depth()>>>(dsiDim, dsi, lastSumCost, pathDSI);
  
  cuerr = cudaThreadSynchronize();
  
  if ( !cuerr.good() ) {
    cudaFree(lastSumCost);
    cudaFree(pathDSI);
    cuerr.checkErr();
    return ;
  }
  
  reduceImage<<<tdv_dsiDim.height(), 1>>>(dsiDim, lastSumCost, pathDSI, dispImg);
  
  bm.end();
  
  cuerr = cudaFree(lastSumCost);  
  cuerr = cudaFree(pathDSI);  
  cuerr.checkErr();  
}