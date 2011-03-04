#include <cuda_runtime.h>
#include <cuda.h>
#include "benchmark.hpp"
#include "cuerr.hpp"
#include "medianfilterwudev.hpp"

TDV_NAMESPACE_BEGIN

void DevMedianFilterRun(const Dim &dim, float *input_d, float *output_d);

void MedianFilterWUDev::process()
{
  CUerrExp cuerr;  
  
  FloatImage inimg;
  while ( m_rpipe->read(&inimg) )
  {
      const Dim dim = inimg.dim();
      float *input_d = inimg.devMem();

      FloatImage outimg = FloatImage::CreateDev(dim);
      float *output_d = outimg.devMem();

      CudaBenchmarker marker;
      
      marker.begin();
      DevMedianFilterRun(dim, input_d, output_d);
      marker.end();
      
      m_wpipe.write(outimg);
  }
  
  m_wpipe.finish();
}

TDV_NAMESPACE_END
