

__global__ void dynamicprogSlice(float *dsi, float *lookUp, 
                                 dim2 slice, float *disp)
{
  for (int x=1; x<slice.x; x++) {
    for (int y=0; y<slice.y; y++) {
      float c = getVoxel(dsi, x, y);
      float c0 = getVoxel(lookUp, x - 1, y - 1);
      float c1 = getVoxel(lookUp, x - 1, y);
      float c2 = getVoxel(lookUp, x - 1, y + 1);      
      
      float m;
      
      if ( c0 < c1 && c0 < c2 ) {
        m = c0;
        setVoxel(path, x, y, -1);
      } else if ( c1 < c2 ) {
        m = c1;
        setVoxel(path, x, y, 0);
      } else {
        m = c2;
        setVoxel(path, x, y, 1);
      }
      
      float cost = c + m;
      setVoxel(lookUp, x, y);
      
    }
      
  }
}

__global__ void dynamicprog(float *dsi, dim3 dsiDim, float *disp)
{
  for (int 
}