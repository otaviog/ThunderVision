#ifndef TDV_TLCTX_H
#define TDV_TLCTX_H

#ifdef __cplusplus
#define CEXTERN extern "C"
#else
#define CEXTERN 
#endif

CEXTERN void tlcSetDistortion(void *tlc, const char *descId, int leftOrRight,
                              double d1, double d2, double d3, double d4, 
                              double d5);

CEXTERN void tlcSetIntrinsic(void *tlc, const char *descId, int leftOrRight, 
                             double mtx[9]);

CEXTERN void tlcSetExtrinsic(void *tlc, const char *descId, int leftOrRight,
                             double mtx[9]);

CEXTERN void tlcSetFundamental(void *tlc, const char *descId, double mtx[9]);

#endif /* TDV_TLCTX_H */
