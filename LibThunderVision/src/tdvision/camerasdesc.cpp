#include <cstring>
#include "camerasdesc.hpp"

TDV_NAMESPACE_BEGIN

static void identity(double m[9])
{
    m[0] = 1.0; m[1] = 0.0; m[2] = 0.0;
    m[3] = 0.0; m[4] = 1.0; m[5] = 0.0;
    m[6] = 0.0; m[7] = 0.0; m[8] = 1.0;
}

CamerasDesc::CamerasDesc()
{
    identity(m_F);
    identity(m_extrinsicsR);
    
    m_extrinsicsT[0] = 0.0;
    m_extrinsicsT[1] = 0.0;
    m_extrinsicsT[2] = 0.0;
    
    m_hasF = false;
    m_hasExt = false;
}

void CamerasDesc::fundamentalMatrix(const double mtx[9])
{
    memcpy(m_F, mtx, sizeof(double)*9);
    m_hasF = true;
}

void CamerasDesc::extrinsics(const double R[9], const double T[3])
{
    memcpy(m_extrinsicsR, R, sizeof(double)*9);
    memcpy(m_extrinsicsT, T, sizeof(double)*3);
    m_hasExt = true;
}

TDV_NAMESPACE_END
