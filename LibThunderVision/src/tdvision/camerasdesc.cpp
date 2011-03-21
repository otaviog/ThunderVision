#include <cstring>
#include "camerasdesc.hpp"

TDV_NAMESPACE_BEGIN

CamerasDesc::CamerasDesc()
{
}

void CamerasDesc::fundamentalMatrix(const double mtx[9])
{
    memcpy(m_F, mtx, sizeof(double)*9);
}

TDV_NAMESPACE_END
