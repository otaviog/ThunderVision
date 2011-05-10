#ifndef TDV_BMDATASETFACTORY_HPP
#define TDV_BMDATASETFACTORY_HPP

#include <tdvbasic/common.hpp>
#include <string>

TDV_NAMESPACE_BEGIN

class BMDataset;

class BMDatasetFactory
{
public:
    BMDataset* CreateDefault(const std::string &basePath);
};

TDV_NAMESPACE_END

#endif /* TDV_BMDATASETFACTORY_HPP */
