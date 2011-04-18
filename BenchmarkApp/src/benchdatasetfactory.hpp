#ifndef TDV_BENCHDATASETFACTORY_HPP
#define TDV_BENCHDATASETFACTORY_HPP

#include <tdvbasic/common.hpp>
#include <string>

TDV_NAMESPACE_BEGIN

class IBenchmarkDataset;

class BenchDatasetFactory
{
public:
    IBenchmarkDataset* CreateDefault(const std::string &basePath);
};

TDV_NAMESPACE_END

#endif /* TDV_BENCHDATASETFACTORY_HPP */
