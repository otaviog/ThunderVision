#include <iostream>
#include <tdvision/commonstereomatcherfactory.hpp>
#include <tdvbasic/exception.hpp>
#include <boost/program_options.hpp>
#include "meanqmetric.hpp"
#include "bmdataset.hpp"
#include "bmdatasetfactory.hpp"
#include "bmrunner.hpp"

static bool commandLine(
    int argc, char *argv[],
    tdv::StereoMatcher **matcher)
{     
    namespace po = boost::program_options;
    po::variables_map vm;
    
    try
    {
    po::options_description desc("Options");
    desc.add_options()
        ("help", "Display help message.")
        ("gpu,g", "Use GPU device computation.")
        ("cpu,c", "Use CPU device computation.")
        ("ssd,s", "Use SSD as metric.")
        ("local,l", "Use local method.")
        ("global,b", "Use global method (Graph Cuts by OpenCV).")
        ("wta,w", "Use Winner Take All as optimization.")
        ("dyp,y", "Use Dynamic Programming as optimization.")
        ("disparity,d", po::value<int>(), "Maximum disparity.");
        po::store(po::parse_command_line(argc, argv, desc), vm);
        po::notify(vm);
    }
    catch ( const po::unknown_option &unkOpt)
    {
        std::cout << unkOpt.what() << std::endl;
        return false;
    }
    
    tdv::CommonStereoMatcherFactory factory;
    if ( vm.count("gpu") )
    {
        factory.computeDev(tdv::CommonStereoMatcherFactory::Device);
        
        if ( vm.count("ssd") )
        {
            factory.matchingCost(tdv::CommonStereoMatcherFactory::SSD);
        }
        
        if ( vm.count("wta") )
        {
            factory.optimization(tdv::CommonStereoMatcherFactory::WTA);
        }
        else if ( vm.count("dyp") )
        {
            factory.optimization(tdv::CommonStereoMatcherFactory::DynamicProg);
        }        
    }
    else if ( vm.count("cpu") )
    {
        factory.computeDev(tdv::CommonStereoMatcherFactory::CPU);
        
        if ( vm.count("global") )
        {
            factory.optimization(tdv::CommonStereoMatcherFactory::Global);
        }
        else if ( vm.count("local") )
        {
            factory.optimization(tdv::CommonStereoMatcherFactory::WTA);
        }
    }    
    
    if ( vm.count("disparity") )
        factory.maxDisparity(vm["disparity"].as<int>());
    
    *matcher = factory.createStereoMatcher();
    
    return true;
}

int main(int argc, char *argv[])
{        
    tdv::StereoMatcher *matcher;
    if ( !commandLine(argc, argv, &matcher) )
        return 1;        

    tdv::BMDataset *dataset = NULL;    
    try
    {
        tdv::BMDatasetFactory dsetFactory;
        dataset = dsetFactory.CreateDefault(
            "../../res/Benchmark");        
    }
    catch (const tdv::Exception &ex)
    {
        std::cout << ex.what() << std::endl;
        return 1;
    }
    
    tdv::QualityMetric *metric = new tdv::MeanQMetric;    
    
    tdv::BMRunner brunner(matcher, dataset, metric);
    brunner.run();
    
    return 0;
}
