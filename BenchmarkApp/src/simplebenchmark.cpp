#include <iostream>
#include <boost/scoped_ptr.hpp>
#include <tdvision/commonstereomatcherfactory.hpp>
#include <tdvision/stereomatcher.hpp>
#include <tdvbasic/exception.hpp>
#include <boost/program_options.hpp>

#include <tdvision/imagereader.hpp>
#include <tdvision/imagewriter.hpp>
#include <tdvision/floatconv.hpp>
#include <tdvision/rgbconv.hpp>
#include <tdvision/process.hpp>
#include <tdvision/processgroup.hpp>
#include <tdvision/processrunner.hpp>
#include <tdvision/exceptionreport.hpp>
#include <tdvision/workunitprocess.hpp>

class ErrorHandler: public tdv::ExceptionReport
{
public:
    void errorOcurred(const std::exception &err)
    {
        std::cout<<err.what()<<std::endl;
    }
};

static bool commandLine(
    int argc, char *argv[],
    tdv::StereoMatcher **matcher,
    std::string &linput,
    std::string &rinput,
    std::string &output)
{     
    namespace po = boost::program_options;
    po::variables_map vm;
    
    try
    {

    po::options_description desc("Options");
    desc.add_options()
        ("help", "Display help message.")
        ("gpu", "Use GPU device computation.")
        ("cpu", "Use CPU device computation.")
        ("ssd", "Use SSD as metric.")
        ("bt", "Use Birchfield Tomasi as metric")
        ("xcorr", "Use Cross Correlation as metric")
        ("local", "Use local method.")
        ("semiglobal", "Use Semiglobal")
        ("global", "Use global method (Graph Cuts by OpenCV).")
        ("wta", "Use Winner Take All as optimization.")
        ("dyp", "Use Dynamic Programming as optimization.")
        ("disparity", po::value<int>(), "Maximum disparity.")
        ("inputl", po::value<std::string>(), "Left input image.")
        ("inputr", po::value<std::string>(), "Right input image.")
        ("output", po::value<std::string>(), "Output disparity image.");
    
        po::store(po::parse_command_line(argc, argv, desc), vm);
        po::notify(vm);
    }
    catch ( const po::unknown_option &unkOpt)
    {
        std::cout << unkOpt.what() << std::endl;
        return false;
    }
    
    if ( vm.count("inputl") && vm.count("inputr") && vm.count("output") )
    {
        linput = vm["inputl"].as<std::string>();
        rinput = vm["inputr"].as<std::string>();
        output = vm["output"].as<std::string>();
    }
    else
    {
        std::cout << "No input and output given" << std::endl;
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
        else if ( vm.count("bt") )
        {
            factory.matchingCost(tdv::CommonStereoMatcherFactory::BirchfieldTomasi);
        }
        else if ( vm.count("xcorr") )
        {
            factory.matchingCost(tdv::CommonStereoMatcherFactory::CrossCorrelationNorm);
        }
        
        if ( vm.count("wta") || vm.count("local") )
        {
            factory.optimization(tdv::CommonStereoMatcherFactory::WTA);
        }
        else if ( vm.count("dyp") )
        {
            factory.optimization(tdv::CommonStereoMatcherFactory::DynamicProg);
        }      
        else if ( vm.count("semiglobal") )
        {
            factory.optimization(tdv::CommonStereoMatcherFactory::SemiGlobal);
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
    std::string linput, rinput, output;
    
    tdv::StereoMatcher *matcher;    
    if ( !commandLine(argc, argv, &matcher, linput, rinput, output) )
        return 1;            
    
    tdv::ImageReader readerL(linput);
    tdv::ImageReader readerR(rinput);    
    tdv::ImageWriter writer(output);
    
    try 
    {
        readerL.update();
        readerR.update();
        readerL.close();
        readerR.close();
    }
    catch (const std::exception &ex)
    {
        std::cout << ex.what() << std::endl;
        delete matcher;
        return 1;
    }
        
    tdv::TWorkUnitProcess<tdv::FloatConv> fconvL, fconvR;
    tdv::TWorkUnitProcess<tdv::RGBConv> rconv;
    
    fconvL.input(readerL.output());
    fconvR.input(readerR.output());
    
    matcher->inputs(fconvL.output(), fconvR.output());    
    
    rconv.input(matcher->output());    
    writer.input(rconv.output());
    
    tdv::ArrayProcessGroup procs;    
    
    procs.addProcess(&fconvL);
    procs.addProcess(&fconvR);
    procs.addProcess(*matcher);
    procs.addProcess(&rconv);
    
    ErrorHandler errHdl;
    tdv::ProcessRunner runner(procs, &errHdl);    
    
    runner.run();
    runner.join();
    
    try
    {
        writer.update();
    }
    catch (const std::exception &ex)
    {
        std::cout << ex.what() << std::endl;
    }

    std::cout<<
        matcher->matchcostBenchmark().millisecs()
        + matcher->optimizationBenchmark().millisecs()
             <<std::endl;
    delete matcher;
    
    return 0;
}

