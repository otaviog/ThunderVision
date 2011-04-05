#include <tdvision/capturestereoinputsource.hpp>
#include "cmdline.hpp"

CMDLine::CMDLine(int argc, char **argv)
{
    namespace po = boost::program_options;

    try
    {
        po::options_description desc("Options");
        desc.add_options()
            ("help", "Displays the help message")
            ("left-img,l", po::value<std::string>(), "Left image input.")
            ("right-img,r", po::value<std::string>(), "Right image input.")
            ("left-vid,j", po::value<std::string>(), "Left video file input.")
            ("right-vid,k", po::value<std::string>(), "Right video file input.")
            ("cameras,c", "Use cameras.")
            ("spec,s", po::value<std::string>(), 
             "Spec filename describing the system preferences, in ThunderLang Language :D.");
    
        po::store(po::parse_command_line(argc, argv, desc), m_vm);
        po::notify(m_vm);
    }
    catch (const po::unknown_option &unknowOpt)
    {
        throw tdv::Exception(unknowOpt.what());
    }
           
}

tdv::StereoInputSource* CMDLine::createInputSource()
{
    tdv::StereoInputSource *inputSrc = NULL;
    
    if ( m_vm.count("left-img") && m_vm.count("right-img") )
    {
        
    }
    else if ( m_vm.count("left-vid") && m_vm.count("right-vid") )
    {
        tdv::CaptureStereoInputSource *cisrc = new tdv::CaptureStereoInputSource;
        cisrc->init(m_vm["left-vid"].as<std::string>(),
                       m_vm["right-vid"].as<std::string>());
        inputSrc = cisrc;        
    }
    else if ( m_vm.count("cameras") )
    {
        tdv::CaptureStereoInputSource *cisrc = new tdv::CaptureStereoInputSource;
        cisrc->init();
        inputSrc = cisrc;
    }

    return inputSrc;
}

tdv::ThunderSpec* CMDLine::createSpec()
{
    tdv::ThunderSpec *spec = NULL;
    if ( m_vm.count("spec") )
    {
        spec = new tdv::ThunderSpec;
        tdv::ThunderLangParser parser(*spec);
        try
        {
            parser.parseFile(m_vm["spec"].as<std::string>());
        }
        catch (const std::exception &ex)
        {
            delete spec;
            throw ex;
        }        
    }
    
    return spec;
}
