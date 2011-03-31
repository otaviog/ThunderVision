#ifndef TDV_CMDLINEGUI_HPP
#define TDV_CMDLINEGUI_HPP

#include <tdvision/stereoinputsource.hpp>
#include <tdvbasic/exception.hpp>
#include <boost/program_options.hpp>

class CMDLine
{
public:
    CMDLine(int argc, char **argv);
    
    tdv::StereoInputSource* createInputSource();
    
private:
    boost::program_options::variables_map m_vm;
};

#endif /* TDV_CMDLINEGUI_HPP */
