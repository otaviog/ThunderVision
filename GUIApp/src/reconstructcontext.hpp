#ifndef TDV_RECONSTRUCTCONTEXT_HPP
#define TDV_RECONSTRUCTCONTEXT_HPP

#include <vector>

class StereoMatcher
{
public:
    
};

class StereoMatcherDev
{
public:
    
private:
    tdv::MacthingCostDev *m_matchCost;
    tdv::OptimizationDev *m_optDev;
};

class StereoMatcherCPU
{
public:
    
};

class ReconstructionContext
{
public:        
    
    void leftInput(tdv::ReadPipe<IplImage*> *leftImage);
    
    void rightInput(tdv::ReadPipe<IplImage*> *rightImage);
                   
    void init();
    
    void dispose();
        
    void enableRectificationImages();
    
    void disableRectificationImages();
    
    void enableDisparityImages();
    
    void disableDisparityImages();
    
    void enableCalibrationContext();
    
    void disableCalibrationContext();
        
private:    
    tdv::WorkUnitProcess<tdv::RectificationCV> m_rectify;
    tdv::WorkUnitProcess<tdv::FloatImageConv> m_floatConv[2];
    
    std::vector<tdv::ImageFilter*> m_preFilters;
        
    tdv::StereoMacther m_stereoMatcher;    
    tdv::ProcessRunner *m_procRunner;
};

#endif /* TDV_RECONSTRUCTCONTEXT_HPP */
