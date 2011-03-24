#ifndef TDV_APPCONTEXT_HPP
#define TDV_APPCONTEXT_HPP

#include <string>

class AppContext
{
public:
    void init(const std::string &config);
    
    void dispose();
    
    void enableIntialStereoImages();
    
    void disableInitialStereoImages();
    
    void enableRectificationImages();
    
    void disableRectificationImages();
    
    void enableDisparityImages();
    
    void disableDisparityImages();
    
    void enableCalibrationContext();
    
    void disableCalibrationContext();
private:
};

#endif /* TDV_APPCONTEXT_HPP */
