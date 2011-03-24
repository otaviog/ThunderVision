#ifndef TDV_INPUTPROCESS_HPP
#define TDV_INPUTPROCESS_HPP

class InputProcess
{
public:
    virtual void init() = 0;
    
    virtual void dispose() = 0;
    
    virtual tdv::ReadPipe<IplImage*>* leftImgOuput() = 0;
    
    virtual tdv::ReadPipe<IplImage*>* rightImgOutput() = 0;
};

class InputProcessFactory
{
public:
    virtual InputProcess* create();
};

#endif /* TDV_INPUTPROCESS_HPP */
