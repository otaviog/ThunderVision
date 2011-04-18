#ifndef TDV_IBENCHMARKDATASET_HPP
#define TDV_IBENCHMARKDATASET_HPP

#include <list>
#include <tdvbasic/common.hpp>
#include <tdvision/floatimage.hpp>

TDV_NAMESPACE_BEGIN

class StereoPair
{
public:
    class Sample
    {
    public:
        Sample(FloatImage limg, FloatImage rimg,
               FloatImage gimg)
        { 
            m_limg = limg;
            m_rimg = rimg;
            m_gimg = gimg;
        }
        
        FloatImage leftImage()
        {
            return m_limg;
        }

        FloatImage rightImage()
        {
            return m_rimg;
        }

        FloatImage groundTruth()
        {
            return m_gimg;
        }
        
        size_t width() const
        {
            return m_limg.dim().width();
        }
        
        size_t height() const
        {
            return m_limg.dim().height();
        }
                
    private:
        FloatImage m_limg, m_rimg, m_gimg;
    };
    
    typedef std::list<StereoPair::Sample> SampleList;
    
    StereoPair(const std::string &name)
        : m_name(name)
    {
    }
    
    SampleList::iterator samplesBegin()
    {
        return m_samples.begin();
    }
    
    SampleList::iterator samplesEnd()
    {
        return m_samples.begin();
    }
    
    void addSample(Sample sample)
    {
        m_samples.push_back(sample);
    }
    
    const std::string& name() const
    {
        return m_name;
    }

private:
    SampleList m_samples;
    std::string m_name;
};

class IBenchmarkDataset
{
public:
    virtual size_t stereoPairCount() = 0;
    
    virtual StereoPair* stereoPair(size_t idx) = 0;
    
private:
};

TDV_NAMESPACE_END

#endif /* TDV_IBENCHMARKDATASET_HPP */
