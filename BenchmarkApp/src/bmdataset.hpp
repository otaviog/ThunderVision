#ifndef TDV_BMDATASET_HPP
#define TDV_BMDATASET_HPP

#include <list>
#include <tdvbasic/common.hpp>
#include <tdvision/floatimage.hpp>
#include <tdvision/dim.hpp>

TDV_NAMESPACE_BEGIN

namespace bmdata
{
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

        const Dim& dim()
        {
            return m_limg.dim();
        }
                
    private:
        FloatImage m_limg, m_rimg, m_gimg;
    };
 
    class StereoPair
    {
    public:    
        typedef std::list<Sample> SampleList;
    
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
            return m_samples.end();
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
    
    class SampleReport
    {
    public:
        SampleReport(double quali, double elapTime, const Dim &dim)
            : m_dim(dim)
        {
            m_quality = quali;
            m_elapsedTime = elapTime;
        }

        double quality() const
        {
            return m_quality;
        }
        
        double elapsedTime() const
        {
            return m_elapsedTime;
        }
        
        const Dim &dim() const
        {
            return m_dim;
        }
        
    private:
        Dim m_dim;
        double m_quality, m_elapsedTime;
    };
    
    class StereoPairReport
    {
    public:
        typedef std::list<SampleReport> SampleList;
        
        StereoPairReport(const std::string &name,
                         const std::string matcherName)
            : m_name(name), m_matcherName(matcherName)
        { }
        
        void addSampleReport(const SampleReport &report)
        {
            m_samples.push_back(report);
        }

        SampleList::iterator samplesBegin()
        {
            return m_samples.begin();
        }
    
        SampleList::iterator samplesEnd()
        {
            return m_samples.end();
        }
        
    private:
        SampleList m_samples;
        std::string m_name, m_matcherName;
        
    };
}

class BMDataset
{
public:
    virtual size_t stereoPairCount() = 0;
    
    virtual bmdata::StereoPair* stereoPair(size_t idx) = 0;
    
private:
};

TDV_NAMESPACE_END

#endif /* TDV_BMDATASET_HPP */
