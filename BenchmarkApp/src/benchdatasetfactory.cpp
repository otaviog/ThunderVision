#include <boost/filesystem/path.hpp>
#include <boost/format.hpp>
#include <tdvision/floatimage.hpp>
#include <cv.h>
#include <highgui.h>
#include <tdvbasic/exception.hpp>
#include "ibenchmarkdataset.hpp"
#include "benchdatasetfactory.hpp"

TDV_NAMESPACE_BEGIN

class DefaultBenchDataset: public IBenchmarkDataset
{
public:
    virtual ~DefaultBenchDataset()
    {
        for (size_t i=0; i<m_stereos.size(); i++)
        {
            delete m_stereos[i];
        }
    }
    
    size_t stereoPairCount()
    {
        return m_stereos.size();
    }
    
    StereoPair* stereoPair(size_t idx)
    {
        return m_stereos[idx];
    }
        
    void addStereoPair(StereoPair *pair)
    {
        m_stereos.push_back(pair);
    }
    
private:
    std::vector<StereoPair*> m_stereos;
};

static FloatImage loadImage(const std::string &filename)
{
    IplImage *img = cvLoadImage(filename.c_str());
    
    if ( img != NULL )
    {
        FloatImage image(img);
        return image;
    }
    else
    {
        throw Exception(boost::format("can't open image: %1%")
                        % filename);
    }
}

IBenchmarkDataset* BenchDatasetFactory::CreateDefault(const std::string &basePath)
{
    namespace fs = boost::filesystem;
    
    fs::path base(basePath);
    DefaultBenchDataset *dataset = new DefaultBenchDataset;
    
    StereoPair *teddyPair = new StereoPair("Teddy");
    teddyPair->addSample(
        StereoPair::Sample(
            loadImage((base / "Teddy/f_left.png").string()),
            loadImage((base / "Teddy/f_right.png").string()),
            loadImage((base / "Teddy/f_true.png").string())));
    teddyPair->addSample(
        StereoPair::Sample(
            loadImage((base / "Teddy/h_left.png").string()),
            loadImage((base / "Teddy/h_right.png").string()),
            loadImage((base / "Teddy/h_true.png").string())));
    teddyPair->addSample(
        StereoPair::Sample(
            loadImage((base / "Teddy/q_left.png").string()),
            loadImage((base / "Teddy/q_right.png").string()),
            loadImage((base / "Teddy/q_true.png").string())));
    dataset->addStereoPair(teddyPair);
        
    StereoPair *conesPair = new StereoPair("Cones");
    conesPair->addSample(
        StereoPair::Sample(
            loadImage((base / "Cones/f_left.png").string()),
            loadImage((base / "Cones/f_right.png").string()),
            loadImage((base / "Cones/f_true.png").string())));
    conesPair->addSample(
        StereoPair::Sample(
            loadImage((base / "Cones/h_left.png").string()),
            loadImage((base / "Cones/h_right.png").string()),
            loadImage((base / "Cones/h_true.png").string())));
    conesPair->addSample(
        StereoPair::Sample(
            loadImage((base / "Cones/q_left.png").string()),
            loadImage((base / "Cones/q_right.png").string()),
            loadImage((base / "Cones/q_true.png").string())));    
    
    dataset->addStereoPair(conesPair);
    
    return dataset;
}

TDV_NAMESPACE_END
