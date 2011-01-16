#ifndef TH_LUMIMAGE_HPP
#define TH_LUMIMAGE_HPP

#include <cv/mat.hpp>

TH_NAMESPACE_BEGIN

class LumImage
{
public:
    class GData
    {
    public:
        scoped_lock();

        ~scoped_lock()
        {
            image.unlockGPU();
        }

    private:
        GData data;
        LumImage &image;
    };

    LumImage();

    virtual ~LumImage();

    GData lockDev(cl);

    void unlockDev();

    cv::Mat lock();

    void unlock();

private:
    cv::Mat mat;
};

TH_NAMESPACE_END

#endif /* TH_LUMIMAGE_HPP */
