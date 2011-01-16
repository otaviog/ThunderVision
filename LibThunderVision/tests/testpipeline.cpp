#include <gtest/gtest.h>

TEST(Pipeline, Rationalle)
{
    th::StereoVisionFlow flow;
    th::StereoVisionFlow flow2;

    th::CLContext *context = th::CLContext::Create();

    th::WUFactory *factory = new th::SystemWUFactory(context);
    th::WUFactory *factory2 = new th::TextWUFactory(
        context,
        "[filter = GPUGaussian]\n"
        "->[matchingcost = GPUSSD]\n"
        "->[agregate = GPUWTA]\n"
        "->[disparitymap = CPUDPA]\n");

    th::WUFactory *factory3 = new th::OpenCLWUFactory(context);
    th::FilterWU *filter = factory.createFilter();
    th::MatchingCostWU *matchCost = factory.createMatchingCost();
    th::AgregateWU *agregate = factory.createAgregate();

    flow.addWU(filter);
    flow.addWU(matchCost);
    flow.addWU(agregate);

    flow2.addWUs(factory2);

    th::LumiImage *imageLeft = th::imreadLumi(th::GetPath("tsukuba_left.png"));
    th::LumiImage *imageRight = th::imreadLumi(th::GetPath("tsukuba_right.png"));
    th::LumiImage disparity;
    flow.process(imageLeft, imageRight);
    flow.output();
}
