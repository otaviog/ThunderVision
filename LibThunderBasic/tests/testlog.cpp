#include <gtest/gtest.h>
#include <sstream>
#include <tdvbasic/log.hpp>

class LogTest: public ::testing::Test
{
public:
    LogTest()
    {
        using namespace tdv;

        boost::shared_ptr<StreamLogOutput> errO1(new StreamLogOutput(debStream));
        boost::shared_ptr<StreamLogOutput> errO2(new StreamLogOutput(warnStream));
        boost::shared_ptr<StreamLogOutput> errO3(new StreamLogOutput(fatalStream));

        EXPECT_TRUE(log.registerOutput(
                        "deb", errO1));

        EXPECT_TRUE(log.registerOutput(
                        "warn", errO2));

        EXPECT_TRUE(log.registerOutput(
                        "fatal", errO3));

    }

    virtual void SetUp()
    { }

    virtual void TearDown()
    { }

protected:
    tdv::Log log;
    std::stringstream debStream, warnStream, fatalStream;
};

TEST_F(LogTest, LogShouldLogMessages)
{
    log("deb") = boost::format("Deb value %1% desc %2%") % 4 % "message";
    EXPECT_EQ("Deb value 4 desc message", debStream.str());

    log("warn").printf("Test value %.1f", 34.5);
    EXPECT_EQ("Test value 34.5", warnStream.str());

    log("fatal").printf("Hello %s %d\n", "world", 5);
    log("fatal") = boost::format("How are you");
    EXPECT_EQ("Hello world 5\nHow are you", fatalStream.str());
}
