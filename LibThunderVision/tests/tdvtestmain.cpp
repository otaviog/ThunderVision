#include <gtest/gtest.h>
#include <tdvbasic/log.hpp>

int main(int argc, char **argv) {
    tdv::TdvGlobalLogDefaultOutputs();
    ::testing::InitGoogleTest(&argc, argv);
   
    return RUN_ALL_TESTS();
}
