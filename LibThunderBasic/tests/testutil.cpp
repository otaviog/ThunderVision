#include <gtest/gtest.h>
#include <tdvbasic/util.hpp>

TEST(UtilTest, ReplaceFirst)
{
    const char *orign = "This is a test please test it";
    const char *search = "test";
    const char *replace1 = "est";
    const char *replace2 = "bigtest";
    const char *search2 = "not found";

    char *nstr = tdv::util::strReplaceFirst(orign, strlen(orign),
                                       search, strlen(search),
                                       replace1, strlen(replace1));

    ASSERT_TRUE(nstr != NULL);
    EXPECT_STREQ("This is a est please test it", nstr);
    delete [] nstr;

    nstr = tdv::util::strReplaceFirst(orign, strlen(orign),
                                 search, strlen(search),
                                 replace2, strlen(replace2));

    ASSERT_TRUE(nstr != NULL);
    EXPECT_STREQ("This is a bigtest please test it", nstr);
    delete [] nstr;

    nstr = tdv::util::strReplaceFirst(orign, strlen(orign),
                                 search2, strlen(search2),
                                 replace2, strlen(replace2));
    ASSERT_TRUE(nstr == NULL);
}


TEST(UtilTest, Replace)
{
    const char *orign = "This is a test please test it. test it goog to be perfect. \ntest\n is fun. don't stop \ttesting";
    const char *search = "test";
    const char *replace1 = "est";
    const char *replace2 = "bigtest";
    const char *search2 = "not found";

    char *nstr = tdv::util::strReplace(orign, search,
                                  replace1, strlen(replace1));

    ASSERT_TRUE(nstr != NULL);
    EXPECT_STREQ("This is a est please est it. est it goog to be perfect. \nest\n is fun. don't stop \testing", nstr);
    delete [] nstr;

    nstr = tdv::util::strReplace(orign, search,
                            replace2, strlen(replace2));

    ASSERT_TRUE(nstr != NULL);
    EXPECT_STREQ("This is a bigtest please bigtest it. bigtest it goog to be perfect. \nbigtest\n is fun. don't stop \tbigtesting", nstr);
    delete [] nstr;

    nstr = tdv::util::strReplace(orign, search2,
                            replace2, strlen(replace2));
    ASSERT_TRUE(nstr == NULL);
}
