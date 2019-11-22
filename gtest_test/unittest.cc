#include <gtest/gtest.h>

#define HELLOWORLD "Hello World"

TEST(GoogleTest, HelloWorld) {
	ASSERT_EQ("Hello World", HELLOWORLD);
}

int main (int argc, char ** argv) {
	testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
