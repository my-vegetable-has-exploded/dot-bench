#include "distance.h"
#include <stdio.h>

int main() {
#if TARGET_SVE
	printf("SVE\n");
#else
	printf("NEON\n");
#endif

}
