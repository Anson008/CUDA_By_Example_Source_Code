#include "../common/book.h"
#include <iostream>

__global__ void kernel() {}

int main() {
    kernel<<<1, 1>>>();
    printf("Hello, World!\n");
    return 0;
}