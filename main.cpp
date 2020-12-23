#include <iostream>
#include <tensorflow/c/c_api.h>

int main()
{
    std::cout << "TensorFlow Version: " << TF_Version() << std::endl;
    return 0;
}