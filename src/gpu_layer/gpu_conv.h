#include"../layer/conv.h"

class CNN_cuda_v1: public Conv{
public:
    void forward(const Matrix& input);
};