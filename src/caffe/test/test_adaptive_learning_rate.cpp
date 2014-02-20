// Copyright 2014 kloudkl@github.com

#include <cuda_runtime.h>

#include "gtest/gtest.h"
#include "caffe/adaptive_learning_rate.hpp"
#include "caffe/common.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

extern cudaDeviceProp CAFFE_TEST_CUDA_PROP;

template<typename Dtype>
class AdaptiveLearningRateTest : public ::testing::Test {
 protected:
  AdaptiveLearningRateTest()
      : blob_bottom_(new Blob<Dtype>(2, 3, 6, 5)),
        blob_top_(new Blob<Dtype>()) {
    // fill the values
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }
  ;
  virtual ~AdaptiveLearningRateTest() {
    delete blob_bottom_;
    delete blob_top_;
  }
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

typedef ::testing::Types<float, double> Dtypes;
TYPED_TEST_CASE(AdaptiveLearningRateTest, Dtypes);

TYPED_TEST(AdaptiveLearningRateTest, TestOne) {

}

TYPED_TEST(AdaptiveLearningRateTest, TestTwo) {

}

TYPED_TEST(AdaptiveLearningRateTest, TestCPUThree) {

}

TYPED_TEST(AdaptiveLearningRateTest, TestGPUThree) {

}

}
