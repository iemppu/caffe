// Copyright 2014 kloudkl@github.com

#include <cstdio>

#include <algorithm>
#include <string>
#include <vector>

#include "caffe/adaptive_learning_rate.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/solver.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"

using std::max;
using std::min;

namespace caffe {

template<typename Dtype>
AdaptiveLearningRate<Dtype>::AdaptiveLearningRate(const LayerParameter& param)
    : param_(param) {
}

template<typename Dtype>
AdaptiveLearningRate<Dtype>* GetAdaptiveLearningRate(
    const LayerParameter param) {
  const ::caffe::LayerParameter_AdaptiveLearningRateType& type = param
      .adaptive_learning_rate();
  if (type == ADA_LR_TYPE(ADA_GRAD)) {
    return new AdaGradAdaptiveLearningRate();
  } else if (type == ADA_LR_TYPE(ADA_GRAD)) {
    return new AdaDecAdaptiveLearningRate();
  } else {
    LOG(FATAL)<< "Unknown adaptive learning rate: " << type;
  }
  return (AdaptiveLearningRate<Dtype>*) (NULL);
}

INSTANTIATE_CLASS(AdaptiveLearningRate);
INSTANTIATE_CLASS(AdaGradAdaptiveLearningRate);
INSTANTIATE_CLASS(AdaDecAdaptiveLearningRate);

}  // namespace caffe
