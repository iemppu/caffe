// Copyright 2014 kloudkl@github.com

#ifndef CAFFE_ADAPTIVE_LEARNING_RATE_HPP_
#define CAFFE_ADAPTIVE_LEARNING_RATE_HPP_

#include <map>
#include <string>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

using std::map;
using std::vector;
using std::string;

namespace caffe {

template<typename Dtype>
class AdaptiveLearningRate {
 public:
  AdaptiveLearningRate(const LayerParameter& param);
  virtual ~AdaptiveLearningRate() {
  }

  // Initialize a network with the network parameter.
  void Init(const NetParameter& param);

 protected:
  LayerParameter& param_;
DISABLE_COPY_AND_ASSIGN(AdaptiveLearningRate);
};

// Reference:
// John Duchi, Elad Hazan, and Yoram Singer. Adaptive Subgradient Methods for
//   Online Learning and Stochastic Optimization, Journal of Machine Learning
//   Research (JMLR 2011).
template<typename Dtype>
class AdaGradAdaptiveLearningRate : public AdaptiveLearningRate<Dtype> {

};

// Reference:
// Andrew Senior, Georg Heigold, Marc'aurelio Ranzato, Ke Yang. An Empirical
//   study of learning rates in deep neural networks for speech recognition
//   Proceedings of the IEEE International Conference on Acoustics, Speech,
//   and Signal Processing (ICASSP), IEEE, Vancouver, CA (2013).
template<typename Dtype>
class AdaDecAdaptiveLearningRate : public AdaptiveLearningRate<Dtype> {

};

#define ADA_LR_TYPE(type) ADA_LR_TYPE_PASTE(type)
#define ADA_LR_TYPE_PASTE(type) LayerParameter_AdaptiveLearningRateType_##type

template<typename Dtype>
AdaptiveLearningRate<Dtype>* GetAdaptiveLearningRate(
    const LayerParameter param);

}  // namespace caffe

#endif  // CAFFE_ADAPTIVE_LEARNING_RATE_HPP_
