/*
 * Created by Marcos Luciano
 * https://www.github.com/marcoslucianops
 * 
 * 
 * 
 * Example:
 * 
 * [convolutional]
  batch_normalize=1
  filters=32
  size=6
  stride=2
  pad=2
  activation=silu
 */

#include "convolutional_layer.h"

#include <cassert>
#include <math.h>

nvinfer1::ITensor*
convolutionalLayer(int layerIdx, std::map<std::string, std::string>& block, std::vector<float>& weights,
    std::vector<nvinfer1::Weights>& trtWeights, int& weightPtr, std::string weightsType, int& inputChannels, float eps,
    nvinfer1::ITensor* input, nvinfer1::INetworkDefinition* network, std::string layerName)
{
  //Makes a tensor object to return
  nvinfer1::ITensor* output;

  //Make assertions about the input values of the function, to make sure we have valid input
  assert(block.at("type") == "convolutional" || block.at("type") == "c2f");
  assert(block.find("filters") != block.end());
  assert(block.find("pad") != block.end());
  assert(block.find("size") != block.end());
  assert(block.find("stride") != block.end());

  // assign each variable from the config file
  int filters = std::stoi(block.at("filters"));
  int padding = std::stoi(block.at("pad"));
  int kernelSize = std::stoi(block.at("size"));
  int stride = std::stoi(block.at("stride"));
  std::string activation = block.at("activation");
  int bias = filters;

  // For yoloV5, batchNormalize will be set to true
  bool batchNormalize = false;
  if (block.find("batch_normalize") != block.end()) {
    bias = 0;
    batchNormalize = (block.at("batch_normalize") == "1");
  }

  // For yoloV5, no bias or groups
  if (block.find("bias") != block.end()) {
    bias = std::stoi(block.at("bias"));
    if (bias == 1)
      bias = filters;
  }

  int groups = 1;
  if (block.find("groups") != block.end())
    groups = std::stoi(block.at("groups"));

  int pad;
  if (padding)
    pad = (kernelSize - 1) / 2;
  else
    pad = 0;

  int size = filters * inputChannels * kernelSize * kernelSize / groups;
  std::vector<float> bnBiases;
  std::vector<float> bnWeights;
  std::vector<float> bnRunningMean;
  std::vector<float> bnRunningVar;
  nvinfer1::Weights convWt {nvinfer1::DataType::kFLOAT, nullptr, size};
  nvinfer1::Weights convBias {nvinfer1::DataType::kFLOAT, nullptr, bias};

  // It's not of type "weights" for YoloV5, move to else
  if (weightsType == "weights") {
    if (batchNormalize == false) {
      float* val;
      if (bias != 0) {
        val = new float[filters];
        for (int i = 0; i < filters; ++i) {
            val[i] = weights[weightPtr];
            ++weightPtr;
        }
        convBias.values = val;
        trtWeights.push_back(convBias);
      }
      val = new float[size];
      for (int i = 0; i < size; ++i) {
          val[i] = weights[weightPtr];
          ++weightPtr;
      }
      convWt.values = val;
      trtWeights.push_back(convWt);
    }
    else {
      for (int i = 0; i < filters; ++i) {
        bnBiases.push_back(weights[weightPtr]);
        ++weightPtr;
      }
      for (int i = 0; i < filters; ++i) {
        bnWeights.push_back(weights[weightPtr]);
        ++weightPtr;
      }
      for (int i = 0; i < filters; ++i) {
        bnRunningMean.push_back(weights[weightPtr]);
        ++weightPtr;
      }
      for (int i = 0; i < filters; ++i) {
        bnRunningVar.push_back(sqrt(weights[weightPtr] + 1.0e-5));
        ++weightPtr;
      }
      float* val;
      if (bias != 0) {
        val = new float[filters];
        for (int i = 0; i < filters; ++i) {
          val[i] = weights[weightPtr];
          ++weightPtr;
        }
        convBias.values = val;
      }
      val = new float[size];
      for (int i = 0; i < size; ++i) {
        val[i] = weights[weightPtr];
        ++weightPtr;
      }
      convWt.values = val;
      trtWeights.push_back(convWt);
      if (bias != 0)
        trtWeights.push_back(convBias);
    }
  }
  // Run this else
  else {
    if (batchNormalize == false) {
      float* val = new float[size];
      for (int i = 0; i < size; ++i) {
        val[i] = weights[weightPtr];
        ++weightPtr;
      }
      convWt.values = val;
      trtWeights.push_back(convWt);
      if (bias != 0) {
        val = new float[filters];
        for (int i = 0; i < filters; ++i) {
          val[i] = weights[weightPtr];
          ++weightPtr;
        }
        convBias.values = val;
        trtWeights.push_back(convBias);
      }
    }
    // Run this else
    // weightPtr starts out at 0 for the first conv layer, but is incremented here.
      // since the weightPtr variable is a referenced int (int&) , we are changing the weightPtr variable in yolo.cpp as well
    else {
      float* val = new float[size];
      // Load the weight values for this conv layer
      for (int i = 0; i < size; ++i) {
        val[i] = weights[weightPtr];
        ++weightPtr;
      }
      convWt.values = val;
      // No bias for yolov5
      if (bias != 0) {
        val = new float[filters];
        for (int i = 0; i < filters; ++i) {
          val[i] = weights[weightPtr];
          ++weightPtr;
        }
        convBias.values = val;
      }
      // The weights file is followed by these headeders, reads in the following data from them
        // Still yet to completely sure what bn is supposed to be for, and why its present on every conolutional layer
      for (int i = 0; i < filters; ++i) {
        bnWeights.push_back(weights[weightPtr]);
        ++weightPtr;
      }
      for (int i = 0; i < filters; ++i) {
        bnBiases.push_back(weights[weightPtr]);
        ++weightPtr;
      }
      for (int i = 0; i < filters; ++i) {
        bnRunningMean.push_back(weights[weightPtr]);
        ++weightPtr;
      }
      for (int i = 0; i < filters; ++i) {
        bnRunningVar.push_back(sqrt(weights[weightPtr] + eps));
        ++weightPtr;
      }
      trtWeights.push_back(convWt);
      if (bias != 0)
        trtWeights.push_back(convBias);
    }
  }

  // Creates a convolutional layer object based on nvinfer defined classes
  nvinfer1::IConvolutionLayer* conv = network->addConvolutionNd(*input, filters, nvinfer1::Dims{2, {kernelSize, kernelSize}},
      convWt, convBias);
  assert(conv != nullptr);
  // Make up some layer name (I assume the layerName variable to be empty)
  std::string convLayerName = "conv_" + layerName + std::to_string(layerIdx);
  conv->setName(convLayerName.c_str());
  conv->setStrideNd(nvinfer1::Dims{2, {stride, stride}});
  conv->setPaddingNd(nvinfer1::Dims{2, {pad, pad}});

  if (block.find("groups") != block.end())
    conv->setNbGroups(groups);

  output = conv->getOutput(0);

  // A bunch of batchNormalize operations. Need to do more research into this
  if (batchNormalize == true) {
    size = filters;
    nvinfer1::Weights shift {nvinfer1::DataType::kFLOAT, nullptr, size};
    nvinfer1::Weights scale {nvinfer1::DataType::kFLOAT, nullptr, size};
    nvinfer1::Weights power {nvinfer1::DataType::kFLOAT, nullptr, size};
    float* shiftWt = new float[size];
    for (int i = 0; i < size; ++i)
      shiftWt[i] = bnBiases.at(i) - ((bnRunningMean.at(i) * bnWeights.at(i)) / bnRunningVar.at(i));
    shift.values = shiftWt;
    float* scaleWt = new float[size];
    for (int i = 0; i < size; ++i)
      scaleWt[i] = bnWeights.at(i) / bnRunningVar[i];
    scale.values = scaleWt;
    float* powerWt = new float[size];
    for (int i = 0; i < size; ++i)
      powerWt[i] = 1.0;
    power.values = powerWt;
    trtWeights.push_back(shift);
    trtWeights.push_back(scale);
    trtWeights.push_back(power);

    nvinfer1::IScaleLayer* batchnorm = network->addScale(*output, nvinfer1::ScaleMode::kCHANNEL, shift, scale, power);
    assert(batchnorm != nullptr);
    std::string batchnormLayerName = "batchnorm_" + layerName + std::to_string(layerIdx);
    batchnorm->setName(batchnormLayerName.c_str());
    output = batchnorm->getOutput(0);
  }

  output = activationLayer(layerIdx, activation, output, network, layerName);
  assert(output != nullptr);

  return output;
}
