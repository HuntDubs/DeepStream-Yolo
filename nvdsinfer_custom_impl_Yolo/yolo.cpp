/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 *
 * Edited by Marcos Luciano
 * https://www.github.com/marcoslucianops
 * 
 * output with a confidence score and a class I.D.
 * 
 * This parser is made by some rando, so locate his parts and not the nvidia parts
 */

#include "yolo.h"
#include "yoloPlugins.h"

#ifdef OPENCV
#include "calibrator.h"
#endif

// Initialize all private variables
Yolo::Yolo(const NetworkInfo& networkInfo) : m_InputBlobName(networkInfo.inputBlobName),
    m_NetworkType(networkInfo.networkType), m_ConfigFilePath(networkInfo.configFilePath),
    m_WtsFilePath(networkInfo.wtsFilePath), m_Int8CalibPath(networkInfo.int8CalibPath), m_DeviceType(networkInfo.deviceType),
    m_NumDetectedClasses(networkInfo.numDetectedClasses), m_ClusterMode(networkInfo.clusterMode),
    m_NetworkMode(networkInfo.networkMode), m_ScoreThreshold(networkInfo.scoreThreshold), m_InputH(0), m_InputW(0),
    m_InputC(0), m_InputSize(0), m_NumClasses(0), m_LetterBox(0), m_NewCoords(0), m_YoloCount(0)
{
}

Yolo::~Yolo()
{
  destroyNetworkUtils();
}

nvinfer1::ICudaEngine* 
Yolo::createEngine(nvinfer1::IBuilder* builder, nvinfer1::IBuilderConfig* config)
{
  assert (builder);

  m_ConfigBlocks = parseConfigFile(m_ConfigFilePath);
  parseConfigBlocks();

  //Creates a network definiton object. networkv2 supports dynamic shapes and explicit bash permissions
  nvinfer1::INetworkDefinition *network = builder->createNetworkV2(0);
  // Check if parseModel was a sucess before continuing
  if (parseModel(*network) != NVDSINFER_SUCCESS) {
    delete network;
    return nullptr;
  }

  std::cout << "Building the TensorRT Engine\n" << std::endl;

  if (m_NumClasses != m_NumDetectedClasses) {
    std::cout << "NOTE: Number of classes mismatch, make sure to set num-detected-classes=" << m_NumClasses
        << " in config_infer file\n" << std::endl;
  }
  if (m_LetterBox == 1) {
      std::cout << "NOTE: letter_box is set in cfg file, make sure to set maintain-aspect-ratio=1 in config_infer file"
          << " to get better accuracy\n" << std::endl;
  }
  if (m_ClusterMode != 2) {
      std::cout << "NOTE: Wrong cluster-mode is set, make sure to set cluster-mode=2 in config_infer file\n" << std::endl;
  }

  if (m_NetworkMode == "INT8" && !fileExists(m_Int8CalibPath)) {
    assert(builder->platformHasFastInt8());
#ifdef OPENCV
    std::string calib_image_list;
    int calib_batch_size;
    if (getenv("INT8_CALIB_IMG_PATH"))
      calib_image_list = getenv("INT8_CALIB_IMG_PATH");
    else {
      std::cerr << "INT8_CALIB_IMG_PATH not set" << std::endl;
      assert(0);
    }
    if (getenv("INT8_CALIB_BATCH_SIZE"))
      calib_batch_size = std::stoi(getenv("INT8_CALIB_BATCH_SIZE"));
    else {
      std::cerr << "INT8_CALIB_BATCH_SIZE not set" << std::endl;
      assert(0);
    }
    nvinfer1::IInt8EntropyCalibrator2 *calibrator = new Int8EntropyCalibrator2(calib_batch_size, m_InputC, m_InputH,
        m_InputW, m_LetterBox, calib_image_list, m_Int8CalibPath);
    config->setFlag(nvinfer1::BuilderFlag::kINT8);
    config->setInt8Calibrator(calibrator);
#else
    std::cerr << "OpenCV is required to run INT8 calibrator\n" << std::endl;
    assert(0);
#endif
  }

  nvinfer1::ICudaEngine *engine = builder->buildEngineWithConfig(*network, *config);
  if (engine)
    std::cout << "Building complete\n" << std::endl;
  else
    std::cerr << "Building engine failed\n" << std::endl;

  delete network;
  return engine;
}

// Based on output to console, this is where we start execution
NvDsInferStatus
Yolo::parseModel(nvinfer1::INetworkDefinition& network) {
  destroyNetworkUtils();

  //read the weights stored in the weights file
    //find where loadWeights() is being defined
  std::vector<float> weights = loadWeights(m_WtsFilePath, m_NetworkType);
  std::cout << "Building YOLO network\n" << std::endl;
  //Build the yolo nerwork by building each yolo layer
  NvDsInferStatus status = buildYoloNetwork(weights, network);

  if (status == NVDSINFER_SUCCESS)
    std::cout << "Building YOLO network complete" << std::endl;
  else
    std::cerr << "Building YOLO network failed" << std::endl;

  return status;
}

/*
typedef enum {
       NVDSINFER_SUCCESS = 0,
       NVDSINFER_CONFIG_FAILED,
       NVDSINFER_CUSTOM_LIB_FAILED,
      NVDSINFER_INVALID_PARAMS,
      NVDSINFER_OUTPUT_PARSING_FAILED,
       NVDSINFER_CUDA_ERROR,
       NVDSINFER_TENSORRT_ERROR,
       NVDSINFER_RESOURCE_ERROR,
       NVDSINFER_TRTIS_ERROR,
       NVDSINFER_UNKNOWN_ERROR
   } NvDsInferStatus;
*/
//We want to return NVDSINFER_SUCCESS as a success, with no other errors
NvDsInferStatus
Yolo::buildYoloNetwork(std::vector<float>& weights, nvinfer1::INetworkDefinition& network)
{
  // std::string::npos simply means 'until the end of the string'

  int weightPtr = 0;

  std::string weightsType = "wts";
  if (m_WtsFilePath.find(".weights") != std::string::npos)
      weightsType = "weights";

  float eps = 1.0e-5;
  if (m_NetworkType.find("yolov5") != std::string::npos || m_NetworkType.find("yolov6") != std::string::npos ||
      m_NetworkType.find("yolov7") != std::string::npos || m_NetworkType.find("yolov8") != std::string::npos ||
      m_NetworkType.find("yolox") != std::string::npos)
    eps = 1.0e-3;
  else if (m_NetworkType.find("yolor") != std::string::npos)
    eps = 1.0e-4;


  // Add the information we extracted from the [net] layer as input for the network
  nvinfer1::ITensor* data = network.addInput(m_InputBlobName.c_str(), nvinfer1::DataType::kFLOAT,
      nvinfer1::Dims{3, {static_cast<int>(m_InputC), static_cast<int>(m_InputH), static_cast<int>(m_InputW)}});
  assert(data != nullptr && data->getDimensions().nbDims > 0);

  nvinfer1::ITensor* previous = data;
  std::vector<nvinfer1::ITensor*> tensorOutputs;

  nvinfer1::ITensor* yoloTensorInputs[m_YoloCount];
  uint yoloCountInputs = 0;

  int modelType = -1;

  // Loop through all of the blocks
    // starting with the [net] block, which will print the column headers
  for (uint i = 0; i < m_ConfigBlocks.size(); ++i) {
    std::string layerIndex = "(" + std::to_string(tensorOutputs.size()) + ")";

    if (m_ConfigBlocks.at(i).at("type") == "net") //Print the headers for each column in the Yolo network
        printLayerInfo("", "Layer", "Input Shape", "Output Shape", "WeightPtr");
    else if (m_ConfigBlocks.at(i).at("type") == "convolutional") {
      int channels = getNumChannels(previous);
      // The input shape
      std::string inputVol = dimsToString(previous->getDimensions());
      // Creates a convolutionalLayer object based on convolutinal_layer.h and convolutional_layer.cpp
        //Will be similar for all other types of layers
      previous = convolutionalLayer(i, m_ConfigBlocks.at(i), weights, m_TrtWeights, weightPtr, weightsType, channels, eps,
          previous, &network);
      assert(previous != nullptr);
      // The output shape
      std::string outputVol = dimsToString(previous->getDimensions());
      // Add the convolutionalLayer as a tensor to the tensorOutputs vector
      tensorOutputs.push_back(previous);
      // Layer name , ex. Conv_silu , ex. Conv_logistic
      std::string layerName = "conv_" + m_ConfigBlocks.at(i).at("activation");
      // Print information to the command line
      printLayerInfo(layerIndex, layerName, inputVol, outputVol, std::to_string(weightPtr));
    }
    else if (m_ConfigBlocks.at(i).at("type") == "deconvolutional") { 
      int channels = getNumChannels(previous);
      std::string inputVol = dimsToString(previous->getDimensions());
      previous = deconvolutionalLayer(i, m_ConfigBlocks.at(i), weights, m_TrtWeights, weightPtr, weightsType, channels,
          previous, &network);
      assert(previous != nullptr);
      std::string outputVol = dimsToString(previous->getDimensions());
      tensorOutputs.push_back(previous);
      std::string layerName = "deconv";
      printLayerInfo(layerIndex, layerName, inputVol, outputVol, std::to_string(weightPtr));
    }
    else if (m_ConfigBlocks.at(i).at("type") == "c2f") {
      std::string inputVol = dimsToString(previous->getDimensions());
      previous = c2fLayer(i, m_ConfigBlocks.at(i), weights, m_TrtWeights, weightPtr, weightsType, eps, previous, &network);
      assert(previous != nullptr);
      std::string outputVol = dimsToString(previous->getDimensions());
      tensorOutputs.push_back(previous);
      std::string layerName = "c2f_" + m_ConfigBlocks.at(i).at("activation");
      printLayerInfo(layerIndex, layerName, inputVol, outputVol, std::to_string(weightPtr));
    }
    else if (m_ConfigBlocks.at(i).at("type") == "batchnorm") {
      std::string inputVol = dimsToString(previous->getDimensions());
      previous = batchnormLayer(i, m_ConfigBlocks.at(i), weights, m_TrtWeights, weightPtr, weightsType, eps, previous,
          &network);
      assert(previous != nullptr);
      std::string outputVol = dimsToString(previous->getDimensions());
      tensorOutputs.push_back(previous);
      std::string layerName = "batchnorm_" + m_ConfigBlocks.at(i).at("activation");
      printLayerInfo(layerIndex, layerName, inputVol, outputVol, std::to_string(weightPtr));
    }
    else if (m_ConfigBlocks.at(i).at("type") == "implicit_add" || m_ConfigBlocks.at(i).at("type") == "implicit_mul") {
      previous = implicitLayer(i, m_ConfigBlocks.at(i), weights, m_TrtWeights, weightPtr, &network);
      assert(previous != nullptr);
      std::string outputVol = dimsToString(previous->getDimensions());
      tensorOutputs.push_back(previous);
      std::string layerName =  m_ConfigBlocks.at(i).at("type");
      printLayerInfo(layerIndex, layerName, "-", outputVol, std::to_string(weightPtr));
    }
    else if (m_ConfigBlocks.at(i).at("type") == "shift_channels" || m_ConfigBlocks.at(i).at("type") == "control_channels") {
      assert(m_ConfigBlocks.at(i).find("from") != m_ConfigBlocks.at(i).end());
      int from = stoi(m_ConfigBlocks.at(i).at("from"));
      if (from > 0)
        from = from - i + 1;
      assert((i - 2 >= 0) && (i - 2 < tensorOutputs.size()));
      assert((i + from - 1 >= 0) && (i + from - 1 < tensorOutputs.size()));
      assert(i + from - 1 < i - 2);

      std::string inputVol = dimsToString(previous->getDimensions());
      previous = channelsLayer(i, m_ConfigBlocks.at(i), previous, tensorOutputs[i + from - 1], &network);
      assert(previous != nullptr);
      std::string outputVol = dimsToString(previous->getDimensions());
      tensorOutputs.push_back(previous);
      std::string layerName = m_ConfigBlocks.at(i).at("type") + ": " + std::to_string(i + from - 1);
      printLayerInfo(layerIndex, layerName, inputVol, outputVol, "-");
    }
    else if (m_ConfigBlocks.at(i).at("type") == "shortcut") {

      assert(m_ConfigBlocks.at(i).find("from") != m_ConfigBlocks.at(i).end());
      int from = stoi(m_ConfigBlocks.at(i).at("from"));
      if (from > 0)
        from = from - i + 1;
      assert((i - 2 >= 0) && (i - 2 < tensorOutputs.size()));
      assert((i + from - 1 >= 0) && (i + from - 1 < tensorOutputs.size()));
      assert(i + from - 1 < i - 2);

      std::string mode = "add";
      if (m_ConfigBlocks.at(i).find("mode") != m_ConfigBlocks.at(i).end())
        mode = m_ConfigBlocks.at(i).at("mode");

      std::string activation = "linear";
      if (m_ConfigBlocks.at(i).find("activation") != m_ConfigBlocks.at(i).end())
        activation = m_ConfigBlocks.at(i).at("activation");

      std::string inputVol = dimsToString(previous->getDimensions());
      std::string shortcutVol = dimsToString(tensorOutputs[i + from - 1]->getDimensions());
      previous = shortcutLayer(i, mode, activation, inputVol, shortcutVol, m_ConfigBlocks.at(i), previous,
          tensorOutputs[i + from - 1], &network);
      assert(previous != nullptr);
      std::string outputVol = dimsToString(previous->getDimensions());
      tensorOutputs.push_back(previous);
      std::string layerName = "shortcut_" + mode + "_" + activation + ": " + std::to_string(i + from - 1);
      printLayerInfo(layerIndex, layerName, inputVol, outputVol, "-");

      if (mode == "add" && inputVol != shortcutVol)
        std::cout << inputVol << " +" << shortcutVol << std::endl;
    }
    else if (m_ConfigBlocks.at(i).at("type") == "route") {
      std::string layers;
      previous = routeLayer(i, layers, m_ConfigBlocks.at(i), tensorOutputs, &network);
      assert(previous != nullptr);
      std::string outputVol = dimsToString(previous->getDimensions());
      tensorOutputs.push_back(previous);
      std::string layerName = "route: " + layers;
      printLayerInfo(layerIndex, layerName, "-", outputVol, "-");
    }
    else if (m_ConfigBlocks.at(i).at("type") == "upsample") {
      std::string inputVol = dimsToString(previous->getDimensions());
      previous = upsampleLayer(i, m_ConfigBlocks[i], previous, &network);
      assert(previous != nullptr);
      std::string outputVol = dimsToString(previous->getDimensions());
      tensorOutputs.push_back(previous);
      std::string layerName = "upsample";
      printLayerInfo(layerIndex, layerName, inputVol, outputVol, "-");
    }
    else if (m_ConfigBlocks.at(i).at("type") == "maxpool" || m_ConfigBlocks.at(i).at("type") == "avgpool") {
      std::string inputVol = dimsToString(previous->getDimensions());
      previous = poolingLayer(i, m_ConfigBlocks.at(i), previous, &network);
      assert(previous != nullptr);
      std::string outputVol = dimsToString(previous->getDimensions());
      tensorOutputs.push_back(previous);
      std::string layerName = m_ConfigBlocks.at(i).at("type");
      printLayerInfo(layerIndex, layerName, inputVol, outputVol, "-");
    }
    else if (m_ConfigBlocks.at(i).at("type") == "reorg") {
      std::string inputVol = dimsToString(previous->getDimensions());
      if (m_NetworkType.find("yolov2") != std::string::npos) {
        nvinfer1::IPluginV2* reorgPlugin = createReorgPlugin(2);
        assert(reorgPlugin != nullptr);
        nvinfer1::IPluginV2Layer* reorg = network.addPluginV2(&previous, 1, *reorgPlugin);
        assert(reorg != nullptr);
        std::string reorglayerName = "reorg_" + std::to_string(i);
        reorg->setName(reorglayerName.c_str());
        previous = reorg->getOutput(0);
      }
      else
        previous = reorgLayer(i, m_ConfigBlocks.at(i), previous, &network);
      assert(previous != nullptr);
      std::string outputVol = dimsToString(previous->getDimensions());
      tensorOutputs.push_back(previous);
      std::string layerName = "reorg";
      printLayerInfo(layerIndex, layerName, inputVol, outputVol, "-");
    }
    else if (m_ConfigBlocks.at(i).at("type") == "reduce") {
      std::string inputVol = dimsToString(previous->getDimensions());
      previous = reduceLayer(i, m_ConfigBlocks.at(i), previous, &network);
      assert(previous != nullptr);
      std::string outputVol = dimsToString(previous->getDimensions());
      tensorOutputs.push_back(previous);
      std::string layerName = "reduce";
      printLayerInfo(layerIndex, layerName, inputVol, outputVol, "-");
    }
    else if (m_ConfigBlocks.at(i).at("type") == "shuffle") {
      std::string inputVol = dimsToString(previous->getDimensions());
      previous = shuffleLayer(i, m_ConfigBlocks.at(i), previous, tensorOutputs, &network);
      assert(previous != nullptr);
      std::string outputVol = dimsToString(previous->getDimensions());
      tensorOutputs.push_back(previous);
      std::string layerName = "shuffle";
      printLayerInfo(layerIndex, layerName, inputVol, outputVol, "-");
    }
    else if (m_ConfigBlocks.at(i).at("type") == "softmax") {
      std::string inputVol = dimsToString(previous->getDimensions());
      previous = softmaxLayer(i, m_ConfigBlocks.at(i), previous, &network);
      assert(previous != nullptr);
      std::string outputVol = dimsToString(previous->getDimensions());
      tensorOutputs.push_back(previous);
      std::string layerName = "softmax";
      printLayerInfo(layerIndex, layerName, inputVol, outputVol, "-");
    }
    else if (m_ConfigBlocks.at(i).at("type") == "yolo" || m_ConfigBlocks.at(i).at("type") == "region") {
      // For yolov5, our modelType = 1
      if (m_ConfigBlocks.at(i).at("type") == "yolo")
        if (m_NetworkType.find("yolor") != std::string::npos)
          modelType = 2;
        else
          modelType = 1;
      else
        modelType = 0;

      // our modelType is 1, so blobName will be yolo_someindex
      std::string blobName = modelType != 0 ? "yolo_" + std::to_string(i) : "region_" + std::to_string(i);
      // get Deminsions of previous layer (a conv layer)
      nvinfer1::Dims prevTensorDims = previous->getDimensions();
      // Get TensorInfo struct of current yolo layer (0,1,or 2)
      TensorInfo& curYoloTensor = m_YoloTensors.at(yoloCountInputs);
      // assign it's blobName
        //Set the gridSize as the previous layers dimensions
      curYoloTensor.blobName = blobName;
      curYoloTensor.gridSizeX = prevTensorDims.d[2];
      curYoloTensor.gridSizeY = prevTensorDims.d[1];

      // Do all this for printing to terminal
        // Also increase the yoloCountInputs so we work with the next TensorInfo struct next time around
      std::string inputVol = dimsToString(previous->getDimensions());
      tensorOutputs.push_back(previous);
      yoloTensorInputs[yoloCountInputs] = previous;
      ++yoloCountInputs;
      std::string layerName = modelType != 0 ? "yolo" : "region";
      printLayerInfo(layerIndex, layerName, inputVol, "-", "-");
    }
    else if (m_ConfigBlocks.at(i).at("type") == "cls") {
      modelType = 3;

      std::string blobName = "cls_" + std::to_string(i);
      nvinfer1::Dims prevTensorDims = previous->getDimensions();
      TensorInfo& curYoloTensor = m_YoloTensors.at(yoloCountInputs);
      curYoloTensor.blobName = blobName;
      curYoloTensor.numBBoxes = prevTensorDims.d[1];
      m_NumClasses = prevTensorDims.d[0];

      std::string inputVol = dimsToString(previous->getDimensions());
      previous = clsLayer(i, m_ConfigBlocks.at(i), previous, &network);
      assert(previous != nullptr);
      std::string outputVol = dimsToString(previous->getDimensions());
      tensorOutputs.push_back(previous);
      yoloTensorInputs[yoloCountInputs] = previous;
      ++yoloCountInputs;
      std::string layerName = "cls";
      printLayerInfo(layerIndex, layerName, inputVol, outputVol, "-");
    }
    else if (m_ConfigBlocks.at(i).at("type") == "reg") {
      modelType = 3;

      std::string blobName = "reg_" + std::to_string(i);
      nvinfer1::Dims prevTensorDims = previous->getDimensions();
      TensorInfo& curYoloTensor = m_YoloTensors.at(yoloCountInputs);
      curYoloTensor.blobName = blobName;
      curYoloTensor.numBBoxes = prevTensorDims.d[1];

      std::string inputVol = dimsToString(previous->getDimensions());
      previous = regLayer(i, m_ConfigBlocks.at(i), weights, m_TrtWeights, weightPtr, previous, &network);
      assert(previous != nullptr);
      std::string outputVol = dimsToString(previous->getDimensions());
      tensorOutputs.push_back(previous);
      yoloTensorInputs[yoloCountInputs] = previous;
      ++yoloCountInputs;
      std::string layerName = "reg";
      printLayerInfo(layerIndex, layerName, inputVol, outputVol, std::to_string(weightPtr));
    }
    else if (m_ConfigBlocks.at(i).at("type") == "detect_v8") {
      modelType = 4;

      std::string blobName = "detect_v8_" + std::to_string(i);
      nvinfer1::Dims prevTensorDims = previous->getDimensions();
      TensorInfo& curYoloTensor = m_YoloTensors.at(yoloCountInputs);
      curYoloTensor.blobName = blobName;
      curYoloTensor.numBBoxes = prevTensorDims.d[1];

      std::string inputVol = dimsToString(previous->getDimensions());
      previous = detectV8Layer(i, m_ConfigBlocks.at(i), weights, m_TrtWeights, weightPtr, previous, &network);
      assert(previous != nullptr);
      std::string outputVol = dimsToString(previous->getDimensions());
      tensorOutputs.push_back(previous);
      yoloTensorInputs[yoloCountInputs] = previous;
      ++yoloCountInputs;
      std::string layerName = "detect_v8";
      printLayerInfo(layerIndex, layerName, inputVol, outputVol, std::to_string(weightPtr));
    }
    else if (m_ConfigBlocks.at(i).at("type") == "detect_x") {
      modelType = 5;

      std::string blobName = "detect_x_" + std::to_string(i);
      nvinfer1::Dims prevTensorDims = previous->getDimensions();
      TensorInfo& curYoloTensor = m_YoloTensors.at(yoloCountInputs);
      curYoloTensor.blobName = blobName;
      curYoloTensor.numBBoxes = prevTensorDims.d[0];
      m_NumClasses = prevTensorDims.d[1] - 5;

      std::string outputVol = dimsToString(previous->getDimensions());
      tensorOutputs.push_back(previous);
      yoloTensorInputs[yoloCountInputs] = previous;
      ++yoloCountInputs;
      std::string layerName = "detect_x";
      printLayerInfo(layerIndex, layerName, "-", outputVol, std::to_string(weightPtr));
    }
    else {
      std::cerr << "\nUnsupported layer type --> \"" << m_ConfigBlocks.at(i).at("type") << "\"" << std::endl;
      assert(0);
    }
  }

  // If the weightPtr didn't make it to the end of the file, we might have a problem!
  if ((int) weights.size() != weightPtr) {
    std::cerr << "\nNumber of unused weights left: " << weights.size() - weightPtr << std::endl;
    assert(0);
  }

  // m_YoloCount was initialized to 0 in the class constructor, incremented for each yolo tensor in parseConfigBlocks()
  // yoloCountInputs was incremented for each yolo tensor in buildYoloNetwork()
      //Should be the exact same (in yolov5, 3)
  // yoloTensorInputs is a list of Itensors (the list is as long as m_YoloCount) that hold the output from the tensor before the yolo layer (gonna be a conv layer)
  // m_YoloTensors is a vector of type TensorInfo that was populated in parseConfigBlocks()
  if (m_YoloCount == yoloCountInputs) {
    assert((modelType != -1) && "\nCould not determine model type"); 

    uint64_t outputSize = 0;
    // Iterate over every TensorInfo object
    for (uint j = 0; j < yoloCountInputs; ++j) {
      TensorInfo& curYoloTensor = m_YoloTensors.at(j);
      // modelType for Yolov5 should be 1, so take the else
        // Set outputSize to be the previous layers x and y dimensions (which were stored in the TensorInfo struct) and numBBoxes (which is 9 in yolov5) all multiplied
      if (modelType == 3 || modelType == 4 || modelType == 5)
        outputSize = curYoloTensor.numBBoxes;
      else
        outputSize += curYoloTensor.gridSizeX * curYoloTensor.gridSizeY * curYoloTensor.numBBoxes;
    }

    // YoloLayer is a class defined in yoloPlugins.h
      // Add the YoloLayer to the network as a plugin
    nvinfer1::IPluginV2* yoloPlugin = new YoloLayer(m_InputW, m_InputH, m_NumClasses, m_NewCoords, m_YoloTensors, outputSize,
        modelType, m_ScoreThreshold);
    assert(yoloPlugin != nullptr);
    // Add a plugin layer to the network, addPluginV2(The input tensors to the layer, The number of input tensors, The layer plugin)
    nvinfer1::IPluginV2Layer* yolo = network.addPluginV2(yoloTensorInputs, m_YoloCount, *yoloPlugin);
    assert(yolo != nullptr);
    // Set the name for the pluginLayer
    std::string yoloLayerName = "yolo";
    yolo->setName(yoloLayerName.c_str());

    // yolo->getOutput(x) returns an output tensor at index x
      // Get each output tensor and set their names to be what they represent (num_detections, etc.)
    std::string outputlayerName;
    nvinfer1::ITensor* num_detections = yolo->getOutput(0);
    outputlayerName = "num_detections";
    num_detections->setName(outputlayerName.c_str());
    nvinfer1::ITensor* detection_boxes = yolo->getOutput(1);
    outputlayerName = "detection_boxes";
    detection_boxes->setName(outputlayerName.c_str());
    nvinfer1::ITensor* detection_scores = yolo->getOutput(2);
    outputlayerName = "detection_scores";
    detection_scores->setName(outputlayerName.c_str());
    nvinfer1::ITensor* detection_classes = yolo->getOutput(3);
    outputlayerName = "detection_classes";
    detection_classes->setName(outputlayerName.c_str());
    // Mark a tensor as a network output 
    network.markOutput(*num_detections);
    network.markOutput(*detection_boxes);
    network.markOutput(*detection_scores);
    network.markOutput(*detection_classes);

    // I was just fucking around with this, didn't work
    // std::cout << "detection_classes is network output?: " << detection_classes->isNetworkOutput() << "\n";
    // std::cout << "detection_classes dimensions: " << detection_classes->getDimensions() << "\n";
  }
  else {
    std::cerr << "\nError in yolo cfg file" << std::endl;
    assert(0);
  }

  std::cout << "\nOutput YOLO blob names: " << std::endl;
  for (auto& tensor : m_YoloTensors)
    std::cout << tensor.blobName << std::endl;

  int nbLayers = network.getNbLayers();
  std::cout << "\nTotal number of YOLO layers: " << nbLayers << "\n" << std::endl;

  return NVDSINFER_SUCCESS;
}

std::vector<std::map<std::string, std::string>>
Yolo::parseConfigFile(const std::string cfgFilePath)
{
  assert(fileExists(cfgFilePath));
  std::ifstream file(cfgFilePath);
  assert(file.good());
  std::string line;
  std::vector<std::map<std::string, std::string>> blocks;
  std::map<std::string, std::string> block;

  while (getline(file, line)) {
    if (line.size() == 0 || line.front() == ' ' || line.front() == '#')
      continue;

    line = trim(line);
    // For example the line , [convolutional] , would take this if 
    if (line.front() == '[') {
      if (block.size() > 0) {
        blocks.push_back(block);
        block.clear();
      }
      std::string key = "type";
      std::string value = trim(line.substr(1, line.size() - 2));
      block.insert(std::pair<std::string, std::string>(key, value));
    }
    // For example the line, filters=32 , would take this else 
    else {
      int cpos = line.find('=');
      std::string key = trim(line.substr(0, cpos));
      std::string value = trim(line.substr(cpos + 1));
      block.insert(std::pair<std::string, std::string>(key, value));
    }
  }
  blocks.push_back(block);
  return blocks;
}

// Go through all the blocks and make sure specific blocks have the information we need
  // Specifically for us, we need correct [net], [yolo], 
void
Yolo::parseConfigBlocks()
{
  for (auto block : m_ConfigBlocks) {
    if (block.at("type") == "net") {
      assert((block.find("height") != block.end()) && "Missing 'height' param in network cfg");
      assert((block.find("width") != block.end()) && "Missing 'width' param in network cfg");
      assert((block.find("channels") != block.end()) && "Missing 'channels' param in network cfg");

      m_InputH = std::stoul(block.at("height"));
      m_InputW = std::stoul(block.at("width"));
      m_InputC = std::stoul(block.at("channels"));
      m_InputSize = m_InputC * m_InputH * m_InputW;

      if (block.find("letter_box") != block.end())
        m_LetterBox = std::stoul(block.at("letter_box"));
    }
    else if ((block.at("type") == "region") || (block.at("type") == "yolo"))
    {
      // Checks for the absolutely neccessary information
      assert((block.find("num") != block.end()) &&
          std::string("Missing 'num' param in " + block.at("type") + " layer").c_str());
      assert((block.find("classes") != block.end()) &&
          std::string("Missing 'classes' param in " + block.at("type") + " layer").c_str());
      assert((block.find("anchors") != block.end()) &&
          std::string("Missing 'anchors' param in " + block.at("type") + " layer").c_str());

      ++m_YoloCount;

      m_NumClasses = std::stoul(block.at("classes"));

      if (block.find("new_coords") != block.end())
        m_NewCoords = std::stoul(block.at("new_coords"));

      //TensorInfo is a struct defined in yolo.h
      TensorInfo outputTensor;

      // Add all anchor values to the TensorInfo instance
      std::string anchorString = block.at("anchors");
      while (!anchorString.empty()) {
        int npos = anchorString.find_first_of(',');
        if (npos != -1) {
          float anchor = std::stof(trim(anchorString.substr(0, npos)));
          outputTensor.anchors.push_back(anchor);
          anchorString.erase(0, npos + 1);
        }
        else {
          float anchor = std::stof(trim(anchorString));
          outputTensor.anchors.push_back(anchor);
          break;
        }
      }

      // Add all mask values to the TensorInfo instanc
      if (block.find("mask") != block.end()) {
        std::string maskString = block.at("mask");
        while (!maskString.empty()) {
          int npos = maskString.find_first_of(',');
          if (npos != -1) {
            int mask = std::stoul(trim(maskString.substr(0, npos)));
            outputTensor.mask.push_back(mask);
            maskString.erase(0, npos + 1);
          }
          else {
            int mask = std::stoul(trim(maskString));
            outputTensor.mask.push_back(mask);
            break;
          }
        }
      }
      
      // Add scale to the outputTensor
        // This does exist in the yolov5 config, so we set scaleXY = 2
      if (block.find("scale_x_y") != block.end())
        outputTensor.scaleXY = std::stof(block.at("scale_x_y"));
      else
        outputTensor.scaleXY = 1.0;

      // For yolov5, num = 9 , and for each yolo model we have mask = {0,1,2} , mask = {3,4,5}, mask = {6,7,8}
      // Essentially, if the number of masks is greater than 0, use that as the number of bboxes, if not then use the value at num
        //In our case, these numbers are the exact same (9) so it doesn't really matter
      outputTensor.numBBoxes = outputTensor.mask.size() > 0 ? outputTensor.mask.size() : std::stoul(trim(block.at("num")));
      
      // Adds a TensorInfo object to our YoloTensors variable. This output variable is made from information in the [yolo] block
      m_YoloTensors.push_back(outputTensor);
    }
    // We don't take this path with yolov5 config
    else if ((block.at("type") == "cls") || (block.at("type") == "reg")) {
      ++m_YoloCount;
      TensorInfo outputTensor;
      m_YoloTensors.push_back(outputTensor);
    }
    // Or this path
    else if (block.at("type") == "detect_v8") {
      ++m_YoloCount;

      m_NumClasses = std::stoul(block.at("classes"));
      
      TensorInfo outputTensor;
      m_YoloTensors.push_back(outputTensor);
    }
    // Or this path
    else if (block.at("type") == "detect_x") {
      ++m_YoloCount;
      TensorInfo outputTensor;

      std::vector<int> strides;

      std::string stridesString = block.at("strides");
      while (!stridesString.empty()) {
        int npos = stridesString.find_first_of(',');
        if (npos != -1) {
          int stride = std::stof(trim(stridesString.substr(0, npos)));
          strides.push_back(stride);
          stridesString.erase(0, npos + 1);
        }
        else {
          int stride = std::stof(trim(stridesString));
          strides.push_back(stride);
          break;
        }
      }

      for (uint i = 0; i < strides.size(); ++i) {
        int num_grid_y = m_InputH / strides[i];
        int num_grid_x = m_InputW / strides[i];
        for (int g1 = 0; g1 < num_grid_y; ++g1) {
          for (int g0 = 0; g0 < num_grid_x; ++g0) {
            outputTensor.anchors.push_back((float) g0);
            outputTensor.anchors.push_back((float) g1);
            outputTensor.mask.push_back(strides[i]);
          }
        }
      }

      m_YoloTensors.push_back(outputTensor);
    }
  }
}

void
Yolo::destroyNetworkUtils()
{
  for (uint i = 0; i < m_TrtWeights.size(); ++i)
    if (m_TrtWeights[i].count > 0)
      free(const_cast<void*>(m_TrtWeights[i].values));
  m_TrtWeights.clear();
}
