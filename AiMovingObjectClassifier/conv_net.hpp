//
//  conv_net.hpp
//  AiMovingObjectClassifier
//
//  Created by o_0 on 2017-01-12.
//  Copyright © 2017 Magnus. All rights reserved.
//

#ifndef conv_net_hpp
#define conv_net_hpp
#include <Eigen/Dense>
#include <iostream>
#include "debug_helper_info.h"
#include "utils.hpp"
#include "neural_net_common.hpp"
#include <opencv/cv.h>


/**
 ConvHyperParam hyper parameters for a layer on the convolutional neural network (CNN)

 @param ImageDim input dimensions without depth ( 32x32 x3(rgb) -> 32)
 @param ImageDepth image depth , depth ( 32x32 x3(rgb) -> 3), or equal to number of filters on the prev layer
 @param FilterDim1 dimensions without depth ( filters depth is the same as image)
 @param FilterCount1 how many filters this layer has
 @param Stride1 what stride the layer should take
 @param Padding1 if the input should be padded, (the padding is done autmatic so do not include it in imgdim)
 @return is used to set the parm on the convnet
 */
template <int ImageDim,int ImageDepth,int FilterDim1,int FilterCount1, int Stride1,int Padding1 = 0,int FilterLength1 = FilterDim1*FilterDim1*ImageDepth,int CalcRowIn1 = ImageDim + 2*Padding1,int CalcColIn1 = (ImageDim + 2*Padding1)*ImageDepth,int CalcRowOut1 = FilterDim1*FilterDim1*ImageDepth, int CalcOutDimTmp = (ImageDim - FilterDim1 + 2*Padding1)/Stride1 + 1,int CalcColOut1 = CalcOutDimTmp*CalcOutDimTmp,int CalcOutDimColTmp = CalcOutDimTmp*FilterCount1>
struct ConvHyperParam {
    //    const int In{ImageDepth};
    using ImgDim = NetSize<ImageDim>;
    using ImgDepth = NetSize<ImageDepth>;
    using FilterDim = NetSize<FilterDim1>;
    using FilterCount = NetSize<FilterCount1>;
    using FilterLength = NetSize<FilterLength1>;
    using Stride = NetSize<Stride1>;
    using Padding = NetSize<Padding1>;
    
    // calculation parm
    using CalcInRow = NetSize<CalcRowIn1>;
    using CalcInCol = NetSize<CalcColIn1>;
    using CalcOutRow = NetSize<CalcRowOut1>;
    using CalcOutCol = NetSize<CalcColOut1>;
    
    using OutputRow = NetSize<CalcOutDimTmp>;
    using OutputCol = NetSize<CalcOutDimColTmp>;
    
};


/**
 ConvNetLayer layer on the convolutional neural network (CNN)
 
 @param HyperParam this layers hyper parameters
 */
template <typename HyperParam>
class ConvNetLayer {
public:
    // so it can be access later
    using ImgDim		 = typename HyperParam::ImgDim;
    using ImgDepth		 = typename HyperParam::ImgDepth;
    using FilterDim		 = typename HyperParam::FilterDim;
    using FilterCount	 = typename HyperParam::FilterCount;
    using FilterLength	 = typename HyperParam::FilterLength;
    using Stride		 = typename HyperParam::Stride;
    using Padding		 = typename HyperParam::Padding;
    
    // Calc helper
    using CalcInRow = typename HyperParam::CalcInRow;
    using CalcInCol = typename HyperParam::CalcInCol;
    using CalcOutRow = typename HyperParam::CalcOutRow;
    using CalcOutCol = typename HyperParam::CalcOutCol;
    
    using OutputRow = typename HyperParam::OutputRow;
    using OutputCol = typename HyperParam::OutputCol;
    
    
    // compile time geters for the parm,
    constexpr int getImageDim() {return ImgDim::value;}
    constexpr int getImgDepth() {return ImgDepth::value;}
    constexpr int getFilterDim() {return FilterDim::value;}
    constexpr int getFilterCount() {return FilterCount::value;}
    constexpr int getFilterLength() {return FilterLength::value;}
    constexpr int getStride() {return Stride::value;}
    constexpr int getPadding() {return Padding::value;}
    
    
    /**
     When padding is used there needs to be a offset into the eigen matrix to properly load data

     @return padding offset x-axis
     */
    constexpr int getPaddingOffsetX() {return ImgDim::value * Padding::value*ImgDepth::value;}
    /**
     When padding is used there needs to be a offset into the eigen matrix to properly load data
     
     @return padding offset y-axis
     */
    constexpr int getPaddingOffsetY() {return Padding::value;}
    
    
    constexpr int getCalcInRow() {return CalcInRow::value;}
    constexpr int getCalcInCol() {return CalcInCol::value;}
    constexpr int getCalcOutRow() {return CalcOutRow::value;}
    constexpr int getCalcOutCol() {return CalcOutCol::value;}
    
    
    constexpr int getOutputRow() {return OutputRow::value;}
    constexpr int getOutputCol() {return OutputCol::value;}
    
    using type = HyperParam;
    
    // holds the filters for this layer
    Eigen::Matrix<double, FilterCount::value,FilterLength::value> filters;
    // holds the bias for this layer
    Eigen::Matrix<double, FilterCount::value,1> bias;
};

/**
 Convolutional neural network (CNN) , 4 layers
 takes a ConvNetLayer as template parm with the correct settings
  This keeps track of the weigths and bias of all layers
 this is then used in the learning moudle to preforme the training.
 */
template <typename L1,typename L2,typename L3,typename L4>
class ConvNet {
    
public:
    ConvNetLayer<L1> layer1;
    ConvNetLayer<L2> layer2;
    ConvNetLayer<L3> layer3;
    ConvNetLayer<L4> layer4;
    using Layer1 = L1;
    using Layer2 = L2;
    using Layer3 = L3;
    using Layer4 = L4;
    void randomizeAll() {
        layer1.filters.setRandom();
        layer2.filters.setRandom();
        layer3.filters.setRandom();
        layer4.filters.setRandom();
        layer1.bias.setRandom();
        layer2.bias.setRandom();
        layer3.bias.setRandom();
        layer4.bias.setRandom();
    }
};

class ObjectImages;

template<class T>
using LayerOutputImage = Eigen::Matrix<double,T::OutputRow::value,T::OutputCol::value>;
template<class T, int Cols = T::ImgDim::value * T::ImgDepth::value>
using LayerInputImage = Eigen::Matrix<double,T::ImgDim::value,Cols>;
// LearningModule used to train the networks
template<class T>
class LearningModule {
    ConvNet<typename T::Layer1,typename T::Layer2,typename T::Layer3,typename T::Layer4>& m_convNet;
    decltype(auto) getConvNet()
    {
        return m_convNet;
    }
    ObjectImages* m_storage = nullptr;
    Eigen::Matrix<double,32,96> m_currentImage;
    LayerOutputImage< typename T::Layer1> m_outputImage1;
    LayerOutputImage< typename T::Layer2> m_outputImage2;
    LayerOutputImage< typename T::Layer3> m_outputImage3;
    LayerOutputImage< typename T::Layer4> m_outputImage4;
    LayerInputImage< typename T::Layer1> m_outputImage;
    // methods
public:
    void trainlastImg();
    int convForward();
    
public:
    
    template <typename T1>
    LearningModule(T1& conv) : m_convNet(conv) {
        m_currentImage.setZero();
        m_outputImage.setZero();
    }
    
    void setStorage(ObjectImages* storage)
    {
        m_storage = storage;
    }
    
    
private:
    template<typename  T1>
    struct LayerImageOutput
    {
        cv::Mat eigenTest2 = cv::Mat::zeros(T1::OutputRow::value,T1::OutputCol::value,CV_64FC1);
        bool saveEigenImg(int imgId,Eigen::Matrix<double,T1::OutputRow::value,T1::OutputCol::value>& input)
        {
            Eigen::Map<Eigen::Matrix<double,T1::OutputRow::value,T1::OutputCol::value,Eigen::RowMajor>> b((double*)eigenTest2.data + 0);
            b = (input.array()*(double)1/255).matrix();
            return true;
        }
        void fillMat(cv::Mat& output){
            if(output.empty() || eigenTest2.empty()){
                std::cout<<"\ngetLastImg empty\n";
                return;
            }
            eigenTest2.copyTo(output);
        }
        
        void fillMat(cv::Mat& output,cv::Rect& bound){
            if(output.empty() || eigenTest2.empty()){
                std::cout<<"\ngetLastImg empty\n";
                return;
            }
//            std::cout<<"\n bound = "<< bound<<"\n";
//            cv::Mat dst = output(bound);
            eigenTest2.copyTo(output(bound));
            
//            eigenTest2.assignTo(dst);
        }
        
    };
//    ConvHyperParam<T::Layer1::ImgDim::value, T::Layer1::ImgDim::value, 3, T::Layer1::ImgDim::value, 1,1>;
    LayerImageOutput<ConvHyperParam<T::Layer1::ImgDim::value, T::Layer1::ImgDepth::value, 3, T::Layer1::ImgDepth::value, 1,1>> layer0Out;
    LayerImageOutput<typename T::Layer1> layer1Out;
    LayerImageOutput<typename T::Layer2> layer2Out;
    LayerImageOutput<typename T::Layer3> layer3Out;
    LayerImageOutput<typename T::Layer4> layer4Out;
    // Debug stuff
public:
    bool debug_show_layeroutput = false;

    void fillMat(cv::Mat& output,cv::Rect& bounding,int layerLevel){
        if(debug_show_layeroutput == false) return;
        switch(layerLevel)
        {
            case 0: layer0Out.fillMat(output, bounding);break;
            case 1: layer1Out.fillMat(output, bounding);break;
            case 2: layer2Out.fillMat(output, bounding);break;
            case 3: layer3Out.fillMat(output, bounding);break;
            case 4: layer4Out.fillMat(output, bounding);break;
        }
    }
    void fillMat(cv::Mat& output,int layerLevel){
        if(debug_show_layeroutput == false) return;
        switch(layerLevel)
        {
            case 1: layer1Out.fillMat(output);break;
            case 2: layer2Out.fillMat(output);break;
            case 3: layer3Out.fillMat(output);break;
            case 4: layer4Out.fillMat(output);break;
        }
    }
    
    int m_loop_count = 0;
    decltype(auto) getDebugConvNet()
    {
        return m_convNet;
    }
    
};


void test_eigen_opencv(ObjectImages* storage);
void test_loadEigenImage(ObjectImages* storage,Eigen::Matrix<double,32,96>& output, int& imgId);
void test_saveEigenImage(ObjectImages* storage,Eigen::Matrix<double,32,96>& input, int imgId,int objectId);

/**
 Trains on the last network, this will use the storage and get the last image, try to identify it and return
 it will save a image to be displayed
 */
template<typename T>
void LearningModule<T>::trainlastImg()
{
//    test_eigen_opencv(m_storage);
    int my_id = -1;
    test_loadEigenImage(m_storage, m_currentImage, my_id);
//    std::cout<<"\n My id = "<<my_id<<"\n";
    m_loop_count++;
    if(m_loop_count > 14) {
        my_id = convForward();
//        std::cout<<"\n m_loop_count = "<<m_loop_count<<" convForward res = "<<res<<"\n";
    }else {
//        m_outputImage = m_currentImage;
    }
    //    test_saveEigenImage(m_storage, m_outputImage, my_id, 5);
    layer0Out.saveEigenImg(my_id, m_outputImage);
    layer1Out.saveEigenImg(my_id, m_outputImage1);
    layer2Out.saveEigenImg(my_id, m_outputImage2);
    layer3Out.saveEigenImg(my_id, m_outputImage3);
    layer4Out.saveEigenImg(my_id, m_outputImage4);
}

// ConvInputMat used to make the input layer matrix with padding and so on
template<typename HyperParam>
using ConvInputMat = Eigen::Matrix<double, HyperParam::CalcInRow::value, HyperParam::CalcInCol::value>;

// ConvLayer used to make the output layer so it can store the im2col data
template<typename HyperParam>
using ConvLayer = Eigen::Matrix<double, HyperParam::CalcOutRow::value, HyperParam::CalcOutCol::value>;

// ConvLayerDyn used to make the output layer so it can store the im2col data, dyn is better for big matrixs
template<typename HyperParam>
using ConvLayerDyn = Eigen::MatrixXd;

// ConvResult the result of the filter*ConvLayerDyn is stored here
template<typename HyperParam>
using ConvResult = Eigen::Matrix<double,HyperParam::FilterCount::value,HyperParam::CalcOutCol::value>;

// ConvOutput the correct shaped output data for this layer
template<typename HyperParam>
using ConvOutput = Eigen::Matrix<double,HyperParam::OutputRow::value,HyperParam::OutputCol::value>;

// GetOutputMap map the ConvResult so the output can be set
template<typename HyperParam>
using GetOutputMap = Eigen::Map<Eigen::Matrix<double,HyperParam::OutputRow::value,HyperParam::OutputRow::value>,0,Eigen::Stride<HyperParam::FilterCount::value, HyperParam::OutputCol::value>>;

//SetOutputMap map the ConvOutput so the output can be set
template<typename HyperParam>
using SetOutputMap = Eigen::Map<Eigen::Matrix<double,HyperParam::OutputRow::value,HyperParam::OutputRow::value>,0,Eigen::Stride<1, HyperParam::OutputCol::value>>;


/**
 im2Col transform a MxNxD matrix into a AxB matrix ,
 where every col(B) represent a output entry with the input field A transformed to be the data to dot with the filters

 @param HyperParam
 @param input ConvInputMat sized matrix
 @param output ConvLayer sized matrix
 */
template<typename HyperParam>
void im2Col(const Eigen::Matrix<double,HyperParam::CalcInRow::value,HyperParam::CalcInCol::value>& input,
             Eigen::Matrix<double,HyperParam::CalcOutRow::value,HyperParam::CalcOutCol::value>& output)
{
    
    const int resultDim = HyperParam::CalcOutCol::value;
    const int filterDim = HyperParam::FilterDim::value;
    const int depth = HyperParam::ImgDepth::value;
    const int stride = HyperParam::Stride::value;
    const int padding = HyperParam::Padding::value;
    const int imageDim = HyperParam::ImgDim::value;
    
    const int numcols = (imageDim - filterDim + 2*padding)/stride + 1;//
    for(int i = 0; i < resultDim  ;++i)
    {
        int x = i%(numcols);
        int y = (i - x)/(numcols);
        x *=stride;
        y *=stride*depth;
        
        for (int r = 0; r < filterDim ; ++r) {
            
            output.col(i).block(r*filterDim*depth, 0, filterDim*depth, 1) = input.template block<filterDim,filterDim*depth>(x,y).template block<1,filterDim*depth>(r,0).transpose();
            
        }
        
    }
}

/**
 im2Col transform a MxNxD matrix into a AxB matrix ,
 where every col(B) represent a output entry with the input field A transformed to be the data to dot with the filters
 
 @param HyperParam
 @param input ConvInputMat sized matrix
 @param output ConvLayerDyn sized matrix
 */
template<typename HyperParam>
void im2Col(const Eigen::Matrix<double,HyperParam::CalcInRow::value,HyperParam::CalcInCol::value>& input,
            Eigen::MatrixXd& output)
{
    
    const int resultDim = HyperParam::CalcOutCol::value;
    const int filterDim = HyperParam::FilterDim::value;
    const int depth = HyperParam::ImgDepth::value;
    const int stride = HyperParam::Stride::value;
    const int padding = HyperParam::Padding::value;
    const int imageDim = HyperParam::ImgDim::value;
    
    const int numcols = (imageDim - filterDim + 2*padding)/stride + 1;//
//    std::cout<<"\n";
    for(int i = 0; i < resultDim  ;++i)
    {
        int x = i%(numcols);
        int y = (i - x)/(numcols);
        x *=stride;
        y *=stride*depth;
        
        for (int r = 0; r < filterDim ; ++r) {

            output.col(i).block(r*filterDim*depth, 0, filterDim*depth, 1) = input.block(x,y,filterDim,filterDim*depth).block(r,0,1,filterDim*depth).transpose();
            
        }
        
    }
}

/**
 NextLayer maps the ConvInputMat in a way so that the ConvResult can be correcly reshaped and set
 @param HyperParam1 from layer type
 @param HyperParam2 to layer type
 */
template<typename HyperParm1,typename HyperParam2>
using NextLayer = Eigen::Map<Eigen::Matrix<double, HyperParm1::OutputRow::value, HyperParm1::OutputCol::value>,0,Eigen::Stride<HyperParam2::CalcInRow::value, 1>>;

/**
 SideBySide maps thid data to a single value grayscale square
 @param HyperParam to layer type
 */
template<typename HyperParm>
using SideBySide = Eigen::Map<Eigen::Matrix<double, HyperParm::OutputRow::value, HyperParm::OutputRow::value>>;
/**
 FromResult maps the ConvInputMat in a way so that the ConvResult can be correcly reshaped and set
 @param HyperParam to layer type
 */
template<typename HyperParm>
using FromResult = Eigen::Map<Eigen::Matrix<double, HyperParm::OutputRow::value, HyperParm::OutputRow::value>,0,Eigen::Stride<HyperParm::OutputCol::value, HyperParm::FilterCount::value>>;


/**
 SideBySide maps thid data to a single value grayscale square
 @param HyperParam to layer type
 */
template<typename HyperParm>
using SideBySideInput = Eigen::Map<Eigen::Matrix<double, HyperParm::ImgDim::value, HyperParm::ImgDim::value>>;
template<typename HyperParm,int StrideCol = HyperParm::ImgDim::value * HyperParm::ImgDepth::value>
using FromInput = Eigen::Map<Eigen::Matrix<double, HyperParm::ImgDim::value, HyperParm::ImgDim::value>,0,Eigen::Stride<StrideCol, 1>>;

template<typename HyperParm>
constexpr int outputDepthStride()
{
    return HyperParm::CalcOutCol::value;
}
//Eigen::Matrix<double,HyperParam::FilterCount::value,HyperParam::CalcOutCol::value>;

// HyperParm::OutputCol::value
// Eigen::Map<double, HyperParm1::OutputRow::value, HyperParm1::OutputCol::value>,0,Eigen::Stride<1, 25>>(output.data() )

/**
 FromLayer maps the ConvResult in a way so that the NextLayer map can be correcly reshaped and set
 @param HyperParam from layer type
 */
template<typename HyperParm>
using FromLayer = Eigen::Map<Eigen::Matrix<double,HyperParm::OutputRow::value, HyperParm::OutputCol::value>,0,Eigen::Stride<1, HyperParm::OutputCol::value>>;


/**
 Does on forward pass with the conv net, on the m_currentImage

 @return a id number
 */
template<typename T>
int LearningModule<T>::convForward()
{
    using Layer1 = typename decltype(m_convNet.layer1)::type;
    using Layer2 = typename decltype(m_convNet.layer2)::type;
    using Layer3 = typename decltype(m_convNet.layer3)::type;
    using Layer4 = typename decltype(m_convNet.layer4)::type;
    
    auto& layer1 = m_convNet.layer1;
    auto& layer2 = m_convNet.layer2;
    auto& layer3 = m_convNet.layer3;
    auto& layer4 = m_convNet.layer4;
    
//    const int move1 = layer1.getPaddingOffsetX() + layer1.getPaddingOffsetY();
    const int move2 = layer2.getPaddingOffsetX() + layer2.getPaddingOffsetY();
    const int move3 = layer3.getPaddingOffsetX() + layer3.getPaddingOffsetY();
    const int move4 = layer4.getPaddingOffsetX() + layer4.getPaddingOffsetY();
//    m_currentImage
    
    ConvInputMat<Layer1> convInput;
    
//    auto& layer1 = m_convNet.layer1;
    convInput.setZero();
//    convInput.block<Layer1::ImgDim::value,Layer1::ImgDim::value * Layer1::ImgDepth::value>(layer1.getPadding(),layer1.getPadding()* layer1.getImgDepth()) = m_currentImage;
    const int l1Row = layer1.getImageDim();
    const int l1Col = layer1.getImageDim()*layer1.getImgDepth();
    const int l1x = layer1.getPadding() * layer1.getImgDepth();
    const int l1y = layer1.getPadding();
    //convinput my be padded , this stores the m_currentImage in correct position
    convInput.block(l1y,l1x,l1Row,l1Col) = m_currentImage;

    // used for the im2col layer output
    ConvLayerDyn<Layer1> convLayer(layer1.getCalcOutRow(),layer1.getCalcOutCol());
//    print_size(convLayer);
    
    // generate the convLayer so the colums represent one output cell, that is multiplyed by the filters
    im2Col<Layer1>(convInput, convLayer);
    
//    std::cout<<"\n convLayer() = "<<convLayer.sum()<<"\n";
//    print_size(convLayer);
    
    // Do the actual forward propagation.
    // this is equal to have the filter move accross the inputimage in a sliding window fashion
    // now its only a simple matrix multipley operation tanks to im2col
    ConvResult<Layer1> result1 = layer1.filters * convLayer;
//    print_size(result1);
    // Add bias
    for (int i = 0; i < layer1.getFilterCount(); ++i) {
        result1.row(i) = result1.row(i).array() + layer1.bias(i);
    }

    // Relu
    result1 = result1.cwiseMax(0);
    
   

    //the padding offset into the next layer
//    const int move1 = layer2.getPaddingOffsetX() + layer2.getPaddingOffsetY();
    //////// layer 2 ////////////
    
    ConvInputMat<Layer2> convInput2;
    convInput2.setZero();
    // inserts the data to the input of the next layer
    NextLayer<Layer1,Layer2>(convInput2.data() + move2) = FromLayer<Layer1>(result1.data());
    
    ConvLayerDyn<Layer2> convLayer2(layer2.getCalcOutRow(),layer2.getCalcOutCol());
    im2Col<Layer2>(convInput2, convLayer2);
    ConvResult<Layer2> result2 = layer2.filters * convLayer2;
    for (int i = 0; i < layer2.getFilterCount(); ++i) {
        result2.row(i) = result2.row(i).array() + layer2.bias(i);
    }
    result2 = result2.cwiseMax(0); // relu
    
    //////// layer 3 ////////////
    
    ConvInputMat<Layer3> convInput3;
    convInput3.setZero();
    // inserts the data to the input of the next layer
    NextLayer<Layer2,Layer3>(convInput3.data() + move3) = FromLayer<Layer2>(result2.data());
    
    ConvLayerDyn<Layer3> convLayer3(layer3.getCalcOutRow(),layer3.getCalcOutCol());
    im2Col<Layer3>(convInput3, convLayer3);
    ConvResult<Layer3> result3 = layer3.filters * convLayer3;
    for (int i = 0; i < layer3.getFilterCount(); ++i) {
        result3.row(i) = result3.row(i).array() + layer3.bias(i);
    }
    result3 = result3.cwiseMax(0); // relu
    
    
    // interleave
//    std::cout<<"\n Layer1::OutputRow::value = "<<Layer1::OutputRow::value<<" Layer1::OutputCol::value = "<<Layer1::OutputCol::value<<"\n";
//    std::cout<<"\n Layer2::OutputRow::value = "<<Layer2::OutputRow::value<<" Layer1::OutputCol::value = "<<Layer2::OutputCol::value<<"\n";
//    std::cout<<"\n Layer4::OutputRow::value = "<<Layer4::OutputRow::value<<" Layer4::OutputCol::value = "<<Layer4::OutputCol::value<<"\n";
//    
//    std::cout<<"\n outputDepthStride<Layer2>() = "<< outputDepthStride<Layer2>() <<" layer4 = " << outputDepthStride<Layer4>() <<"\n";
//

    
    //////// layer 4 ////////////
    
    ConvInputMat<Layer4> convInput4;
    convInput4.setZero();
    // inserts the data to the input of the next layer
    NextLayer<Layer3,Layer4>(convInput4.data() + move4) = FromLayer<Layer3>(result3.data());
    
    ConvLayerDyn<Layer4> convLayer4(layer4.getCalcOutRow(),layer4.getCalcOutCol());
    im2Col<Layer4>(convInput4, convLayer4);
    ConvResult<Layer4> result4 = layer4.filters * convLayer4;
    for (int i = 0; i < layer4.getFilterCount(); ++i) {
        result4.row(i) = result4.row(i).array() + layer4.bias(i);
    }
    result4 = result4.cwiseMax(0); // relu

    // finnished
    ConvOutput<Layer1> output;
    output.setZero();
    NextLayer<Layer4,Layer4>(output.data() + move4) = FromLayer<Layer4>(result4.data());
//    std::cout<<"\n----------end skeleton--------------\n";
    
    
    if(debug_show_layeroutput) {
        for (int i = 0; i < layer1.getImgDepth() ; ++i) {
            SideBySideInput<Layer1>(m_outputImage.data() + i* layer1.getImageDim()*layer1.getImageDim() )
            = FromInput<Layer1>(m_currentImage.data() + i*layer1.getImageDim());
            //        = Eigen::Map<Eigen::Matrix<double, 32, 32>,0,Eigen::Stride<96,3>>(result2.data() + i);
        }
        for (int i = 0; i < layer1.getFilterCount() ; ++i) {
            SideBySide<Layer1>(m_outputImage1.data() + i* outputDepthStride<Layer1>() )
            = FromResult<Layer1>(result1.data() + i);
            //        = Eigen::Map<Eigen::Matrix<double, 32, 32>,0,Eigen::Stride<96,3>>(result2.data() + i);
        }
        for (int i = 0; i < layer2.getFilterCount() ; ++i) {
            SideBySide<Layer2>(m_outputImage2.data() + i* outputDepthStride<Layer2>() )
            = FromResult<Layer2>(result2.data() + i);
            //        = Eigen::Map<Eigen::Matrix<double, 32, 32>,0,Eigen::Stride<96,3>>(result2.data() + i);
        }
        for (int i = 0; i < layer3.getFilterCount() ; ++i) {
            SideBySide<Layer3>(m_outputImage3.data() + i* outputDepthStride<Layer3>() )
            = FromResult<Layer3>(result3.data() + i);
            //        = Eigen::Map<Eigen::Matrix<double, 32, 32>,0,Eigen::Stride<96,3>>(result2.data() + i);
        }
        for (int i = 0; i < layer4.getFilterCount() ; ++i) {
            SideBySide<Layer4>(m_outputImage4.data() + i* outputDepthStride<Layer4>() )
            = FromResult<Layer4>(result4.data() + i);
            //        = Eigen::Map<Eigen::Matrix<double, 32, 32>,0,Eigen::Stride<96,3>>(result2.data() + i);
        }
    }

    
//    Eigen::Map<Eigen::Matrix<double,32 , 96>,0,Eigen::Stride<1, 1>>(m_outputImage.data())
//    = Eigen::Map<Eigen::Matrix<double,32 , 96>,0,Eigen::Stride<1, 1024>>(result2.data());//FromLayer<Layer2>(result2.data());
//    m_outputImage = output;
    return -1;
}

#endif /* conv_net_hpp */
