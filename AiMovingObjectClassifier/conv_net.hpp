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


/**
 Used to turn a non-template parm into a template parm
 */
template <int N>
struct NetSize{
    enum{value = N};
};



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
    Eigen::Matrix<double,32,96> m_outputImage;
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
    
    
    
    // Debug stuff
public:
    int m_loop_count = 0;
    decltype(auto) getDebugConvNet()
    {
        return m_convNet;
    }
    
};


void test_eigen_opencv(ObjectImages* storage);
void test_loadEigenImage(ObjectImages* storage,Eigen::Matrix<double,32,96>& output, int& imgId);
void test_saveEigenImage(ObjectImages* storage,Eigen::Matrix<double,32,96>& input, int imgId,int objectId);
//#include <Eigen/Dense>



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
    std::cout<<"\n My id = "<<my_id<<"\n";
    m_loop_count++;
    if(m_loop_count > 14) {
        int res = convForward();
        std::cout<<"\n m_loop_count = "<<m_loop_count<<" convForward res = "<<res<<"\n";
    }else {
        m_outputImage = m_currentImage;
    }
    test_saveEigenImage(m_storage, m_outputImage, my_id, 5);
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
    std::cout<<"\n";
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

template<typename HyperParm1,typename HyperParam2>
using NextLayer = Eigen::Map<Eigen::Matrix<double, HyperParm1::OutputRow::value, HyperParm1::OutputCol::value>,0,Eigen::Stride<HyperParam2::CalcInRow::value, 1>>;
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
    
//    m_currentImage
    
    ConvInputMat<Layer1> convInput;
    
    auto& layer1 = m_convNet.layer1;
    convInput.setZero();
//    convInput.block<Layer1::ImgDim::value,Layer1::ImgDim::value * Layer1::ImgDepth::value>(layer1.getPadding(),layer1.getPadding()* layer1.getImgDepth()) = m_currentImage;
    const int l1Row = layer1.getImageDim();
    const int l1Col = layer1.getImageDim()*layer1.getImgDepth();
    const int l1x = layer1.getPadding() * layer1.getImgDepth();
    const int l1y = layer1.getPadding();
    
    convInput.block(l1y,l1x,l1Row,l1Col) = m_currentImage;

//    std::cout<<"\n convInput = \n"<<convInput<<"\n";
    
//    ConvLayer<Layer1> convLayer;
    ConvLayerDyn<Layer1> convLayer(layer1.getCalcOutRow(),layer1.getCalcOutCol());
    print_size(convLayer);
    im2Col<Layer1>(convInput, convLayer);
    
    //testing_im2Col_old<layerDim, filterDim,padding,layerDepth,stride>(convInput, convLayer);
    std::cout<<"\n convLayer() = "<<convLayer.sum()<<"\n";
    print_size(convLayer);
    ConvResult<Layer1> result = layer1.filters * convLayer;
    print_size(result);
    // bias
    for (int i = 0; i < layer1.getFilterCount(); ++i) {
        result.row(i) = result.row(i).array() + layer1.bias(i);
    }
    //std::cout<<"\n result = \n"<<result<<"\n";
    
    // Relu
    
    result = result.cwiseMax(0);
    
    
    ConvOutput<Layer1> output;
    
    output.setZero();
    /// images side by side
//        Eigen::Map<Eigen::Matrix<double, Layer1::OutputRow::value, Layer1::OutputCol::value>,0,Eigen::Stride<1, Layer1::OutputCol::value>>(output.data() )
//        = Eigen::Map<Eigen::Matrix<double, Layer1::OutputRow::value, Layer1::OutputCol::value>,0,Eigen::Stride<Layer1::OutputRow::value, 1>>(result.data() );
    
    // interleave
    std::cout<<"\n Layer1::OutputRow::value = "<<Layer1::OutputRow::value<<" Layer1::OutputCol::value = "<<Layer1::OutputCol::value<<"\n";
//    Eigen::Map<Eigen::Matrix<double,Layer1::OutputRow::value,Layer1::OutputCol::value>,0,Eigen::Stride<Layer1::OutputRow::value, 1>>(output.data() )
//    = Eigen::Map<Eigen::Matrix<double,Layer1::OutputRow::value, Layer1::OutputCol::value>,0,Eigen::Stride<1, Layer1::OutputCol::value>>(result.data() );
    
    

    ConvInputMat<Layer2> convInput2;
    convInput2.setZero();
    
    auto& layer2 = m_convNet.layer2;
    
    
    const int move1 = layer2.getPaddingOffsetX() + layer2.getPaddingOffsetY();
    
    NextLayer<Layer1,Layer2>(convInput2.data() + move1) = FromLayer<Layer1>(result.data());
    
    
    ConvLayerDyn<Layer1> convLayer2(layer2.getCalcOutRow(),layer2.getCalcOutCol());
    print_size(convLayer2);
    im2Col<Layer1>(convInput2, convLayer2);
    
    std::cout<<"\n convLayer2() = "<<convLayer2.sum()<<"\n";
    print_size(convLayer2);
    ConvResult<Layer2> result2 = layer2.filters * convLayer2;
    
    for (int i = 0; i < layer2.getFilterCount(); ++i) {
        result2.row(i) = result2.row(i).array() + layer2.bias(i);
    }

    result2 = result2.cwiseMax(0); // relu
    
    // output to image
    //std::cout<<"\n output = \n"<<output<<"\n";
//    Eigen::Map<Eigen::Matrix<double,Layer2::OutputRow::value,Layer2::OutputCol::value>,0,Eigen::Stride<Layer2::OutputRow::value, 1>>(output.data() )
//    = Eigen::Map<Eigen::Matrix<double,Layer2::OutputRow::value, Layer2::OutputCol::value>,0,Eigen::Stride<1, Layer2::OutputCol::value>>(result2.data() );
    auto& layer4 = m_convNet.layer4;
    const int move4 = layer4.getPaddingOffsetX() + layer4.getPaddingOffsetY();
    NextLayer<Layer2,Layer4>(output.data() + move4) = FromLayer<Layer2>(result2.data());
    std::cout<<"\n----------end skeleton--------------\n";
    m_outputImage = output;
    return -1;
}

#endif /* conv_net_hpp */
