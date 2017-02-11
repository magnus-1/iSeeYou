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
template <int ImageDim,int ImageDepth,int FilterDim1,int FilterCount1, int Stride1,int Padding1 = 0,int FilterLength1 = FilterDim1*FilterDim1*ImageDepth,int Filter2dWidth1 = FilterDim1*ImageDepth,int CalcRowIn1 = ImageDim + 2*Padding1,int CalcColIn1 = (ImageDim + 2*Padding1)*ImageDepth,int CalcRowOut1 = FilterDim1*FilterDim1*ImageDepth, int CalcOutDimTmp = (ImageDim - FilterDim1 + 2*Padding1)/Stride1 + 1,int CalcColOut1 = CalcOutDimTmp*CalcOutDimTmp,int CalcOutDimColTmp = CalcOutDimTmp*FilterCount1>
struct ConvHyperParam {
    //    const int In{ImageDepth};
    using ImgDim = NetSize<ImageDim>;
    using ImgDepth = NetSize<ImageDepth>;
    using FilterDim = NetSize<FilterDim1>;
    using FilterCount = NetSize<FilterCount1>;
    using FilterLength = NetSize<FilterLength1>;
    using Stride = NetSize<Stride1>;
    using Padding = NetSize<Padding1>;
    
    
    using Filter2dWidth = NetSize<Filter2dWidth1>;
    
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
    
    
    using Filter2dWidth	 = typename HyperParam::Filter2dWidth;
    
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
    
    constexpr int getFilter2dWidth() {return Filter2dWidth::value;}
    
    // (ImgDim::value + 2*Padding::value)* Padding::value*ImgDepth::value
    // ImgDim::value * Padding::value*ImgDepth::value
    /**
     When padding is used there needs to be a offset into the eigen matrix to properly load data

     @return padding offset x-axis
     */
    constexpr int getPaddingOffsetX() {return (ImgDim::value + 2*Padding::value)* Padding::value*ImgDepth::value;}
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

template<class T>
class DebugLayerOutputImage
{
public:
    LayerOutputImage< typename T::Layer1> m_outputImage1;
    LayerOutputImage< typename T::Layer2> m_outputImage2;
    LayerOutputImage< typename T::Layer3> m_outputImage3;
    LayerOutputImage< typename T::Layer4> m_outputImage4;
    void setZero()
    {
        m_outputImage1.setZero();
        m_outputImage2.setZero();
        m_outputImage3.setZero();
        m_outputImage4.setZero();
    }
};

struct IdentificationResult {
    double probability = 0.0;
    int prevId = -1;
    int resultId = -1;
    IdentificationResult(double prob,int prevId_,int resultId_) : probability(prob), prevId(prevId_), resultId(resultId_) {}
    IdentificationResult() : probability(0.0), prevId(-1), resultId(-1) {}
};
using IdentificationResult_t = struct IdentificationResult;
// LearningModule used to train the networks
template<class T>
class LearningModule {
    ConvNet<typename T::Layer1,typename T::Layer2,typename T::Layer3,typename T::Layer4>& m_convNet;
    classfier::training_conf m_conf_fc;  // fully connected layers
    classfier::training_conf m_conf_conv; // conv layers
    decltype(auto) getConvNet()
    {
        return m_convNet;
    }
    ObjectImages* m_storage = nullptr;
    Eigen::Matrix<double,32,96> m_currentImage;
//    LayerOutputImage< typename T::Layer1> m_outputImage1;
//    LayerOutputImage< typename T::Layer2> m_outputImage2;
//    LayerOutputImage< typename T::Layer3> m_outputImage3;
//    LayerOutputImage< typename T::Layer4> m_outputImage4;
    DebugLayerOutputImage<T> m_debugOutput;
    LayerInputImage< typename T::Layer1> m_outputImage;
    // methods
public:
    IdentificationResult_t checkLastImg();
    void trainlastImg();
    void trainEpoch(int epochCount,int storageSize);
    IdentificationResult_t convForward();
    int convTrainBackprop(int imgId);
    
public:
    
    template <typename T1>
    LearningModule(T1& conv) : m_convNet(conv) {
        m_currentImage.setZero();
        m_outputImage.setZero();
        m_outputImage.setZero();
    }
    
    void setStorage(ObjectImages* storage)
    {
        m_storage = storage;
    }
    
    void setConfigurationConv(classfier::training_conf& conf)
    {
        m_conf_conv = conf;
    }
    
private:
    template<typename  T1>
    struct LayerImageOutput
    {
        cv::Mat eigenTest2 = cv::Mat::zeros(T1::OutputRow::value,T1::OutputCol::value,CV_64FC1);
        bool saveEigenImg(int imgId,Eigen::Matrix<double,T1::OutputRow::value,T1::OutputCol::value>& input)
        {
            Eigen::Map<Eigen::Matrix<double,T1::OutputRow::value,T1::OutputCol::value,Eigen::RowMajor>> b((double*)eigenTest2.data + 0);
            const double min = input.minCoeff();
            const double max = input.maxCoeff();
            b = ((input.array() - min)/(max - min)).matrix(); //*(double)1/255).matrix();
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
    
    // remove
    int tmp_func_remove_convforward();
    
};


void test_eigen_opencv(ObjectImages* storage);
void test_loadEigenImage(ObjectImages* storage,Eigen::Matrix<double,32,96>& output, int& imgId);
bool test_loadEigenImageAt(ObjectImages* storage,int atStorageLocation,Eigen::Matrix<double,32,96>& output, int& imgId,int& classId);
void test_saveEigenImage(ObjectImages* storage,Eigen::Matrix<double,32,96>& input, int imgId,int objectId);

template<typename T>
void LearningModule<T>::trainEpoch(int epochCount,int storageSize)
{
    
    
    //    test_eigen_opencv(m_storage);
    int my_id = -1;
    int classId = -1;
    //m_conf_conv
    std::cout<<"\n Conf:  = "<< m_conf_conv <<"\n";
    for (int epoch = 0; epoch < epochCount; ++epoch) {
        for (int i = 0; i < storageSize; ++i) {
            if(test_loadEigenImageAt(m_storage,i, m_currentImage, my_id, classId)) {
                const int theId = (classId < 0) ? my_id : classId;
                int resultId = convTrainBackprop(theId);
                std::cout<<"\n My id = "<<my_id<<" theId: "<<theId<<" classId: "<<resultId<<"\n";
            }
            
        }
    }
    

}


/**
Trains on the last network, this will use the storage and get the last image, try to identify it and return
it will save a image to be displayed
*/
template<typename T>
IdentificationResult_t LearningModule<T>::checkLastImg()
{
    //    test_eigen_opencv(m_storage);
    int my_id = -1;
    test_loadEigenImage(m_storage, m_currentImage, my_id);
    //    std::cout<<"\n My id = "<<my_id<<"\n";
    m_loop_count++;
    IdentificationResult_t result;
    if(m_loop_count > 14) {
//        int result = convForward();
        result = convForward();
        if(result.probability > 0.2) {
            result.prevId = my_id;
        }else {
            result.prevId = -1;
        }
    }
    //    test_saveEigenImage(m_storage, m_outputImage, my_id, 5);
    layer0Out.saveEigenImg(my_id, m_outputImage);
    layer1Out.saveEigenImg(my_id, m_debugOutput.m_outputImage1);
    layer2Out.saveEigenImg(my_id, m_debugOutput.m_outputImage2);
    layer3Out.saveEigenImg(my_id, m_debugOutput.m_outputImage3);
    layer4Out.saveEigenImg(my_id, m_debugOutput.m_outputImage4);
    return result;
}

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
        IdentificationResult_t result = convForward();
        if(result.probability > 0.5) {
            result.prevId = my_id;
        }else {
            result.prevId = -1;
        }
        if(result.resultId != my_id)
        {
            
//            m_storage->getImgAt(
            //std::cout<<"\n m_loop_count = "<<m_loop_count<<" convForward res = "<<result<<" myid = "<<my_id <<" not equal\n";
            convTrainBackprop(my_id);
        }else {
            //std::cout<<"\n m_loop_count = "<<m_loop_count<<" convForward res = "<<result<<" myid = "<<my_id <<"\n";
        }
//        std::cout<<"\n m_loop_count = "<<m_loop_count<<" convForward res = "<<res<<"\n";
    }else {
//        m_outputImage = m_currentImage;
    }
    //    test_saveEigenImage(m_storage, m_outputImage, my_id, 5);
    layer0Out.saveEigenImg(my_id, m_outputImage);
    layer1Out.saveEigenImg(my_id, m_debugOutput.m_outputImage1);
    layer2Out.saveEigenImg(my_id, m_debugOutput.m_outputImage2);
    layer3Out.saveEigenImg(my_id, m_debugOutput.m_outputImage3);
    layer4Out.saveEigenImg(my_id, m_debugOutput.m_outputImage4);
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
 NextLayer maps the ConvInputMat in a way so that the ConvResult can be correcly reshaped and set
 @param HyperParam1 from layer type
 @param HyperParam2 to layer type
 */
template<typename HyperParm1,typename HyperParam2>
using NextLayer = Eigen::Map<Eigen::Matrix<double, HyperParm1::OutputRow::value, HyperParm1::OutputCol::value>,0,Eigen::Stride<HyperParam2::CalcInRow::value, 1>>;

/**
 FromLayer maps the ConvResult in a way so that the NextLayer map can be correcly reshaped and set
 @param HyperParam from layer type
 */
template<typename HyperParm>
using FromLayer = Eigen::Map<Eigen::Matrix<double,HyperParm::OutputRow::value, HyperParm::OutputCol::value>,0,Eigen::Stride<1, HyperParm::OutputCol::value>>;

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

template<typename HyperParam,int col = HyperParam::ImgDim::value * HyperParam::ImgDepth::value>
using ConvTestImage = Eigen::Matrix<double, HyperParam::ImgDim::value, col>;

/**
 ConvHyperParam creates a fake ConvHyperParam type that represents the raw image input
 @param HyperParam ConvHyperParam for the first layer
 */
template <typename Layer1>
using RawInputImage = ConvHyperParam<Layer1::ImgDim::value, Layer1::ImgDepth::value, Layer1::FilterDim::value,  Layer1::ImgDepth::value, 1,1>;


inline const bool inRange(const int a,const int value, const int b) {
    //    const int left = value - a  ;
    //    const int right = b ;
    //    return static_cast<unsigned>(left) < static_cast<unsigned>(right);
    return a <= value && value < b;
}

//template<typename Layer>
//const int multidiff(const int loc)
//{
//    return inRange(Layer::FilterDim::value,loc,Layer::ImgDim::value + 0*Layer::Padding::value - Layer::FilterDim::value ) ? 0 : ( loc > Layer::FilterDim::value ? -1 : 1);
//}

//template<typename Layer>
//const int col2imScaling(const int loc,int scalingFactor)
//{
//    if(inRange(Layer::FilterDim::value,loc,Layer::ImgDim::value + 0*Layer::Padding::value - Layer::FilterDim::value ) == false)
//    {
//        scalingFactor += loc > Layer::FilterDim::value ? -1 : 1;
//        return scalingFactor > Layer::FilterDim::value ? Layer::FilterDim::value : scalingFactor;
//    }
//    return scalingFactor;
//}

template<typename Layer>
const int col2imScalingW(const int loc,int scalingFactor)
{
    //    scalingFactor = 0;
    const int min = Layer::FilterDim::value * Layer::ImgDepth::value - Layer::ImgDepth::value;
    const int max = Layer::CalcInCol::value - Layer::FilterDim::value*Layer::ImgDepth::value + Layer::ImgDepth::value ;
    //    std::cout<<"\n min = " << min<<" loc = " << loc<<" max = " << max;
    if(inRange(min, loc, max))
    {
        return Layer::FilterDim::value;
    }
    if(loc % Layer::ImgDepth::value != 0) {
        return scalingFactor;
    }
    scalingFactor += loc > min ? -1 : 1;
    return scalingFactor;//scalingFactor > Layer::FilterDim::value ? Layer::FilterDim::value : scalingFactor;
}

template<typename Layer>
const int col2imScalingH(const int loc,int scalingFactor)
{
    if(inRange(Layer::FilterDim::value,loc,Layer::CalcInRow::value + 0*Layer::Padding::value - Layer::FilterDim::value +1 ))
    {
        return Layer::FilterDim::value;
    }
    scalingFactor += loc > Layer::FilterDim::value ? -1 : 1;
    return scalingFactor > Layer::FilterDim::value ? Layer::FilterDim::value : scalingFactor;
}

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
 col2im transform a AxB matrix into a  MxNxD matrix ,
 where every col(B) represent a output entry with the input field A transformed to be the data to dot with the filters
 
 @param HyperParam
 @param input ConvLayerDyn sized matrix
 @param output ConvInputMat sized matrix
 */
template<typename HyperParam>
void col2im(Eigen::MatrixXd& input, Eigen::Matrix<double,HyperParam::CalcInRow::value,HyperParam::CalcInCol::value>& output )
{
    
    const int resultDim = HyperParam::CalcOutCol::value;
    const int filterDim = HyperParam::FilterDim::value;
    const int depth = HyperParam::ImgDepth::value;
    const int stride = HyperParam::Stride::value;
    const int padding = HyperParam::Padding::value;
    const int imageDim = HyperParam::ImgDim::value;
    Eigen::Matrix<double,HyperParam::CalcInRow::value,HyperParam::CalcInCol::value> outputTest;
    outputTest.setZero();
    const int numcols = (imageDim - filterDim + 2*padding)/stride + 1;///usr/local/Cellar/eigen/3.3.1/include/eigen3/Eigen/src/Core/util/Macros.h
    //    std::cout<<"\n";
    for(int i = 0; i < resultDim  ;++i)
    {
        int x = i%(numcols);
        int y = (i - x)/(numcols);
        x *=stride;
        y *=stride*depth;
        
        for (int r = 0; r < filterDim ; ++r) {

            output.block(x,y,filterDim,filterDim*depth).block(r,0,1,filterDim*depth) += input.col(i).block(r*filterDim*depth, 0, filterDim*depth, 1).transpose();

        }
        
    }
    // Becouse the windows overlapp we need to divide by the number of overlaps, this creates the propper overlapp vectors
    Eigen::Matrix<double, 1, HyperParam::CalcInCol::value> scalingW;
    Eigen::Matrix<double, 1, HyperParam::CalcInRow::value> scalingH;
    int factorW = 0;
    for (int loc = 0; loc < scalingW.size(); ++loc) {
        factorW = col2imScalingW<HyperParam>(loc,factorW);
        scalingW(0,loc) = factorW*9;
    }
    int factorH = 0;
    for (int loc = 0; loc < scalingH.size(); ++loc) {
        factorH = col2imScalingH<HyperParam>(loc,factorH);
        scalingH(0,loc) = factorH*9;
    }
    // multiply by the invers of tha number of overlapps
    output =  output.array() * (scalingH.cwiseInverse().transpose()* scalingW.cwiseInverse()).array();

}

template<typename Layer,typename Result>
void convlayer_addbias(Layer& layer,Eigen::MatrixBase<Result>& result)
{
    for (int i = 0; i < layer.getFilterCount(); ++i) {
        result.row(i) = result.row(i).array() + layer.bias(i);
    }
}

template <typename HyperParam1,typename HyperParam2>
void print_next_info() {
    std::cout<<"\nM( HyperParm1::OutputRow::value = "<< HyperParam1::OutputRow::value <<" , HyperParm1::OutputCol::value = " << HyperParam1::OutputCol::value <<"),Stride( HyperParam2::CalcInRow::value = "<<HyperParam2::CalcInRow::value<<", 1)"<<" \n";
}

template <typename HyperParm1>
void print_from_info() {
    std::cout<<"M( HyperParm1::OutputRow::value = "<< HyperParm1::OutputRow::value <<" , HyperParm::OutputCol::value = " << HyperParm1::OutputCol::value <<"),Stride(1, HyperParam2::CalcInRow::value = "<<HyperParm1::OutputCol::value<<")"<<" \n";
}
template<typename HyperParm>
constexpr int outputDepthStride()
{
    return HyperParm::CalcOutCol::value;
}
//Eigen::Matrix<double,HyperParam::FilterCount::value,HyperParam::CalcOutCol::value>;

// HyperParm::OutputCol::value
// Eigen::Map<double, HyperParm1::OutputRow::value, HyperParm1::OutputCol::value>,0,Eigen::Stride<1, 25>>(output.data() )


template <typename ...T,typename InputImg,typename Result,typename Debug = ConvNet<T...>>
void conv_forward_pass(ConvNet<T...>& convNet, Eigen::MatrixBase<InputImg>& inputImg, Eigen::MatrixBase<Result>& prop_score,bool debugMode = false,DebugLayerOutputImage<Debug>* debugOutImage = nullptr) {
    using Layer1 = decltype(convNet.layer1);
    using Layer2 = decltype(convNet.layer2);
    using Layer3 = decltype(convNet.layer3);
    using Layer4 = decltype(convNet.layer4);
    
    using Layer0 = RawInputImage<Layer1>;
    //    using Layer0 = ConvHyperParam<Layer1::ImgDim::value, Layer1::ImgDepth::value, Layer1::FilterDim::value,  Layer1::ImgDepth::value, 1,1>;
    auto& layer1 = convNet.layer1;
    auto& layer2 = convNet.layer2;
    auto& layer3 = convNet.layer3;
    auto& layer4 = convNet.layer4;
    const int move1 = layer1.getPaddingOffsetX() + layer1.getPaddingOffsetY();
    const int move2 = layer2.getPaddingOffsetX() + layer2.getPaddingOffsetY();
    const int move3 = layer3.getPaddingOffsetX() + layer3.getPaddingOffsetY();
    const int move4 = layer4.getPaddingOffsetX() + layer4.getPaddingOffsetY();
    ConvInputMat<Layer1> l1in;
    ConvInputMat<Layer2> l2in;
    ConvInputMat<Layer3> l3in;
    ConvInputMat<Layer4> l4in;
    
    l1in.setZero();
    l2in.setZero();
    l3in.setZero();
    l4in.setZero();
    
    ConvLayerDyn<Layer1> convLayer1(layer1.getCalcOutRow(),layer1.getCalcOutCol());
    ConvLayerDyn<Layer2> convLayer2(layer2.getCalcOutRow(),layer2.getCalcOutCol());
    ConvLayerDyn<Layer3> convLayer3(layer3.getCalcOutRow(),layer3.getCalcOutCol());
    ConvLayerDyn<Layer4> convLayer4(layer4.getCalcOutRow(),layer4.getCalcOutCol());
    
    convLayer1.setZero();
    convLayer2.setZero();
    convLayer3.setZero();
    convLayer4.setZero();
    
    //    ConvResult<Layer4> target;
    //    ConvResult<Layer4> prop_score;
    ConvResult<Layer4> dscore;
    //    target.setZero();
//    prop_score.setZero();
    NextLayer<Layer0,Layer1>(l1in.data() + move1) = inputImg;
    
    //    std::cout<<"\n --------- layer 1 ---------"<<"\n";
    
    im2Col<Layer1>(l1in,convLayer1);
    ConvResult<Layer1> result1 = layer1.filters * convLayer1;
    convlayer_addbias(layer1,result1);
    result1 = result1.cwiseMax(0);
    
    //    std::cout<<"\n --------- layer 2 ---------"<<"\n";
    NextLayer<Layer1,Layer2>(l2in.data() + move2) = FromLayer<Layer1>(result1.data());
    im2Col<Layer2>(l2in,convLayer2);
    ConvResult<Layer2> result2 = layer2.filters * convLayer2;
    convlayer_addbias(layer2,result2);
    result2 = result2.cwiseMax(0);
    
    //    std::cout<<"\n --------- layer 3 ---------"<<"\n";
    
    NextLayer<Layer2,Layer3>(l3in.data() + move3) = FromLayer<Layer2>(result2.data());
    im2Col<Layer3>(l3in,convLayer3);
    ConvResult<Layer3> result3 = layer3.filters * convLayer3;
    convlayer_addbias(layer3,result3);
    result3 = result3.cwiseMax(0);
    
    //    std::cout<<"\n --------- layer 4 ---------"<<"\n";
    NextLayer<Layer3,Layer4>(l4in.data() + move4) = FromLayer<Layer3>(result3.data());
    im2Col<Layer4>(l4in,convLayer4);
    ConvResult<Layer4> result4 = layer4.filters * convLayer4;
    convlayer_addbias(layer4,result4);
    //        result4 = result4.cwiseMax(0);
    
    classfier::probability_score_v2(result4, prop_score);
    
    if(debugMode)
    {
        
        for (int i = 0; i < layer1.getFilterCount() ; ++i) {
            SideBySide<Layer1>(debugOutImage->m_outputImage1.data() + i* outputDepthStride<Layer1>() )
            = FromResult<Layer1>(result1.data() + i);
            //        = Eigen::Map<Eigen::Matrix<double, 32, 32>,0,Eigen::Stride<96,3>>(result2.data() + i);
        }
        for (int i = 0; i < layer2.getFilterCount() ; ++i) {
            SideBySide<Layer2>(debugOutImage->m_outputImage2.data() + i* outputDepthStride<Layer2>() )
            = FromResult<Layer2>(result2.data() + i);
            //        = Eigen::Map<Eigen::Matrix<double, 32, 32>,0,Eigen::Stride<96,3>>(result2.data() + i);
        }
        for (int i = 0; i < layer3.getFilterCount() ; ++i) {
            SideBySide<Layer3>(debugOutImage->m_outputImage3.data() + i* outputDepthStride<Layer3>() )
            = FromResult<Layer3>(result3.data() + i);
            //        = Eigen::Map<Eigen::Matrix<double, 32, 32>,0,Eigen::Stride<96,3>>(result2.data() + i);
        }
        for (int i = 0; i < layer4.getFilterCount() ; ++i) {
            SideBySide<Layer4>(debugOutImage->m_outputImage4.data() + i* outputDepthStride<Layer4>() )
            = FromResult<Layer4>(result4.data() + i);
            //        = Eigen::Map<Eigen::Matrix<double, 32, 32>,0,Eigen::Stride<96,3>>(result2.data() + i);
        }
    }
    
}

template <typename ...T,typename InputImg,typename Result>
void conv_train_fwd_bwd_pass(ConvNet<T...>& convNet, Eigen::MatrixBase<InputImg>& inputImg, Eigen::MatrixBase<Result>& target, classfier::training_conf& conf) {
    
//        classfier::training_conf conf{1,0.08,20,0.01,0.001, 1.00,1.0};
    using Layer1 = decltype(convNet.layer1);
    using Layer2 = decltype(convNet.layer2);
    using Layer3 = decltype(convNet.layer3);
    using Layer4 = decltype(convNet.layer4);
    
    using Layer0 = RawInputImage<Layer1>;
    //    using Layer0 = ConvHyperParam<Layer1::ImgDim::value, Layer1::ImgDepth::value, Layer1::FilterDim::value,  Layer1::ImgDepth::value, 1,1>;
    auto& layer1 = convNet.layer1;
    auto& layer2 = convNet.layer2;
    auto& layer3 = convNet.layer3;
    auto& layer4 = convNet.layer4;
    const int move1 = layer1.getPaddingOffsetX() + layer1.getPaddingOffsetY();
    const int move2 = layer2.getPaddingOffsetX() + layer2.getPaddingOffsetY();
    const int move3 = layer3.getPaddingOffsetX() + layer3.getPaddingOffsetY();
    const int move4 = layer4.getPaddingOffsetX() + layer4.getPaddingOffsetY();
    ConvInputMat<Layer1> l1in;
    ConvInputMat<Layer2> l2in;
    ConvInputMat<Layer3> l3in;
    ConvInputMat<Layer4> l4in;
    
    l1in.setZero();
    l2in.setZero();
    l3in.setZero();
    l4in.setZero();
    
    ConvLayerDyn<Layer1> convLayer1(layer1.getCalcOutRow(),layer1.getCalcOutCol());
    ConvLayerDyn<Layer2> convLayer2(layer2.getCalcOutRow(),layer2.getCalcOutCol());
    ConvLayerDyn<Layer3> convLayer3(layer3.getCalcOutRow(),layer3.getCalcOutCol());
    ConvLayerDyn<Layer4> convLayer4(layer4.getCalcOutRow(),layer4.getCalcOutCol());
    
    convLayer1.setZero();
    convLayer2.setZero();
    convLayer3.setZero();
    convLayer4.setZero();
    
    
    ConvResult<Layer4> prop_score;
    ConvResult<Layer4> dscore;
    prop_score.setZero();
    dscore.setZero();
    //    std::cout<<"\n dHidden3 ";print_size(dHidden3); //std::cout<<dHidden4 <<"\n";
//    std::cout<<"\n convLayer3 ";print_size(convLayer3);
//    std::cout<<"\n l3in ";print_size(l3in);
    double reg = conf.reg;// 0.001;
    
    // load image
    
//    double loss = 0.0;
//    double loss_old = 0.0;
//    // test
//    const int l1Row = layer1.getImageDim();
//    const int l1Col = layer1.getImageDim()*layer1.getImgDepth();
//    const int l1x = layer1.getPadding() * layer1.getImgDepth();
//    const int l1y = layer1.getPadding();
    
//    NextLayer<Layer0,Layer1>(l1in.data() + move1) = inputImg;
    //for (int i = 0; i <conf.epoch_count; ++i) {
        
        NextLayer<Layer0,Layer1>(l1in.data() + move1) = inputImg;
//        l1in.block(l1y,l1x,l1Row,l1Col) = inputImg;
        //    std::cout<<"\n --------- layer 1 ---------"<<"\n";
        
        im2Col<Layer1>(l1in,convLayer1);
        ConvResult<Layer1> result1 = layer1.filters * convLayer1;
        convlayer_addbias(layer1,result1);
        result1 = result1.cwiseMax(0);
        
        //    std::cout<<"\n --------- layer 2 ---------"<<"\n";
        NextLayer<Layer1,Layer2>(l2in.data() + move2) = FromLayer<Layer1>(result1.data());
        im2Col<Layer2>(l2in,convLayer2);
        ConvResult<Layer2> result2 = layer2.filters * convLayer2;
        convlayer_addbias(layer2,result2);
        result2 = result2.cwiseMax(0);
        
        //    std::cout<<"\n --------- layer 3 ---------"<<"\n";
        
        NextLayer<Layer2,Layer3>(l3in.data() + move3) = FromLayer<Layer2>(result2.data());
        im2Col<Layer3>(l3in,convLayer3);
        ConvResult<Layer3> result3 = layer3.filters * convLayer3;
        convlayer_addbias(layer3,result3);
        result3 = result3.cwiseMax(0);
        
        //    std::cout<<"\n --------- layer 4 ---------"<<"\n";
        NextLayer<Layer3,Layer4>(l4in.data() + move4) = FromLayer<Layer3>(result3.data());
        im2Col<Layer4>(l4in,convLayer4);
        ConvResult<Layer4> result4 = layer4.filters * convLayer4;
        convlayer_addbias(layer4,result4);
        //        result4 = result4.cwiseMax(0);
        
        /* ------------------------ score calculations  ----------------------- */
//        double maxVal = classfier::probability_score_v2(result4, prop_score,true);
        classfier::probability_score(result4, prop_score);
        classfier::deltascore(prop_score, target, dscore);
//        std::cout<<"dscore = "<<dscore;
        /* ------------------------ backward pass ----------------------- */
        Eigen::MatrixXd dW4 = dscore*convLayer4.transpose() + layer4.filters*reg;
        Eigen::MatrixXd dB4 = dscore.rowwise().sum() / dscore.cols();
        
        // stack allco to big ...2 step instead of 1
        Eigen::MatrixXd dHidden4tmp = (layer4.filters).transpose()*dscore;
        Eigen::MatrixXd dHidden4 =  convLayer4.binaryExpr( dHidden4tmp,classfier::BackProp_Relu<>());
        
        col2im<Layer4>(dHidden4,l4in);
        
        //    std::cout<<"\n --------- layer 4 - layer 3 back---------"<<"\n";
        FromLayer<Layer3>(result3.data()) = NextLayer<Layer3,Layer4>(l4in.data() + move4);
        Eigen::MatrixXd dW3 = result3*convLayer3.transpose() + layer3.filters*reg;
        Eigen::MatrixXd dB3 = result3.rowwise().sum() / result3.cols();
        
        Eigen::MatrixXd dHidden3tmp = (layer3.filters).transpose()*result3;
        Eigen::MatrixXd dHidden3 =  convLayer3.binaryExpr( dHidden3tmp,classfier::BackProp_Relu<>());
        
        col2im<Layer3>(dHidden3,l3in);
        
        //    std::cout<<"\n --------- layer 3 - layer 2 back---------"<<"\n";
        FromLayer<Layer2>(result2.data()) = NextLayer<Layer2,Layer3>(l3in.data() + move3);
        Eigen::MatrixXd dW2 = result2*convLayer2.transpose() + layer2.filters*reg;
        Eigen::MatrixXd dB2 = result2.rowwise().sum() / result2.cols();;
        
        Eigen::MatrixXd dHidden2tmp = (layer2.filters).transpose()*result2;
        Eigen::MatrixXd dHidden2 =  convLayer2.binaryExpr( dHidden2tmp,classfier::BackProp_Relu<>());
        //        std::cout<<"\n dHidden2 = \n"<< dHidden2 <<"\n";print_size(dHidden2);
        col2im<Layer2>(dHidden2,l2in);
        
        //    std::cout<<"\n --------- layer 2 - layer 1 back---------"<<"\n";
        FromLayer<Layer1>(result1.data()) = NextLayer<Layer1,Layer2>(l2in.data() + move2);
        Eigen::MatrixXd dW1 = result1*convLayer1.transpose() + layer1.filters*reg;
        Eigen::MatrixXd dB1 = result1.rowwise().sum() / result1.cols();;
        
        Eigen::MatrixXd dHidden1tmp = (layer1.filters).transpose()*result1;
        Eigen::MatrixXd dHidden1 =  convLayer1.binaryExpr( dHidden1tmp,classfier::BackProp_Relu<>());
        col2im<Layer1>(dHidden1,l1in);
        //backImg
        //    std::cout<<"\n --------- layer 1 back done -> backImg ---------"<<"\n";
        //    backImg = NextLayer<Layer0,Layer1>(l1in.data() + move1);// = img1;
        
        layer1.filters += -conf.step_size * dW1;
        layer2.filters += -conf.step_size * dW2;
        layer3.filters += -conf.step_size * dW3;
        layer4.filters += -conf.step_size * dW4;
        
        layer1.bias += -conf.step_size * dB1;
        layer2.bias += -conf.step_size * dB2;
        layer3.bias += -conf.step_size * dB3;
        layer4.bias += -conf.step_size * dB4;
        
//        loss = classfier::compute_cross_entropy_loss(prop_score, layer4.filters, target, conf,true);
        double loss = classfier::compute_cross_entropy_loss(prop_score, layer4.filters, target, conf);
        std::cout<<"\nloss = {"<< loss<<"}";
    
   // }
}

/**
 Does on forward pass with the conv net, on the m_currentImage

 @return a id number
 */
template<typename T>
IdentificationResult_t LearningModule<T>::convForward()
{
    using Layer1 = typename decltype(m_convNet.layer1)::type;
    using Layer4 = typename decltype(m_convNet.layer4)::type;//decltype(convNet.layer4);
    auto& layer1 = m_convNet.layer1;
    if(debug_show_layeroutput) {
        for (int i = 0; i < layer1.getImgDepth() ; ++i) {
            SideBySideInput<Layer1>(m_outputImage.data() + i* layer1.getImageDim()*layer1.getImageDim() )
            = FromInput<Layer1>(m_currentImage.data() + i*layer1.getImageDim());
            //        = Eigen::Map<Eigen::Matrix<double, 32, 32>,0,Eigen::Stride<96,3>>(result2.data() + i);
        }
    }
    ConvResult<Layer4> prop_score;
    prop_score.setZero();
    conv_forward_pass(getConvNet(), m_currentImage, prop_score,debug_show_layeroutput,&m_debugOutput);
    Eigen::Index label1 = 0;
    Eigen::Index label2 = 0;
    prop_score.maxCoeff(&label1,&label2);
    double result_prob =  prop_score(label1,label2);
    std::cout<<"prop_score: max("<< label1 <<", "<< label2 <<") "<<" prob = "<<result_prob<<"\n";
//    std::cout<<"\n prop_score = \n"<<prop_score<<"\n";
    int res = (int)label2;
    IdentificationResult_t result(result_prob,0,res);
//    conv_forward_pass
    return result;
}

//convTrainBackprop
template<typename T>
int LearningModule<T>::convTrainBackprop(int imgId)
{
    using Layer1 = typename decltype(m_convNet.layer1)::type;
    using Layer4 = typename decltype(m_convNet.layer4)::type;
    ConvResult<Layer4> target;
    target.setZero();
    const int classId = imgId % target.size();
    target(classId) = 1;
    conv_train_fwd_bwd_pass(getConvNet(), m_currentImage, target , m_conf_conv);
    return classId;
}
template<typename T>
int LearningModule<T>::tmp_func_remove_convforward()
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
//    std::cout<<"\n outputDepthStride<Layer2>() = "<< outputDepthStride<Layer2>() <<" layer4 = " << outputDepthStride<Layer4>() <<"\n";

    
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
            SideBySide<Layer1>(m_debugOutput.m_outputImage1.data() + i* outputDepthStride<Layer1>() )
            = FromResult<Layer1>(result1.data() + i);
            //        = Eigen::Map<Eigen::Matrix<double, 32, 32>,0,Eigen::Stride<96,3>>(result2.data() + i);
        }
        for (int i = 0; i < layer2.getFilterCount() ; ++i) {
            SideBySide<Layer2>(m_debugOutput.m_outputImage2.data() + i* outputDepthStride<Layer2>() )
            = FromResult<Layer2>(result2.data() + i);
            //        = Eigen::Map<Eigen::Matrix<double, 32, 32>,0,Eigen::Stride<96,3>>(result2.data() + i);
        }
        for (int i = 0; i < layer3.getFilterCount() ; ++i) {
            SideBySide<Layer3>(m_debugOutput.m_outputImage3.data() + i* outputDepthStride<Layer3>() )
            = FromResult<Layer3>(result3.data() + i);
            //        = Eigen::Map<Eigen::Matrix<double, 32, 32>,0,Eigen::Stride<96,3>>(result2.data() + i);
        }
        for (int i = 0; i < layer4.getFilterCount() ; ++i) {
            SideBySide<Layer4>(m_debugOutput.m_outputImage4.data() + i* outputDepthStride<Layer4>() )
            = FromResult<Layer4>(result4.data() + i);
            //        = Eigen::Map<Eigen::Matrix<double, 32, 32>,0,Eigen::Stride<96,3>>(result2.data() + i);
        }
    }

    
//    Eigen::Map<Eigen::Matrix<double,32 , 96>,0,Eigen::Stride<1, 1>>(m_outputImage.data())
//    = Eigen::Map<Eigen::Matrix<double,32 , 96>,0,Eigen::Stride<1, 1024>>(result2.data());//FromLayer<Layer2>(result2.data());
//    m_outputImage = output;
    return -1;
}

/*
cv::Mat eigenTest2 = cv::Mat::zeros(T1::OutputRow::value,T1::OutputCol::value,CV_64FC1);
bool saveEigenImg(int imgId,Eigen::Matrix<double,T1::OutputRow::value,T1::OutputCol::value>& input)
{
    Eigen::Map<Eigen::Matrix<double,T1::OutputRow::value,T1::OutputCol::value,Eigen::RowMajor>> b((double*)eigenTest2.data + 0);
    const double min = input.minCoeff();
    const double max = input.maxCoeff();
    b = ((input.array() - min)/(max - min)).matrix(); // *(double)1/255).matrix();
    return true;
    */
//void conv_train_fwd_bwd_pass(ConvNet<T...>& convNet, Eigen::MatrixBase<InputImg>& inputImg, Eigen::MatrixBase<Result>& target, classfier::training_conf& conf)
template <typename ...T, typename Layer = ConvNet<T...> >
void show_filters(ConvNet<T...>& convNet, cv::Mat& output,cv::Rect& bound)
{
    cv::Mat eigenTest2 = cv::Mat::zeros(Layer::FilterDim::value, Layer::Filter2dWidth::value,CV_64FC1);
    Eigen::Map<Eigen::Matrix<double, Layer::FilterDim::value, Layer::Filter2dWidth::value,Eigen::RowMajor>> b((double*)eigenTest2.data + 0);
    Eigen::Map<Eigen::Matrix<double, 1, Layer::FilterLength::value>> filter((double*)eigenTest2.data + 0);
  //  const double min = input.minCoeff();
  //  const double max = input.maxCoeff();
   // b = ((input.array() - min)/(max - min)).matrix();
    
//    cv::cvtColor(lastSample->frame, lastSample->grayScale , cv::COLOR_BGR2GRAY);
    // cv::COLOR_GRAY2BGR
    
//    cv::Mat::convertTo(<#OutputArray m#>, <#int rtype#>)
//
//    if(output.empty() || eigenTest2.empty()){
//        std::cout<<"\ngetLastImg empty\n";
//        return;
//    }
//    //            std::cout<<"\n bound = "<< bound<<"\n";
//    //            cv::Mat dst = output(bound);
    eigenTest2.copyTo(output(bound));
    
    //            eigenTest2.assignTo(dst);
}

#endif /* conv_net_hpp */
