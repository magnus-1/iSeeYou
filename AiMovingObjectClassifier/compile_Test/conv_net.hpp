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

template <int N>
struct NetSize{
    enum{value = N};
};

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


template <typename HyperParam>
class ConvNetLayer {
public:
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
    
    
    
    constexpr int getImageDim() {return ImgDim::value;}
    constexpr int getImgDepth() {return ImgDepth::value;}
    constexpr int getFilterDim() {return FilterDim::value;}
    constexpr int getFilterCount() {return FilterCount::value;}
    constexpr int getFilterLength() {return FilterLength::value;}
    constexpr int getStride() {return Stride::value;}
    constexpr int getPadding() {return Padding::value;}
    
    constexpr int getPaddingOffsetX() {return ImgDim::value * Padding::value*ImgDepth::value;}
    constexpr int getPaddingOffsetY() {return Padding::value;}
    
    
    constexpr int getCalcInRow() {return CalcInRow::value;}
    constexpr int getCalcInCol() {return CalcInCol::value;}
    constexpr int getCalcOutRow() {return CalcOutRow::value;}
    constexpr int getCalcOutCol() {return CalcOutCol::value;}
    
    
    constexpr int getOutputRow() {return OutputRow::value;}
    constexpr int getOutputCol() {return OutputCol::value;}
    
    using type = HyperParam;
    
    Eigen::Matrix<double, FilterCount::value,FilterLength::value> filters;
    Eigen::Matrix<double, FilterCount::value,1> bias;
};

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


//template <typename Derived>
//void print_size(const Eigen::EigenBase<Derived>& b)
//{
//    std::cout << "size (rows, cols): " << b.size() << " (" << b.rows()
//    << ", " << b.cols() << ")" << "\n";
//}
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

template<int InputDim,int Filterdim,int Padding,int Stride,int Dim = (InputDim - Filterdim + 2*Padding)/Stride + 1>
constexpr int OutputDim()
{
    return Dim;
}

//template<typename HyperParam>
//void im2Col4(const Eigen::Matrix<double,HyperParam::CalcInRow::value,HyperParam::CalcInCol::value>& input,
//             Eigen::Matrix<double,HyperParam::CalcOutRow::value,HyperParam::CalcOutCol::value>& output)
//{

//template<int Dim,int InputDepth,int PaddingHight = 0,int PaddingWidth = PaddingHight,int Dim2 = Dim + 2*PaddingHight,int Length = Dim *InputDepth + 2*PaddingWidth>
template<typename HyperParam>
using ConvInputMat = Eigen::Matrix<double, HyperParam::CalcInRow::value, HyperParam::CalcInCol::value>;

template<typename HyperParam>
using ConvLayer = Eigen::Matrix<double, HyperParam::CalcOutRow::value, HyperParam::CalcOutCol::value>;
template<typename HyperParam>
using ConvLayerDyn = Eigen::MatrixXd;
//template<int OutputDim,int FilterCount,int Length = OutputDim*OutputDim>
template<typename HyperParam>
using ConvResult = Eigen::Matrix<double,HyperParam::FilterCount::value,HyperParam::CalcOutCol::value>;

template<typename HyperParam>
using ConvOutput = Eigen::Matrix<double,HyperParam::OutputRow::value,HyperParam::OutputCol::value>;

//template<int OutputDim,int FilterCount>
template<typename HyperParam>
using GetOutputMap = Eigen::Map<Eigen::Matrix<double,HyperParam::OutputRow::value,HyperParam::OutputRow::value>,0,Eigen::Stride<HyperParam::FilterCount::value, HyperParam::OutputCol::value>>;

//template<int OutputDim,int FilterCount,int Skip = OutputDim*FilterCount>
template<typename HyperParam>
using SetOutputMap = Eigen::Map<Eigen::Matrix<double,HyperParam::OutputRow::value,HyperParam::OutputRow::value>,0,Eigen::Stride<1, HyperParam::OutputCol::value>>;


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
//            std::cout<<"r = "<<r<<" block("<<r*filterDim*depth<<", 0, "<<filterDim*depth<<", 1) = "
//            <<" input.block("<<x<<", "<<y<<", "<<filterDim<<", "<<filterDim*depth
//            <<").block( "<<r<<", 0, 1, "<<filterDim*depth<<").transpose()"<<"\n";
            output.col(i).block(r*filterDim*depth, 0, filterDim*depth, 1) = input.block(x,y,filterDim,filterDim*depth).block(r,0,1,filterDim*depth).transpose();
            
        }
        
    }
}

template<typename HyperParm1,typename HyperParam2>
using NextLayer = Eigen::Map<Eigen::Matrix<double, HyperParm1::OutputRow::value, HyperParm1::OutputCol::value>,0,Eigen::Stride<HyperParam2::CalcInRow::value, 1>>;
template<typename HyperParm>
using FromLayer = Eigen::Map<Eigen::Matrix<double,HyperParm::OutputRow::value, HyperParm::OutputCol::value>,0,Eigen::Stride<1, HyperParm::OutputCol::value>>;
//returns the class
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
    return true;
}

#endif /* conv_net_hpp */
