//
//  conv_net.cpp
//  AiMovingObjectClassifier
//
//  Created by o_0 on 2017-01-12.
//  Copyright © 2017 Magnus. All rights reserved.
//

#include "conv_net.hpp"
#include <iostream>
#include <Eigen/Dense>
#include "imagestorage.hpp"

// this will generate a output matrix where the columns will represent
// the subsample of a FilterDim*FilterDim matrix with stride and depth
template<int ImageDim,int FilterDim,int Padding = 0,int Depth = 1,int Stride = 1,int ResultDim = ((ImageDim - FilterDim + 2*Padding)/Stride + 1) * ((ImageDim - FilterDim + 2*Padding)/Stride + 1),int ImgDim = ImageDim + 2*Padding>
void testing_im2Col_old(const Eigen::Matrix<double,ImgDim,ImgDim*Depth>& input,Eigen::Matrix<double,FilterDim*FilterDim*Depth,ResultDim>& output)
{

    const int resultDim = ResultDim;
    std::cout<<" FilterDim ==  " <<FilterDim<<" Depth ==  " <<Depth<<" out colsize = " << output.ColsAtCompileTime<<" ResultDim = " <<resultDim <<"\n";

    const int numcols = (ImageDim - FilterDim + 2*Padding)/Stride + 1;//
    for(int i = 0; i < ResultDim  ;++i)
    {
        int x = i%(numcols);
        int y = (i - x)/(numcols);
        x *=Stride;
        y *=Stride*Depth;

        for (int r = 0; r < FilterDim ; ++r) {

            output.col(i).block(r*FilterDim*Depth, 0, FilterDim*Depth, 1) = input.template block<FilterDim,FilterDim*Depth>(x,y).template block<1,FilterDim*Depth>(r,0).transpose();
            
        }
        
    }
    
}

template<typename HyperParam>
void im2Col4(const Eigen::Matrix<double,HyperParam::CalcInRow::value,HyperParam::CalcInCol::value>& input,
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
//
//template<int Dim,int InputDepth,int PaddingHight = 0,int PaddingWidth = PaddingHight,int Dim2 = Dim + 2*PaddingHight,int Length = Dim *InputDepth + 2*PaddingWidth>
//using ConvInputMat = Eigen::Matrix<double, Dim2, Length>;
//
//template<int InputDim,int InputDepth,int FilterDim,int Stride,int Padding,int DimThingy = (InputDim - FilterDim + 2*Padding)/Stride + 1,int Hight = FilterDim*FilterDim*InputDepth,int Length = DimThingy*DimThingy>
//using ConvLayer = Eigen::Matrix<double, Hight, Length>;
//
//template<int OutputDim,int FilterCount,int Length = OutputDim*OutputDim>
//using ConvResult = Eigen::Matrix<double,FilterCount,Length>;
//
//template<int OutputDim,int FilterCount,int Length = OutputDim*FilterCount>
//using ConvOutput = Eigen::Matrix<double,OutputDim,Length>;
//
//template<int Dim,int FilterCount,int InputDepth,int Length = Dim*Dim*InputDepth>
//using ConvFilter = Eigen::Matrix<double, FilterCount, Length>;
//
//template<int InputDim,int Filterdim,int Padding,int Stride,int Dim = (InputDim - Filterdim + 2*Padding)/Stride + 1>
//constexpr int OutputDim()
//{
//    return Dim;
//}
//
//template<int Dim,int Depth,int Length = Dim*Depth>
//using ConvMat = Eigen::Matrix<double, Dim, Length>;
template<int OutputDim,int FilterCount>
using GetOutputMap_old = Eigen::Map<Eigen::Matrix<double,OutputDim,OutputDim>,0,Eigen::Stride<FilterCount, FilterCount*OutputDim>>;

template<int OutputDim,int FilterCount,int Skip = OutputDim*FilterCount>
using SetOutputMap_old = Eigen::Map<Eigen::Matrix<double,OutputDim,OutputDim>,0,Eigen::Stride<1, Skip>>;


template<int ImageDim,int FilterCount = 3,int OutputCol = ImageDim*FilterCount>
void combine3_channels(Eigen::Matrix<double, ImageDim, ImageDim>& image1,
                       Eigen::Matrix<double, ImageDim, ImageDim>& image2,
                       Eigen::Matrix<double, ImageDim, ImageDim>& image3,
                       Eigen::Matrix<double, ImageDim, OutputCol>& output)
{
    SetOutputMap_old<ImageDim,FilterCount>(output.data() + 0*ImageDim) = image1.transpose();
    SetOutputMap_old<ImageDim,FilterCount>(output.data() + 1*ImageDim) = image2.transpose();
    SetOutputMap_old<ImageDim,FilterCount>(output.data() + 2*ImageDim) = image3.transpose();
}

//    Eigen::Matrix<double, 5, 5> image1;
Eigen::Matrix<double, 5, 5> image2;
Eigen::Matrix<double, 5, 5> image3;
template <int ImageDim,int ImageDepth,int FilterDim,int FilterCount, int Stride,int Lenght = ImageDim*ImageDepth,int FilterLength = FilterDim*FilterDim*ImageDepth>
void skeleton(const Eigen::Matrix<double, ImageDim, Lenght>& image,Eigen::Matrix<double, FilterCount,FilterLength>& filters,Eigen::Matrix<double, FilterCount,1>& bias)
{
    std::cout<<"\n----------begin skeleton--------------\n";
//    const int layerDim = ImageDim;
//    const int layerDepth = ImageDepth;
//    const int filterDim = FilterDim;
//    const int filterCount = FilterCount;
//    const int padding = 1;
//    const int stride = Stride;
//    
    //    Eigen::Matrix<double, in_dim, in_dim*in_depth> convLayer;
    //    ConvInputMat<in_dim,in_depth> convLayer;
    // convLayer;
//    ConvInputMat<layerDim,layerDepth,padding,padding*layerDepth> convInput;
//    convInput.setZero();
//    //    convInput.block<layerDim,layerDim>(1,1) = image;
//    const int layerCol = padding*layerDepth;
//    
//    
//    const int outputDim = OutputDim<layerDim, filterDim, padding, stride>();
//    std::cout<<"\n image = \n"<<image<<"\n";
//    convInput.block( 1, layerCol,layerDim,layerDim*layerDepth) = image;
//    
//    std::cout<<"\n convInput = \n"<<convInput<<"\n";
//    
//    ConvLayer<layerDim,layerDepth,filterDim,stride,padding> convLayer;
//    print_size(convLayer);
//    testing_im2Col_old<layerDim, filterDim,padding,layerDepth,stride>(convInput, convLayer);
//    
//    ConvResult<outputDim, filterCount> result;
//    
//    result = filters*convLayer;
//    
//    // bias
//    for (int i = 0; i < filterCount; ++i) {
//        result.row(i) = result.row(i).array() + bias(i);
//    }
//    std::cout<<"\n result = \n"<<result<<"\n";
//    
//    ConvOutput<outputDim,filterCount> output;
//    
//    
//    //    res2next<2,3>(conv,next);
//    for (int i = 0; i < filterCount; ++i) {
//        SetOutputMap<outputDim,filterCount>(output.data() + i*outputDim) = GetOutputMap<outputDim,filterCount>(result.data() + i);
//    }
//    
//    std::cout<<"\n result = \n"<<output<<"\n";
//    std::cout<<"\n----------end skeleton--------------\n";
}



template <typename ...T>
void test_ConvNetMultiLayer2(ConvNet<T...>& convNet) {
    std::cout<<"\n ConvNetMultiLayer2 = "<<"\n";
    //    std::cout<<"\n layer.filters = \n"<<layer.filters <<"\n";
    std::cout<<"\n convNet.layer1.getImageDim() = "<<convNet.layer1.getImageDim() <<"\n";
    std::cout<<" convNet.layer2.getImageDim() = "<<convNet.layer2.getImageDim() <<"\n";
    std::cout<<" convNet.layer3.getImageDim() = "<<convNet.layer3.getImageDim() <<"\n";
    std::cout<<" convNet.layer4.getImageDim() = "<<convNet.layer4.getImageDim() <<"\n";
}

void test_ConvNetMultiLayer()
{
    ConvNet<
    ConvHyperParam<1, 1, 1, 1, 1>
    ,ConvHyperParam<2, 4, 5, 2, 2>
    ,ConvHyperParam<4, 4, 4, 2, 1>
    ,ConvHyperParam<5, 4, 5, 2, 1>
    > convNet;
    
    //    convNet.layer1.getImageDim();
    std::cout<<"\n ConvNetMultiLayer = "<<"\n";
    //    std::cout<<"\n layer.filters = \n"<<layer.filters <<"\n";
    std::cout<<"\n convNet.layer1.getImageDim() = "<<convNet.layer1.getImageDim() <<"\n";
    std::cout<<" convNet.layer2.getImageDim() = "<<convNet.layer2.getImageDim() <<"\n";
    std::cout<<" convNet.layer3.getImageDim() = "<<convNet.layer3.getImageDim() <<"\n";
    std::cout<<" convNet.layer4.getImageDim() = "<<convNet.layer4.getImageDim() <<"\n";
    test_ConvNetMultiLayer2(convNet);
}


void test_loadEigenImage(ObjectImages* storage,Eigen::Matrix<double,32,96>& output, int& imgId)
{
//    int imgId = 0;
    bool res = storage->getLastImg(imgId, output);
    if(res == false){
        std::cout<<"\n test_eigen_opencv false = "<<"\n";
        return;
    }
}

void test_saveEigenImage(ObjectImages* storage,Eigen::Matrix<double,32,96>& input, int imgId,int objectId)
{
    Eigen::Matrix<double,32,96> result = (input.array()*(double)1/255).matrix();
   
    storage->saveEigenImg(imgId, result);
}


void test_eigen_opencv(ObjectImages* storage)
{
    static int a = 0;
    int imgId = 0;
    Eigen::Matrix<double,32,96> output;
    output.setZero();
    bool res = storage->getLastImg(imgId, output);
    if(res == false){
        std::cout<<"\n test_eigen_opencv false = "<<"\n";
        return;
    }
    a++;
    
//    output.block<3,1>(a%28,48).setZero();
    output.block<15,45>(a%16,27).setConstant(255);
    if(a == 32) {
        std::cout<<"\n test_eigen_opencv output = \n"<<output<<"\n";
    }
    Eigen::Matrix<double,32,96> result = (output.array()*(double)1/255).matrix();
    if(a == 32) {
        std::cout<<"\n test_eigen_opencv result = \n"<<result<<"\n";
    }
    storage->saveEigenImg(imgId, result);
    
}

