//
//  imagestorage.hpp
//  AiMovingObjectClassifier
//
//  Created by o_0 on 2017-01-11.
//  Copyright © 2017 Magnus. All rights reserved.
//

#ifndef imagestorage_hpp
#define imagestorage_hpp

#include <Eigen/Dense>

#include <opencv2/opencv.hpp>
#include <opencv/cv.h>
#include <opencv2/core/eigen.hpp>
#include <vector>
#include "detection_area.hpp"

struct LabelImage {
    int regionId = -1;
    cv::Mat image = cv::Mat();
};

using LabelImage = struct LabelImage;

class ObjectImages
{
//    std::vector<cv::Mat> imagebuffer;
    std::vector<LabelImage> imagebuffer;
    
    const int max_size;
    const int regionId;
    int object_identifier = -1;
    int currentIdx = -1;
    const int imageDimH;
    const int imageDimW;
    bool madeTheLoop = false;
    cv::Mat eigenTest;
    cv::Mat eigenTest2;
public:
    ObjectImages(int theregionId ,const int maxSize,
                 const int storageDimH, const int storageDimW,int imageType) : regionId(theregionId), max_size(maxSize),imageDimH(storageDimH),imageDimW(storageDimW)
    {
        imagebuffer.reserve(maxSize);
        for(int i = 0; i < maxSize;++i)
        {
//            imagebuffer.push_back(cv::Mat());
            imagebuffer.push_back(LabelImage());
        }
        eigenTest = cv::Mat::zeros(32,32,CV_64FC3);
        eigenTest2 = cv::Mat::zeros(32,32,CV_64FC3);
    }
private:
    
    void imageResize(const cv::Mat& srcframe, const cv::Rect objectLocation,cv::Mat& dstframe);
public:
    //std::vector<DetectedArea>
    
    // stores the area in the imagebuffer if it has been updated in the last
    // frame -+ wiggle
    void storeframe(const cv::Mat& frame,
                    const std::vector<DetectedArea>& detectAreas,
                    const int frameId,const int wiggle);
    
    void storeframe(const cv::Mat& frame,const DetectedArea& detectArea);
    bool bufferIsFull()
    {
        return madeTheLoop;
    }
    
    void fillMosaic(cv::Mat& output,const int imageCols,const int imageRows);
    
//    LabelImage getLastImg()
//    {
//        return imagebuffer[]
//    }
    void fillEigenTest(cv::Mat& output){
        if(output.empty() || eigenTest2.empty()){
            std::cout<<"\ngetLastImg empty\n";
            return;
        }
        eigenTest2.copyTo(output);
    }
    bool saveEigenImg(int imgId,Eigen::Matrix<double,32,96>& output)
    {
                //        const int width = imageCols;
        //        const int higth = imageRows;
        if(imagebuffer.empty() || currentIdx < 0)return false;
        //cv::eigen2cv(output,eigenTest);
        
//        Eigen::Map<Eigen::Matrix<float,32,32>,0,Eigen::Stride<3,1>>((float*)eigenTest2.data + 0 ) = Eigen::Map<Eigen::Matrix<float,32,32>,0,Eigen::Stride<1, 96>>(output.data() + 0*32);
//        Eigen::Map<Eigen::Matrix<float,32,32>,0,Eigen::Stride<3,1>>((float*)eigenTest2.data + 4) = Eigen::Map<Eigen::Matrix<float,32,32>,0,Eigen::Stride<1, 96>>(output.data() + 1*32);
//        Eigen::Map<Eigen::Matrix<float,32,32>,0,Eigen::Stride<3,1>>((float*)eigenTest2.data + 8) =Eigen::Map<Eigen::Matrix<float,32,32>,0,Eigen::Stride<1, 96>>(output.data() + 2*32);
        Eigen::Map<Eigen::Matrix<double,32,96,Eigen::RowMajor>> b((double*)eigenTest2.data + 0);
        b = output;
        //        cv::cv2eigen(
//        cv::Size targetSize(imageDimW,imageDimH);
//        cv::Point target(0,0);
//        cv::Rect dstRect = cv::Rect(target,targetSize);
//        
//        imgId = imagebuffer[currentIdx].regionId;
//        //        Eigen::Map<Eigen::Matrix<double,32,96>>( output.data()) =( imagebuffer[currentIdx].image.data() );
//        cv::cv2eigen(imagebuffer[currentIdx].image,output);

        return false;
    }
    
    
    
    bool getLastImg(int& imgId,Eigen::Matrix<double,32,96>& output2)
    {

        if(imagebuffer.empty() || currentIdx < 0)return false;
//        cv::cv2eigen(
        cv::Size targetSize(32,32);
        cv::Point target(0,0);
        Eigen::Matrix<double,32,96,0,32,96> output;
        output.setZero();
        
        imgId = imagebuffer[currentIdx].regionId;
       
        cv::Rect dstRect = cv::Rect(target,targetSize);
        cv::Mat src = imagebuffer[currentIdx].image;
        cv::Mat img = src(dstRect);
        
        img.convertTo(eigenTest, CV_64F);
//        cv::Mat tmp;
//        std::cout<<"\n img 1 elemSize ="<<img.elemSize()<<" size = "<<img.size()<<" type() = "<<img.type()<<" depth() = "<<img.depth()<<" total() = "<<img.total()<<" \n";
//        std::cout<<"\ngetLastImg 1 elemSize ="<<eigenTest.elemSize()<<" size = "<<eigenTest.size()<<" type() = "<<eigenTest.type()<<" depth() = "<<eigenTest.depth()<<" total() = "<<eigenTest.total()<<" \n";
//        std::cout<<"\ngetLastImg 2 elemSize ="<<eigenTest2.elemSize()<<" size = "<<eigenTest2.size()<<" type() = "<<eigenTest2.type()<<" depth() = "<<eigenTest2.depth()<<" total() = "<<eigenTest2.total()<<" \n";
//        img.copyTo(tmp);//
//        std::cout<<"\ngetLastImg tmp elemSize ="<<tmp.elemSize()<<" size = "<<tmp.size()<<" tmp.type() = "<<tmp.type()<<" depth() = "<<tmp.depth()<<" total() = "<<tmp.total()<<" \n";
//    

        if(img.empty())
        {
            std::cout<<"\ngetLastImg empty\n";
            return false;
        }


        Eigen::Map<Eigen::Matrix<double,32,96,Eigen::RowMajor>> b((double*)eigenTest.data + 0);

        output2 = b;

        return true;
    }
    
    
    
};

#endif /* imagestorage_hpp */
