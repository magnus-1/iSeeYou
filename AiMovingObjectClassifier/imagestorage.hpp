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

// This is the structure that holds the image to be processed
// and its region id
struct LabelImage {
    int regionId = -1;
    int classId = -1;
    cv::Mat image = cv::Mat();
};

using LabelImage = struct LabelImage;


// Object image is the storage class that holds the last N images
class ObjectImages
{
//    std::vector<cv::Mat> imagebuffer;
    // holds the resized images and its regionId
    std::vector<LabelImage> imagebuffer;
    
    const int max_size;
    const int regionId;
    int object_identifier = -1;
    int currentIdx = -1;
    const int imageDimH;
    const int imageDimW;
    bool madeTheLoop = false;
    int bufferCapacity = 0;
    cv::Mat eigenTest;
    cv::Mat eigenTest2;
public:
    
    /**
     Construct a StorageObject to hold labeld images

     @param theregionId if storage is regionId specific otherwise use -1 or 0
     @param maxSize the maximum number of images to store
     @param storageDimH the hight dimensions
     @param storageDimW the width dimensions
     @param imageType the image type they are stored as
     @return the storage obj
     */
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
//        eigenTest2 = cv::Mat::zeros(32,32,CV_64FC3);
        eigenTest2 = cv::Mat::zeros(32,96,CV_64FC1);
    }
    
    ObjectImages(ObjectImages& other) : regionId(other.regionId), max_size(other.max_size),imageDimH(other.imageDimH),imageDimW(other.imageDimW)
    {
        imagebuffer.reserve(max_size);
        for(int i = 0; i < max_size;++i)
        {
            //            imagebuffer.push_back(cv::Mat());
            imagebuffer.push_back(other.imagebuffer[i]);
        }
        eigenTest = cv::Mat::zeros(32,32,CV_64FC3);
        //        eigenTest2 = cv::Mat::zeros(32,32,CV_64FC3);
        eigenTest2 = cv::Mat::zeros(32,96,CV_64FC1);
    }
private:

    
    void imageResize(const cv::Mat& srcframe, const cv::Rect objectLocation,cv::Mat& dstframe);
public:
    
    void interleveRegionId();

    /**
     stores resized image for area in the imagebuffer if it has been updated in the last frame -+ wiggle
    
     @param frame the frame to extract image data from
     @param detectAreas a vector of the current detected areas
     @param frameId the current frame id ,
     @param wiggle the range of frame id that should be trained on
     */
    void storeframe(const cv::Mat& frame,
                    const std::vector<DetectedArea>& detectAreas,
                    const int frameId,const int wiggle);
    
    
    /**
     store the resized image

     @param frame the frame to extract image data from
     @param detectArea e current detected areas
     */
    void storeframe(const cv::Mat& frame,const DetectedArea& detectArea);
    
    
    /**
     This returns true if the buffer has completed one cycle (maxsize)

     @return true if true =)
     */
    bool bufferIsFull()
    {
        return madeTheLoop;
    }
    
    
    /**
     Prints number of objects, current index, and prints out the images id
     */
    void printStorageInfo();
    
    /**
     filles the output image with all the stored  images in a grid

     @param output the image
     @param imageCols number of adjecent images
     @param imageRows number of images vertecly
     */
    void fillMosaic(cv::Mat& output,const int imageCols,const int imageRows);
    

    
    /**
     used to debug eigen, this filles a image with data from a savedEigenImg

     @param output image
     */
    void fillEigenTest(cv::Mat& output){
        if(output.empty() || eigenTest2.empty()){
            std::cout<<"\ngetLastImg empty\n";
            return;
        }
        eigenTest2.copyTo(output);
    }
    
    

    /**
     savesa eigen matrix into a image
     
     @param imgId the image id
     @param input eigen image
     @return true if it succseids
     */
    bool saveEigenImg(int imgId,Eigen::Matrix<double,32,96>& input)
    {

        if(imagebuffer.empty() || currentIdx < 0)return false;

        Eigen::Map<Eigen::Matrix<double,32,96,Eigen::RowMajor>> b((double*)eigenTest2.data + 0);
        b = input;

        return true;
    }
    
    
    
    /**
     get last updated image

     @param imgId its ide
     @param output2 the image in eigen form
     @return true on success
     */
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

        if(img.empty())
        {
            std::cout<<"\ngetLastImg empty\n";
            return false;
        }


        Eigen::Map<Eigen::Matrix<double,32,96,Eigen::RowMajor>> b((double*)eigenTest.data + 0);

        output2 = b;

        return true;
    }
    
    int getImgAt(int imgLoc,int& imgId,int& classId,Eigen::Matrix<double,32,96>& output2)
    {
        
        if(imagebuffer.empty() || currentIdx < 0 || imgLoc >= bufferCapacity) return -1;
        //        cv::cv2eigen(
        cv::Size targetSize(32,32);
        cv::Point target(0,0);
        Eigen::Matrix<double,32,96,0,32,96> output;
        output.setZero();
        
        imgId = imagebuffer[imgLoc].regionId;
        classId = imagebuffer[imgLoc].classId;
        cv::Rect dstRect = cv::Rect(target,targetSize);
        cv::Mat src = imagebuffer[imgLoc].image;
        cv::Mat img = src(dstRect);
        
        img.convertTo(eigenTest, CV_64F);
        
        if(img.empty())
        {
            std::cout<<"\ngetLastImg empty\n";
            return false;
        }
        
        
        Eigen::Map<Eigen::Matrix<double,32,96,Eigen::RowMajor>> b((double*)eigenTest.data + 0);
        
        output2 = b;
        
        return currentIdx;
    }
    
    
    
};

#endif /* imagestorage_hpp */
