//
//  imagestorage.cpp
//  AiMovingObjectClassifier
//
//  Created by o_0 on 2017-01-11.
//  Copyright © 2017 Magnus. All rights reserved.
//

#include "imagestorage.hpp"

// resize image
void ObjectImages::imageResize(const cv::Mat& srcframe, const cv::Rect objectLocation,cv::Mat& dstframe)
{
    cv::Size targetSize(imageDimW,imageDimH);
    
    cv::Point target(0,0);
    //        cv::Rect dstRect = cv::Rect(target,targetSize);
    
    //        std::cout<<"\n bounding = "<<bounding<< " dstRect = " <<dstRect;
    cv::Mat src = srcframe(objectLocation);
    //        cv::Mat dst = dstframe(dstRect);
    cv::resize(src, dstframe, targetSize);
}


/**
 filles the output image with all the stored  images in a grid
 
 @param output the image
 @param imageCols number of adjecent images
 @param imageRows number of images vertecly
 */
void ObjectImages::fillMosaic(cv::Mat& output,const int imageCols,const int imageRows)
{
    madeTheLoop = false;
    const int width = imageCols;
    const int higth = imageRows;
    cv::Size targetSize(imageDimW,imageDimH);
    cv::Point target(0,0);
    cv::Rect dstRect = cv::Rect(target,targetSize);
    for(int y = 0; y < higth;++y) {
        for(int x = 0; x < width;++x) {
            int idx = y*width + x;
            if(idx >= max_size) {return;}
            if(imagebuffer[idx].image.empty()) {
                dstRect.x +=imageDimW;
                continue;
            }
            cv::Mat src = imagebuffer[idx].image;
            cv::Mat dst = output(dstRect);
            src.copyTo(dst);
            dstRect.x +=imageDimW;
        }
        dstRect.x = 0;
        dstRect.y +=imageDimH;
        
    }
}

/**
 store the resized image
 
 @param frame the frame to extract image data from
 @param detectArea e current detected areas
 */
void ObjectImages::storeframe(const cv::Mat& frame,const DetectedArea& detectArea)
{
    const cv::Rect objectLocation = detectArea.area;
    currentIdx++;
    bufferCapacity = (bufferCapacity < currentIdx) ? currentIdx : bufferCapacity;
    // ring buffer, old pics gets overriden
    if (currentIdx >= max_size) {
        currentIdx = 0;
        madeTheLoop = true;
        bufferCapacity = max_size;
    }
    imageResize(frame, objectLocation, imagebuffer[currentIdx].image);
    imagebuffer[currentIdx].regionId = detectArea.regionId;
    imagebuffer[currentIdx].classId = detectArea.getClassId();
    
}


void ObjectImages::interleveRegionId()
{
    std::vector<LabelImage> tmpBuffer;
    tmpBuffer.reserve(max_size);
    for(int i = 0; i < max_size;++i)
    {
        //            imagebuffer.push_back(cv::Mat());
        tmpBuffer.push_back(imagebuffer[i]);
    }
    std::sort(std::begin(tmpBuffer), std::end(tmpBuffer), [](LabelImage& a,LabelImage& b){return a.regionId < b.regionId;});
    auto dest = std::begin(imagebuffer);
    auto dest_end = std::end(imagebuffer);
    std:size_t stepSize = imagebuffer.size() / 10;
    for (int i = 0; i < stepSize; ++i) {
        auto tmp = std::begin(tmpBuffer) + i;
        auto tmp_end = std::end(tmpBuffer);
        while (dest < dest_end && tmp < tmp_end) {
            *dest = *tmp;
            ++dest;
            tmp = tmp + stepSize;
        }
    }
    printStorageInfo();
}

/**
 stores resized image for area in the imagebuffer if it has been updated in the last frame -+ wiggle
 
 @param frame the frame to extract image data from
 @param detectAreas a vector of the current detected areas
 @param frameId the current frame id ,
 @param wiggle the range of frame id that should be trained on
 */
void ObjectImages::storeframe(const cv::Mat& frame,
                              const std::vector<DetectedArea>& detectAreas,
                              const int frameId,const int wiggle)
{
    if (detectAreas.empty()) {
        return;
    }
    for(auto& area : detectAreas)
    {
        if (area.frameDetectionId + wiggle >= frameId) {
            storeframe(frame,area);
        }
    }
    
}

/**
 Prints number of objects, current index, and prints out the images id
 */
void ObjectImages::printStorageInfo()
{
    std::cout<<"\nCapacity: " << bufferCapacity << " currentIdx: " << currentIdx <<"\n";
    int loc = 0;
    for(auto& imglabel : imagebuffer) {
        std::cout<<"\t at: " <<loc<<"\timglabel: "<<imglabel.regionId<<"" <<"\n";
        loc++;
    }
    std::cout<<"\n";
}




