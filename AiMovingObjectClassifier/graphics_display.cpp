//
//  graphics_display.cpp
//  AiMovingObjectClassifier
//
//  Created by o_0 on 2017-01-11.
//  Copyright © 2017 Magnus. All rights reserved.
//

#include <opencv2/opencv.hpp>
#include <opencv/highgui.h>

#include "graphics_display.hpp"
void display_window(bool isActive,const std::string& name,const cv::Mat& frame)
{
    if(isActive) {
        cv::imshow(name, frame);
    }else {
        cv::destroyWindow(name);
    }
}

void fill_detection_area(cv::Mat& frame, cv::Mat& cameraFrame, const std::vector<cv::Rect>& objects)
{
    if (objects.empty() || objects.size() <= 0) {
        return;
    }
    const cv::Rect bounding = objects.front();
    //    cv::Point2i p(1,2);
    cv::Mat src = cameraFrame(bounding);
    cv::Mat dst = frame(bounding);
    src.copyTo(dst);
    
}

void fill_sample_detection_area(cv::Mat& frame, cv::Mat& cameraFrame, const std::vector<cv::Rect>& objects,const int width,const int hight)
{
    if (objects.empty() || objects.size() <= 0) {
        return;
    }
    
    cv::Size targetSize(width,hight);
    
    cv::Point target(0,0);
    cv::Rect dstRect = cv::Rect(target,targetSize);
    const int imgCount = objects.size() % 4;
    for (int i = 0; i < imgCount; ++i) {
        const cv::Rect bounding = objects[i];
        //        std::cout<<"\n bounding = "<<bounding<< " dstRect = " <<dstRect;
        cv::Mat src = cameraFrame(bounding);
        cv::Mat dst = frame(dstRect);
        cv::resize(src, dst, targetSize);
        dstRect.x +=32;
    }
    
    //    src.copyTo(dst);
    
}

void paintTrackingRects(cv::Mat& cameraframe, const std::vector<DetectedArea>& objects, int frameCount)
{
    if (objects.size() == 0) {
        return;
    }
    for(auto& r : objects) {
        int strength = 255 - (r.frameDetectionId - frameCount);
        strength = strength < 50 ? 50 : strength;
        std::string str = "id: " + std::to_string(r.regionId);
        cv::putText(cameraframe,str, r.area.tl(), 1, 2.0, cv::Scalar(255, 255, 255));
        if (r.isSuperRegion) {
            cv::rectangle(cameraframe, r.area, cv::Scalar(0, 0, strength),2);
        }else if(r.isSubRegion) {
            cv::rectangle(cameraframe, r.area, cv::Scalar(strength, strength, 0),2);
        }else if(frameCount - r.prevFrameId > 10){
            cv::rectangle(cameraframe, r.area, cv::Scalar(strength, strength, strength),2);
        }else if(r.prevFrameId == 0) {
            cv::rectangle(cameraframe, r.area, cv::Scalar(strength, 0, 0),2);
        }else  {
            cv::rectangle(cameraframe, r.area, cv::Scalar(0, strength, strength),2);
        }
    }
    
}
