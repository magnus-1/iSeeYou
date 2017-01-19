//
//  video_feed.hpp
//  AiMovingObjectClassifier
//
//  Created by o_0 on 2017-01-11.
//  Copyright © 2017 Magnus. All rights reserved.
//

#ifndef video_feed_hpp
#define video_feed_hpp

#include <opencv/cv.h>
#include <opencv2/opencv.hpp>

class VideoFeed{
    cv::VideoCapture feed;
    
public:
    VideoFeed()
    {
    }

public:
    void openCameraFeed(const int width,const int hight);
    void setResolution(const int width,const int hight);
    void readFrame(cv::Mat& frame) {
        feed.read(frame);
    }
};

#endif /* video_feed_hpp */
