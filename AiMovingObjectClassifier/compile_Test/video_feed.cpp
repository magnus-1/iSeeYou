//
//  video_feed.cpp
//  AiMovingObjectClassifier
//
//  Created by o_0 on 2017-01-11.
//  Copyright © 2017 Magnus. All rights reserved.
//

#include "video_feed.hpp"


void VideoFeed::openCameraFeed(const int width,const int hight) {
    feed.open(0);
    feed.set(CV_CAP_PROP_FRAME_WIDTH, width);
    feed.set(CV_CAP_PROP_FRAME_HEIGHT, hight);
}

void VideoFeed::setResolution(const int width,const int hight) {
    feed.set(CV_CAP_PROP_FRAME_WIDTH, width);
    feed.set(CV_CAP_PROP_FRAME_HEIGHT, hight);
}

