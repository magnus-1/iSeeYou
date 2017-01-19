//
//  graphics_display.hpp
//  AiMovingObjectClassifier
//
//  Created by o_0 on 2017-01-11.
//  Copyright © 2017 Magnus. All rights reserved.
//

#ifndef graphics_display_hpp
#define graphics_display_hpp

#include <opencv/cv.h>
#include "detection_area.hpp"

void display_window(bool isActive,const std::string& name,const cv::Mat& frame);
void paintTrackingRects(cv::Mat& cameraframe, const std::vector<DetectedArea>& objects, int frameCount);
#endif /* graphics_display_hpp */
