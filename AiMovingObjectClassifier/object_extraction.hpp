//
//  object_extraction.hpp
//  AiMovingObjectClassifier
//
//  Created by o_0 on 2017-01-11.
//  Copyright © 2017 Magnus. All rights reserved.
//

#ifndef object_extraction_hpp
#define object_extraction_hpp
#include <vector>
#include <opencv2/opencv.hpp>
#include "detection_area.hpp"
#include "user_gui.hpp"

struct frame_sample {
    cv::Mat frame;
    cv::Mat grayScale;
};
//std::vector<cv::Rect> detectMovingObjects(cv::Mat& threshold,int maxObjects);
std::vector<cv::Rect>
extract_moving_objects(struct frame_sample* currentframe,
                       struct frame_sample* lastframe,
                       struct display_settings& settings,
                       int maxObjects);

//void filterDetectionArea(std::vector<cv::Rect>& objects,int width, int hight);
#endif /* object_extraction_hpp */
