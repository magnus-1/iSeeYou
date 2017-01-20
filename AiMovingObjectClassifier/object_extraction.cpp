//
//  object_extraction.cpp
//  AiMovingObjectClassifier
//
//  Created by o_0 on 2017-01-11.
//  Copyright © 2017 Magnus. All rights reserved.
//

#include "object_extraction.hpp"
#include "graphics_display.hpp"


// get the larges object and returns a bounding rect around it

/**
 takes a black and white frame and finds all areas

 @param threshold the bw image
 @param maxObjects max number of objects to find
 @return a vector of areas
 */
std::vector<cv::Rect> detectMovingObjects(cv::Mat& threshold,int maxObjects) {
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    // cv::RETR_EXTERNAL only the external contours are created
    // should replace with List or RETR_FLOODFILL or ccomp
    // cv::CHAIN_APPROX_SIMPLE we only get the endpoints of the contours
    cv::findContours(threshold, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    std::vector<cv::Rect> allObjects;
    if (contours.size() > 0) {
        // we found some
        size_t smallets_obj = (contours.size() > maxObjects) ? contours.size() - maxObjects : 0;
        auto lower_bound = contours.begin() + smallets_obj;
        auto back_iter = contours.end() - 1;
        while (lower_bound != back_iter) {
            allObjects.push_back(cv::boundingRect(*back_iter));
            --back_iter;
        }
        allObjects.push_back(cv::boundingRect(*back_iter)); // the last one also
        
    }
    return allObjects;
}



/**
 Findes all areas that have had movment in between frames

 @param currentframe the current frame sample
 @param lastframe the previous frame
 @param settings contains the threshold value
 @param maxObjects maximum  number of object that should be detected
 @return a vector of all the areas that have had movment
 */
std::vector<cv::Rect> extract_moving_objects(struct frame_sample* currentframe,struct frame_sample* lastframe,struct display_settings& settings, int maxObjects)
{
    cv::Mat framdiff;
    cv::Mat threshold;
    cv::cvtColor(currentframe->frame, currentframe->grayScale, cv::COLOR_BGR2GRAY);
    
    cv::absdiff(currentframe->grayScale, lastframe->grayScale, framdiff);
    
    cv::threshold(framdiff, threshold, settings.threshold_value, 255, cv::THRESH_BINARY);
    display_window(settings.show_gray, "grey scale", currentframe->grayScale);
    display_window(settings.show_diff, "diff", framdiff);
    display_window(settings.show_threshold, "threshold", threshold);
    
    // blur so the object becomes one, and then make it solid
    cv::blur(threshold, threshold, cv::Size(15,15));
    cv::threshold(threshold, threshold, settings.threshold_value, 255, cv::THRESH_BINARY);
    display_window(settings.show_blur, "blur", threshold);
    //        cv::Rect obj = detectMovingObject(threshold);
    return detectMovingObjects(threshold, maxObjects);
}




