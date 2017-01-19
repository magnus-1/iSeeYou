//
//  detection_area.hpp
//  opencv_test_prog
//
//  Created by o_0 on 2017-01-03.
//  Copyright © 2017 o_0. All rights reserved.
//

#ifndef detection_area_hpp
#define detection_area_hpp
#include <opencv/cv.h>
#include <vector>



//namespace detection { namespace details {
////    struct int width, int hight {
////        int width = 0;
////        int hight = 0;
////        int width, int hight(int w,int h) : width(w), hight(h) {}
////    };
//} };
//typedef struct detection::details::int width, int hight int width, int hight;

void filterDetectionArea(std::vector<cv::Rect>& objects,int width, int hight);
class DetectedArea
{
public:
    int frameDetectionId; // when this area was detected (at what frame)
    int prevFrameId = 0;
    int regionId = 0;
    cv::Rect area;
    bool isSubRegion = false;
    bool isSuperRegion = false;
    DetectedArea(int regionId_in,int frameId, cv::Rect theArea) : regionId(regionId_in), area(theArea), frameDetectionId(frameId){}
    DetectedArea(int regionId_in,int frameId, cv::Rect theArea, bool subRegion, bool superRegion) : regionId(regionId_in), area(theArea), frameDetectionId(frameId), isSubRegion(subRegion), isSuperRegion(superRegion){}
    void updateFrameId(int frameId)
    {
        if(prevFrameId < frameDetectionId) {
            prevFrameId = frameDetectionId;
            frameDetectionId = frameId;
        }
    }
};

bool intersectingRegions(cv::Rect& area1,cv::Rect& area2);

class CameraAreaTracker
{
    const int m_width;
    const int m_hight;
    std::vector<DetectedArea> changeTracking{};
private:
    int runningId = 0;

public:
    CameraAreaTracker(int width, int hight) : m_width(width), m_hight(hight){}
    
    void updateDetectedObjects(const int frameId,std::vector<cv::Rect>& movementLocation);
public:
    void purgeDublicate();
    
    const std::vector<DetectedArea>& getTrackedAreas() const
    {
        return changeTracking;
    }
    
    void purgeStaleAreas(int ageThreshold,int currentFrameId);
};

#endif /* detection_area_hpp */
