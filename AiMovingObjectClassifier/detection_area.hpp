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


// any area that is smaller then width and hight gets removed
void filterDetectionArea(std::vector<cv::Rect>& objects,int width, int hight);

// this class represent the actual object we are trying to classify and follow,it has the needed metadata about it like regionId and classid. and when it was detected and last updated
class DetectedArea
{
private:
    int m_classId = -1;
    double m_class_prob = -1.0;
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
    void setClassProbability(double probability) {
        m_class_prob = probability;
    }
    
    double getClassProbability() const {
        return m_class_prob;
    }
    void setNewClassId(int classId) {
        m_classId = classId;
    }
    
    int getClassId() const {
        return m_classId;
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
    
    void updateRegion(int regionId,int classId,double result_prob);
};

#endif /* detection_area_hpp */
