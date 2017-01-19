//
//  detection_area.cpp
//  opencv_test_prog
//
//  Created by o_0 on 2017-01-03.
//  Copyright © 2017 o_0. All rights reserved.
//

#include <algorithm>
#include <math.h>
#include <vector>
#include <iostream>
#include "detection_area.hpp"


// any area that is smaller then width and hight gets removed
void filterDetectionArea(std::vector<cv::Rect>& objects, int width, int hight)
{
    objects.erase( std::remove_if(std::begin(objects),
                                  std::end(objects),
                                  [&width,&hight](cv::Rect r){
                                      return r.width < width || r.height < hight;
                                      
                                  }),std::end(objects));
    //    std::for_each(std::begin(objects), std::end(objects), [&limit])
}

bool intersectingRegions(cv::Rect& area1,cv::Rect& area2)
{
    return ( abs(area1.x - area2.x) * 2 < (area1.width + area2.width)) &&
    (abs(area1.y - area2.y) * 2 < (area1.height + area2.height));
}

// merge 2 regions, the one with the lowes id survives
void mergeRegions(DetectedArea& obj1, DetectedArea& obj2)
{
    if(obj1.regionId < obj2.regionId)
    {
        obj2.regionId = -1;
    }else {
        obj1.regionId = -1;
    }
    obj1.updateFrameId(obj2.frameDetectionId);
    obj2.updateFrameId(obj1.frameDetectionId);
    //        if(obj2.f)
}

// this structure is used to merge and check new areas against old regions
struct AreaUpdateOnOverlapp {
    cv::Rect area;
    bool didOverlapp = false;
    bool isSubRegion = false;
    bool isSuperRegion = false;
    int frameId;
    AreaUpdateOnOverlapp(int myframeId, cv::Rect myarea): area(myarea), frameId(myframeId) {}
    bool intersect(cv::Rect& target)
    {
        return ( abs(area.x - target.x) * 2 < (area.width + target.width)) &&
        (abs(area.y - target.y) * 2 < (area.height + target.height));
    }
    void operator()(DetectedArea& target) {
        if(intersect(target.area) == false)
        {
            return; // no overlapp
        }
        didOverlapp = true;
        double ratiox = (double)area.width/target.area.width;
        double ratioy = (double)area.height/target.area.height;
        //            double ratio = (double)area.area()/target.area.area();
        
        double ratio = (ratiox + ratioy)/2;
        //            std::cout<<"\nratio = "<<ratio<<" ratio x = "<<ratiox<<" ratio y = "<<ratioy;
        if(ratio < 0.7) {
            isSubRegion = true;
        }else if(4.0 < ratio) {
            isSuperRegion = true;
        }else {
            
            // update the target area
            target.updateFrameId(frameId);
            target.area = area;
            target.isSubRegion = false;
            target.isSuperRegion = false;
        }
    }
};

void CameraAreaTracker::updateDetectedObjects(const int frameId,std::vector<cv::Rect>& movementLocation)
{
    if(changeTracking.empty()) {
        for(auto& obj : movementLocation) {
            changeTracking.push_back(DetectedArea(++runningId,frameId, obj));
        }
        return;
    }
    if (movementLocation.size() < 1) {
        //            std::cout<<"\nArea count = "<< changeTracking.size() << " new obj = "<<movementLocation.size()<<"\n";
        return;
    }
    std::sort(std::begin(changeTracking),std::end(changeTracking),
              [](DetectedArea& a,DetectedArea& b){
                  return a.area.x < b.area.x;
              });
    
    // naive overlapp check prototype
    for(auto& loc : movementLocation) {
        AreaUpdateOnOverlapp result = std::for_each(std::begin(changeTracking),std::end(changeTracking), AreaUpdateOnOverlapp(frameId,loc));
        
        if (result.didOverlapp && !(result.isSuperRegion || result.isSuperRegion)) {
            // good, we have already updated it
            continue;
        }
        
        changeTracking.push_back(DetectedArea(++runningId,frameId, result.area, result.isSubRegion, result.isSuperRegion));
    }
    //        std::cout<<"\nArea count = "<< changeTracking.size() << " new obj = "<<movementLocation.size()<<"\n";
}

void CameraAreaTracker::purgeDublicate()
{
    for(int i = 0; i < changeTracking.size(); ++i )
    {
        auto& obj1 = changeTracking[i];
        if(obj1.regionId < 0){continue;}
        for(int t = i + 1; t < changeTracking.size(); ++t )
        {
            auto& obj2 = changeTracking[t];
            if(obj2.regionId < 0 ) {continue;}
            if(intersectingRegions(obj1.area,obj2.area) == false){continue;}
            mergeRegions(obj1, obj2);
            if(obj1.regionId < 0){
                break;  // this is a duplicate region
            }
            
        }
    }
    changeTracking.erase( std::remove_if(std::begin(changeTracking),
                                         std::end(changeTracking),
                                         [](DetectedArea& da){
                                             return da.regionId < 0;
                                         }),std::end(changeTracking));
}


void CameraAreaTracker::purgeStaleAreas(int ageThreshold,int currentFrameId)
{
    changeTracking.erase( std::remove_if(std::begin(changeTracking),
                                         std::end(changeTracking),
                                         [ageThreshold, currentFrameId](DetectedArea& da){
                                             return ageThreshold < currentFrameId - da.frameDetectionId;
                                             
                                         }),std::end(changeTracking));
    
}
