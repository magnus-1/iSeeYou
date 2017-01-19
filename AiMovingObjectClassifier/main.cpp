//
//  main.cpp
//  AiMovingObjectClassifier
//
//  Created by o_0 on 2017-01-11.
//  Copyright © 2017 Magnus. All rights reserved.
//

#include <iostream>
//#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
//#include <opencv2/core/eigen.hpp>
//#include <opencv/highgui.h>
#include <opencv/cv.h>
//#include "user_gui.hpp"
//#include "graphics_display.hpp"
//#include "video_feed.hpp"
#include "detectionloop.hpp"
#include "testcases.hpp"
#include <string>


int main(int argc, const char * argv[]) {
//    // insert code here...
    std::cout << "argc "<<argc <<"\n";
    std::cout << "cv version = " <<CV_VERSION<<"\n";
    if (argc > 1) {
        std::cout << " argv "<<argv[1] <<"\n";
        if(std::strcmp("-testcase1", argv[1]) == 0) {
            test_nn_training();
            return 0;
        }
    }
//    const std::string name_original = "Test original window";
//    const std::string name_gray = "Test gray window";
//    const std::string name_diff = "Test diff window";
//    const std::string name_threshold = "Test threshold window";
//    const std::string name_blur = "Test blur window";
//    const std::string name_subsample = "Test subsample window";
//    const std::string name_reszie_subsample = "Test resized subsample window";
    
    std::cout<<"\n Display windows: \n";
    std::cout<<"    o = toggle original frame\n";
    std::cout<<"    d = toggle diff frame\n";
    std::cout<<"    t = toggle threshold frame\n";
    std::cout<<"    b = toggle blur pass frame\n";
    std::cout<<"    g = toggle grayscale frame\n";
    std::cout<<"    s = toggle object sample frame\n";
    std::cout<<"    r = toggle object resize sample frame\n";
    std::cout<<"    x/esc = quit\n";
    mainLoop();
    return 0;
}
