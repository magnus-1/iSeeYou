//
//  main.cpp
//  AiMovingObjectClassifier
//
//  Created by o_0 on 2017-01-11.
//  Copyright © 2017 Magnus. All rights reserved.
//

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv/cv.h>
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
    
    std::cout<<"\n Display windows: \n";
    std::cout<<"    o = toggle original frame\n";
    std::cout<<"    d = toggle diff frame\n";
    std::cout<<"    t = toggle threshold frame\n";
    std::cout<<"    b = toggle blur pass frame\n";
    std::cout<<"    g = toggle grayscale frame\n";
    std::cout<<"    s = toggle object sample frame\n";
    std::cout<<"    r = toggle object resize sample frame\n";
    std::cout<<"    L = toggle debug conv layer output \n";
    std::cout<<"    x/esc = quit\n";
    mainLoop();
    return 0;
}
