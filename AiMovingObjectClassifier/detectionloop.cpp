//
//  detectionloop.cpp
//  AiMovingObjectClassifier
//
//  Created by o_0 on 2017-01-11.
//  Copyright © 2017 Magnus. All rights reserved.
//

#include "detectionloop.hpp"
#include "video_feed.hpp"
#include "user_gui.hpp"
#include "graphics_display.hpp"
#include "object_extraction.hpp"
#include "imagestorage.hpp"
#include "conv_net.hpp"


// this is the mainloop, and ties the diffrent components together.
// readframe -> extract_moving_objects -> areaTracker -> storage -> learningModule -> repeat
void mainLoop()
{
    const std::string name_original = "Test original window";
    const std::string name_mosaic = "Test mosaic window";
    const std::string name_eigen_Test = "Test eigen test window";
    // storage test:
//    ConvHyperParam<32, 3, 3, 3, 1,1>
    //int ImageDim,int ImageDepth,int FilterDim1,int FilterCount1, int Stride1,int Padding1 = 0,
    // sets the hyperparm for the cnn
    ConvNet<
    ConvHyperParam<32, 3, 3, 12, 1,1>
    ,ConvHyperParam<32, 12, 3, 5, 1,1>
    ,ConvHyperParam<32, 5, 3, 3, 1,1>
    ,ConvHyperParam<32, 3, 3, 4, 1,1>
    > convNet;
    // for now it randomize the weigth, add loading here
    convNet.randomizeAll();
    // creates the learning module and set the convnet
    LearningModule<decltype(convNet)> learningModule(convNet);
    
    struct display_settings settings;
    //    cv::VideoCapture camera;
#define FRAME_WIDTH 640
#define FRAME_HIGHT 480
#define FRAME_SAMPLE_WIDTH 32
#define FRAME_SAMPLE_HIGHT 32

    VideoFeed feed;
    feed.openCameraFeed(FRAME_WIDTH, FRAME_HIGHT);
    
    CameraAreaTracker areaTracker(FRAME_WIDTH, FRAME_HIGHT);
    int key = -1;
    struct frame_sample sample1;
    struct frame_sample sample2;
    
    struct frame_sample* currentSample = &sample1;
    struct frame_sample* lastSample = &sample2;
    
    bool runloop = true;
    //    camera.open(0);
    feed.readFrame(lastSample->frame);
    cv::cvtColor(lastSample->frame, lastSample->grayScale , cv::COLOR_BGR2GRAY);
    ObjectImages storage(1,100,32,32,lastSample->frame.type());
    cv::Mat outputFrame = cv::Mat::zeros(FRAME_WIDTH, FRAME_HIGHT, lastSample->frame.type());
    //    cv::Mat eigenTest = cv::Mat::zeros(32, 32, CV_32FC3);
    cv::Mat eigenTest = cv::Mat::zeros(400, 500, CV_32FC1);
//    int yOffset = 0;
    cv::Point target(0,0);
    //    cv::Size targetSize(cv::Point(0,yOffset);
    cv::Rect bound0 (target,cv::Size(convNet.layer1.getImageDim()*convNet.layer1.getImgDepth(),convNet.layer1.getImageDim()));
    target.y += convNet.layer1.getImageDim();
    cv::Rect bound1 (target,cv::Size(convNet.layer1.getOutputCol(),convNet.layer1.getOutputRow()));
    target.y += convNet.layer1.getOutputRow();
    cv::Rect bound2 (target,cv::Size(convNet.layer2.getOutputCol(),convNet.layer2.getOutputRow()));
    target.y += convNet.layer2.getOutputRow();
    cv::Rect bound3 (target,cv::Size(convNet.layer3.getOutputCol(),convNet.layer3.getOutputRow()));
    target.y += convNet.layer3.getOutputRow();
    cv::Rect bound4 (target,cv::Size(convNet.layer4.getOutputCol(),convNet.layer4.getOutputRow()));
    target.y += convNet.layer4.getOutputRow();

    cv::Mat mosaic = cv::Mat::zeros(32*10, 32*10, lastSample->frame.type());
    // connect storage
    learningModule.setStorage(&storage);
    
    learningModule.debug_show_layeroutput = settings.show_debug_conv_layer;
    
    int frameCount = 0;
    while (runloop) {
        // read from video stream
        feed.readFrame(currentSample->frame);
        
        // this will detect all moving objects and return a vector with all the areas
        auto objects = extract_moving_objects(currentSample, lastSample, settings, 6);
        // filters out all areas smaller then 50 by 50
        filterDetectionArea(objects, 50, 50);
        
        // update the areatracking with the new areas, generating regions
        areaTracker.updateDetectedObjects(frameCount, objects);
        // the output frame is the one we draw on , so let it be a copy
        currentSample->frame.copyTo(outputFrame);
        paintTrackingRects(outputFrame, areaTracker.getTrackedAreas(), frameCount);
        
        // storage converts all relevent regions and convert them into a 32 by 32 by 3 image
        storage.storeframe(currentSample->frame, areaTracker.getTrackedAreas(), frameCount, 0);
        if(storage.bufferIsFull()){
            storage.fillMosaic(mosaic, 10, 10);
        }
        
        learningModule.debug_show_layeroutput = settings.show_debug_conv_layer;
        // start the forward pass and train on it
        learningModule.trainlastImg();
        
//        storage.fillEigenTest(eigenTest);
//        learningModule.fillMat(eigenTest1,1);
//        learningModule.fillMat(eigenTest2,2);
//        learningModule.fillMat(eigenTest3,3);
        if(settings.show_debug_conv_layer) {
            learningModule.fillMat(eigenTest,bound0,0);
            learningModule.fillMat(eigenTest,bound1,1);
            learningModule.fillMat(eigenTest,bound2,2);
            learningModule.fillMat(eigenTest,bound3,3);
            learningModule.fillMat(eigenTest,bound4,4);
        }
        
        
        display_window(true, name_original, outputFrame);
        display_window(true, name_mosaic, mosaic);
        display_window(settings.show_debug_conv_layer, name_eigen_Test, eigenTest);
        
        // user controls
        key = cv::waitKey(10);
        runloop = handleInput(key, settings);
        
        // make sure that we do not have any duplicates
        areaTracker.purgeDublicate();
        struct frame_sample* tmp = currentSample;
        currentSample = lastSample;
        lastSample = tmp;
        // we do not want to track regions forever.
        if((frameCount + 1) % 20 == 0){
            areaTracker.purgeStaleAreas(28, frameCount);
        }
        frameCount++;
        
    }
}
