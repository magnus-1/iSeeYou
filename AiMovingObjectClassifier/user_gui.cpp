//
//  user_gui.cpp
//  AiMovingObjectClassifier
//
//  Created by o_0 on 2017-01-11.
//  Copyright © 2017 Magnus. All rights reserved.
//

#include "user_gui.hpp"
#include "keycodes.h"
#include <iostream>



bool handleInput(int key, struct display_settings& settings)
{
    switch (key) {
        case -1:
            std::cout<<".";
            break;
        case IN_KEY_PLUS:
            ++settings.threshold_value;
            std::cout<<"\nt: "<<settings.threshold_value<<"\n";
            break;
        case IN_KEY_MINUS:
            --settings.threshold_value;
            std::cout<<"\nt: "<<settings.threshold_value<<"\n";
            break;
        case IN_KEY_B:
            settings.show_blur = !settings.show_blur;
            break;
        case IN_KEY_G:
            settings.show_gray = !settings.show_gray;
            break;
        case IN_KEY_D:
            settings.show_diff = !settings.show_diff;
            break;
        case IN_KEY_O:
            settings.show_original = !settings.show_original;
            break;
        case IN_KEY_S:   // s
            settings.show_detection_view = !settings.show_detection_view;
            break;
        case IN_KEY_R:   // r
            settings.show_detection_resize_view = !settings.show_detection_resize_view;
            break;
        case IN_KEY_T:
            settings.show_threshold = !settings.show_threshold;
            break;
        case IN_KEY_X:
            return false;
        case IN_KEY_ESC:
            return false;
        default:
            std::cout<<"\nIN_KEY_"<<(char)key<<" = "<<key<<",'";
            //keycount++;
            break;
    }
    
    return true;
}
