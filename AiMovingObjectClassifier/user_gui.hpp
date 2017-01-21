//
//  user_gui.hpp
//  AiMovingObjectClassifier
//
//  Created by o_0 on 2017-01-11.
//  Copyright © 2017 Magnus. All rights reserved.
//

#ifndef user_gui_hpp
#define user_gui_hpp

struct display_settings {
    bool show_original = true;
    bool show_gray = false;
    bool show_diff = false;
    bool show_threshold = false;
    bool show_detection_view = false;
    bool show_detection_resize_view = false;
    bool show_blur = false;
    bool show_debug_conv_layer = false;
    double threshold_value = 18;
};

bool handleInput(int key, struct display_settings& settings);
#endif /* user_gui_hpp */
