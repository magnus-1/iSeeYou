//
//  user_gui.hpp
//  AiMovingObjectClassifier
//
//  Created by o_0 on 2017-01-11.
//  Copyright © 2017 Magnus. All rights reserved.
//

#ifndef user_gui_hpp
#define user_gui_hpp
#include <iostream>
struct display_settings {
    bool show_original = true;
    bool show_gray = false;
    bool show_diff = false;
    bool show_threshold = false;
    bool show_detection_view = false;
    bool show_detection_resize_view = false;
    bool show_blur = false;
    bool show_debug_conv_layer = false;
    bool training_conv_on = false;
    double threshold_value = 18;
};

bool handleInput(int key, struct display_settings& settings);

inline std::ostream & operator<<(std::ostream & str, display_settings const & v) {
    // print something from v to str, e.g: Str << v.getX();
    str <<"original frame "<< v.show_original<<"\n";
    str <<"diff frame "<< v.show_diff<<"\n";
    str <<"threshold frame "<< v.show_threshold<<"\n";
    str <<"blur pass frame "<< v.show_blur<<"\n";
    str <<"grayscale frame "<< v.show_gray<<"\n";
    str <<"object sample frame "<< v.show_detection_view<<"\n";
    str <<"object resize sample frame "<< v.show_detection_resize_view<<"\n";
    str <<"debug conv layer output "<< v.show_debug_conv_layer<<"\n";
    str <<"debug conv layer output "<< v.show_debug_conv_layer<<"\n";
    str <<"debug conv layer training_on "<< v.training_conv_on<<"\n";
    return str;
}
#endif /* user_gui_hpp */
