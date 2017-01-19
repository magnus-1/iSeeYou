//
//  debug_helper_info.h
//  eigen_test
//
//  Created by o_0 on 2016-12-23.
//  Copyright © 2016 o_0. All rights reserved.
//

#ifndef debug_helper_info_h
#define debug_helper_info_h
#include <iostream>
struct string_view
{
    char const* data;
    std::size_t size;
};

inline std::ostream& operator<<(std::ostream& o, string_view const& s)
{
    return o.write(s.data, s.size);
}

template<class T>
constexpr string_view get_name()
{
    char const* p = __PRETTY_FUNCTION__;
    while (*p++ != '=');
    for (; *p == ' '; ++p);
    char const* p2 = p;
    int count = 1;
    for (;;++p2)
    {
        switch (*p2)
        {
            case '[':
                ++count;
                break;
            case ']':
                --count;
                if (!count)
                    return {p, std::size_t(p2 - p)};
        }
    }
    return {};
}
template<class T>
constexpr string_view get_name(T a)
{
    char const* p = __PRETTY_FUNCTION__;
    while (*p++ != '=');
    for (; *p == ' '; ++p);
    char const* p2 = p;
    int count = 1;
    for (;;++p2)
    {
        switch (*p2)
        {
            case '[':
                ++count;
                break;
            case ']':
                --count;
                if (!count)
                    return {p, std::size_t(p2 - p)};
        }
    }
    return {};
}


#endif /* debug_helper_info_h */
