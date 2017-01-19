//
//  utils.hpp
//  AiMovingObjectClassifier
//
//  Created by o_0 on 2017-01-19.
//  Copyright © 2017 Magnus. All rights reserved.
//

#ifndef utils_h
#define utils_h
#include <Eigen/Dense>
#include <iostream>
template <typename Derived>
void print_size(const Eigen::EigenBase<Derived>& b)
{
    std::cout << "size (rows, cols): " << b.size() << " (" << b.rows()
    << ", " << b.cols() << ")" << "\n";
}
#endif /* utils_h */
