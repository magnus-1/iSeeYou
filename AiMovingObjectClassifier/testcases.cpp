//
//  testcases.cpp
//  AiMovingObjectClassifier
//
//  Created by o_0 on 2017-01-19.
//  Copyright © 2017 Magnus. All rights reserved.
//

#include "testcases.hpp"
#include "classifiers.hpp"
template<typename Input,typename Target, typename Result>
void validate_output(Eigen::MatrixBase<Input>& inputset,
                           Eigen::MatrixBase<Target>& targetSet,
                           Eigen::MatrixBase<Result>& result)
{
    std::cout<<"\n validate_output"<<"\n";
    
    for (int i = 0; i < inputset.rows(); ++i) {
        
        std::cout<< "Result: in = " << inputset.row(i)<<" o = "<<result.row(i)<<" t = "<<targetSet.row(i)<<"\n";
    }
    double diff = (result - targetSet).cwiseAbs().sum();
    std::cout<<"diff "<<diff<<"\n";
    if (diff < 0.16) {
        std::cout<<"Testcase passed "<<"\n";
    }else {
        std::cout<<"Testcase failed "<<"\n";
    }
    
}

void test_nn_train_xor()
{
    const int inputCols = 4;
    const int inputRows = 2;
    const int outputCols = inputCols;
    const int outputRows = 2;
    
    const int hiddenLayer1 = 3;
    const int hiddenLayer2 = 2;
    
    classfier::training_conf conf{300,0.08,6};
    
    Eigen::Matrix<double, inputCols, inputRows> training_input1;
    Eigen::Matrix<double, outputCols, inputRows> training_target1;
    // or gate
    //    training_input << 1,1,1,0,0,1,0,0;
    //    training_target<< 1,0,1,0,1,0,0,1;
    
    training_input1 << 1,1,1,0,0,1,0,0;
    //    training_target1<< 1,0,1,0,1,0,0,1; // or
    //    training_target1<< 1,1,1,0; // or
    training_target1<< 0,1,1,0,1,0,0,1; // xor
    
    //    Eigen::Matrix<double, 2, 2> initWeigth1_lin = Eigen::MatrixXd::Random(2, 2);
    //    Eigen::Matrix<double, 1, 2> initBias1_lin = Eigen::MatrixXd::Zero(1,2);//Random(1, 2);
    
    Eigen::Matrix<double, inputRows, hiddenLayer1> initWeigth1 = Eigen::MatrixXd::Random(inputRows, hiddenLayer1);
    Eigen::Matrix<double, 1, hiddenLayer1> initBias1 = Eigen::MatrixXd::Zero(1,hiddenLayer1);//Random(1, 2);
    //    Eigen::Matrix<double, 4, 2> orScore1;
    
    //    Eigen::Matrix<double, 2, 2> initWeigth2 = Eigen::MatrixXd::Random(2, 2);
    //    Eigen::Matrix<double, 1, 2> initBias2 = Eigen::MatrixXd::Zero(1,2);//Random(1, 2);
    Eigen::Matrix<double, hiddenLayer1, hiddenLayer2> initWeigth2 = Eigen::MatrixXd::Random(hiddenLayer1, hiddenLayer2);
    Eigen::Matrix<double, 1, hiddenLayer2> initBias2 = Eigen::MatrixXd::Zero(1,hiddenLayer2);//Random(1, 2);
    
    Eigen::Matrix<double, hiddenLayer2, outputRows> initWeigth3 = Eigen::MatrixXd::Random(hiddenLayer2, outputRows);
    Eigen::Matrix<double, 1, outputRows> initBias3 = Eigen::MatrixXd::Zero(1,outputRows);//Random(1, 2);
    
    Eigen::Matrix<double, outputCols, outputRows> orScore2;
    //    initWeigth1 = initWeigth1*0.01;
    //    initWeigth2 = initWeigth2*0.01;
    //    initWeigth3 = initWeigth3*0.01;
    //        initWeigth1 = initWeigth1.cwiseAbs();
    //        initWeigth2 = initWeigth2.cwiseAbs();
    //    orScore1.setZero();
    orScore2.setZero();
    std::cout<<"\n weigth l1 = \n"<<initWeigth1<<"\n";
    std::cout<<"\n weigth l2 = \n"<<initWeigth2<<"\n";
    std::cout<<"\n weigth l2 = \n"<<initWeigth3<<"\n";
    
    //    test_lin_classfier(training_input, training_target, initWeigth, initBias, orScore);
    //    classfier::nn_layer{initWeigth,initBias;
    //    auto lin_layer1 = classfier::make_layer(initWeigth1_lin, initBias1_lin);
    auto layer1 = classfier::make_layer(initWeigth1, initBias1);
    auto layer2 = classfier::make_layer(initWeigth2, initBias2);
    auto layer3 = classfier::make_layer(initWeigth3, initBias3);
    //    classfier::linear_classfier(training_input1, training_target1, initWeigth1, initBias1, orScore1);
    
    //    classfier::linear_classfier(training_input1, training_target1, lin_layer1, orScore1);
    //    classfier::test_neural_net_classifier(training_input1, training_target1, layer1, layer2, orScore2);
    classfier::neural_net_classifier(training_input1, training_target1, layer1, layer2, layer3, orScore2 ,conf);
    
    Eigen::Matrix<double, outputCols, outputRows> result;
    result.setZero();
    classfier::neural_net_forward_pass(training_input1, layer1, layer2, layer3, result);
    validate_output(training_input1,training_target1,result);
}


void test_nn_training()
{
    test_nn_train_xor();
}
