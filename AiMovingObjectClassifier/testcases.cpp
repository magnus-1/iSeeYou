//
//  testcases.cpp
//  AiMovingObjectClassifier
//
//  Created by o_0 on 2017-01-19.
//  Copyright © 2017 Magnus. All rights reserved.
//

#include "testcases.hpp"
#include "classifiers.hpp"
#include "conv_net.hpp"
#include <iomanip>
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
    
    classfier::training_conf conf{300,0.08,6,0.7,0.001};
    
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

template <typename ...T,typename InputImg>
void display_conv_thingy(ConvNet<T...>& convNet, Eigen::MatrixBase<InputImg>& inputImg,int imgId,const classfier::training_conf& conf) {
    ConvResult<typename ConvNet<T...>::Layer4> target;
    ConvResult<typename ConvNet<T...>::Layer4> prop_score;
    prop_score.setZero();
    Eigen::Index label = 0;
    
    target.setZero();
    int tar = imgId % 3;
    target(tar) = 1;
    
    conv_forward_pass(convNet, inputImg, prop_score,false);
    prop_score.maxCoeff(&label);
    double loss = classfier::compute_cross_entropy_loss(prop_score, convNet.layer4.filters, target, conf);
    std::cout<<"\n Img"<<imgId+1<<" has label = "<<label + 1 ;
    std::cout<<" loss = {"<< loss<<"} ";
    std::cout<<" prop_score_ = "<<prop_score ;
    
//    std::cout<<"\n";
}

void test_eigen_conv_epoch()
{
    //    classfier::training_conf conf{100,0.08,20,0.1,0.001, 1.00,1.0};
//    classfier::training_conf  conf{310,0.08,20,0.001,0.001, 1.00,1.0};
    classfier::training_conf  conf{9,0.08,20,0.1,0.001, 1.00,1.0};
    //    ConvNet<
    //    ConvHyperParam<5, 3, 3, 10, 1,1>,
    //    ConvHyperParam<5, 10, 3, 5, 1,1>,
    //    ConvHyperParam<5, 5, 3, 5, 1,1>,
    //    ConvHyperParam<5, 5, 3, 1, 1,1>
    //    > convNet;
    // use one convNet module to feed into the next convNet modoule
    ConvNet<
    ConvHyperParam<32, 3, 3, 10, 1,1>,
    ConvHyperParam<32, 10, 3, 5, 1,1>,
    ConvHyperParam<32, 5, 5, 5, 1,0>,
    ConvHyperParam<28, 5, 12, 1, 6,2>
    > convNet;
    
    using Layer1 = decltype(convNet.layer1);
    using Layer2 = decltype(convNet.layer2);
    using Layer3 = decltype(convNet.layer3);
    using Layer4 = decltype(convNet.layer4);
    //    using Layer0 = ConvHyperParam<5, 3, 3, 3, 1,1>;
    using Layer0 = RawInputImage<Layer1>;
    
    auto& layer1 = convNet.layer1;
    auto& layer2 = convNet.layer2;
    auto& layer3 = convNet.layer3;
    auto& layer4 = convNet.layer4;
    
    layer1.filters.setRandom();
    layer2.filters.setRandom();
    layer3.filters.setRandom();
    layer4.filters.setRandom();
    layer1.bias.setZero();
    layer2.bias.setZero();
    layer3.bias.setZero();
    layer4.bias.setZero();
    
    
    // random images to train agenst
    ConvTestImage<Layer0> img1;
    ConvTestImage<Layer0> img2;
    ConvTestImage<Layer0> img3;
    ConvTestImage<Layer0> img4;
    
    
    ConvTestImage<Layer0> backImg;
//    
//    img1.setRandom() * 256;
//    img2.setRandom() * 256;
//    img3.setRandom() * 256;
//    img4.setRandom() * 256;
    img1.setRandom();
    img2.setRandom();
    img3.setRandom();
    img4.setRandom();

    
    
    ConvResult<Layer4> target;
    ConvResult<Layer4> prop_score;
    ConvResult<Layer4> dscore;
    target.setZero();

    std::cout << std::fixed;
    std::cout << std::setprecision(3);

    
    std::cout<<"\n --------- start loop ---------"<<"\n";
    
    classfier::training_conf conf2 = conf;
    conf2.epoch_count = 1;
    for (int i = 0; i <conf.epoch_count ; ++i) {
        target.setZero();
        int tar = i % 3;
        target(tar) = 1;
        switch(tar) {
            case 0: conv_train_fwd_bwd_pass(convNet, img1, target, conf2); break;
            case 1: conv_train_fwd_bwd_pass(convNet, img2, target, conf2); break;
            case 2: conv_train_fwd_bwd_pass(convNet, img3, target, conf2); break;
        }
        
        if(i % 1 == 0) {
            std::cout<<"\n ----------- i = "<< i <<"--------------"<<"\n";
            display_conv_thingy(convNet, img1, 0, conf2);
            display_conv_thingy(convNet, img2, 1, conf2);
            display_conv_thingy(convNet, img3, 2, conf2);
            //conf2.step_size /=2.0;
        }

        
    }

    std::cout<<"\n ------------- done ------------"<<"\n";

    display_conv_thingy(convNet, img1, 0, conf2);
    display_conv_thingy(convNet, img2, 1, conf2);
    display_conv_thingy(convNet, img3, 2, conf2);
    std::cout<<"\n -------------------------"<<"\n";
    
    
}



void test_nn_training()
{
    test_nn_train_xor();
}


void test_conv_training()
{
    test_eigen_conv_epoch();
}
