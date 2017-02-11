//
//  classifiers.hpp
//  AiMovingObjectClassifier
//
//  Created by o_0 on 2017-01-12.
//  Copyright © 2017 Magnus. All rights reserved.
//

#ifndef classifiers_hpp
#define classifiers_hpp
#include <Eigen/Dense>
#include <iostream>
#include "utils.hpp"
#include "neural_net_common.hpp"

namespace classfier {
    // avrage cross-entropy loss
    // Hyper parm for the fc nn
    template <typename Weight, typename Bias>
    class nn_layer {
    public:
        using WeightType = Weight;
        using BiasType = Bias;
        Eigen::MatrixBase<Weight>* weight;
        Eigen::MatrixBase<Bias>* bias;
        nn_layer(Eigen::MatrixBase<Weight>* w,Eigen::MatrixBase<Bias>* b) : weight(w),bias(b) {}
    };
    
    // creates a nnlayer
    template<typename Weights, typename Bias>
    nn_layer<Weights, Bias> make_layer(Eigen::MatrixBase<Weights>& weights,Eigen::MatrixBase<Bias>& bias) {
        return nn_layer<Weights, Bias>(&weights,&bias);
    }
    
    // initilize layer to 0
    template<typename... T>
    void layer_setzero(nn_layer<T...>& layer) {
        layer.weight->setZero();
        layer.bias->setZero();
    }
    
    
    template<typename Input, typename... Layer1, typename... Layer2, typename... Layer3, typename Output>
    void neural_net_forward_pass(Eigen::MatrixBase<Input>& trainingSet,
                                 nn_layer<Layer1...>& layer1,
                                 nn_layer<Layer2...>& layer2,
                                 nn_layer<Layer3...>& layer3,
                                 Eigen::MatrixBase<Output>& scoreOut);
    template<typename Input,typename Target, typename... Layer1, typename... Layer2, typename... Layer3, typename Output>
    void neural_net_classifier(Eigen::MatrixBase<Input>& trainingSet,
                               Eigen::MatrixBase<Target>& targetSet,
                               nn_layer<Layer1...>& layer1,
                               nn_layer<Layer2...>& layer2,
                               nn_layer<Layer3...>& layer3,
                               Eigen::MatrixBase<Output>& scoreOut,
                               const struct training_conf& conf);
    
    
    

    
    /**
     Calculate dW and dB for a layer

     @param layer the layer
     @param deltaScore the backprop deltascore
     @param weights the weigths
     @param dlayer the delta for this layer
     */
    template<typename Input,typename DeltaScore, typename Weights,typename... DeltaLayer>
    void delta_weights(const Eigen::MatrixBase<Input>& layer,
                       const Eigen::MatrixBase<DeltaScore>& deltaScore,
                       const Eigen::MatrixBase<Weights>& weights,
                       nn_layer<DeltaLayer...>& dlayer,const training_conf& conf)
    {
        double reg = conf.reg;//0.001;
        //        Eigen::MatrixXd lt = layer.transpose();
        //        std::cout<<"\n layer transpose \n"<<lt<<"\n";
        //        print_size(lt);
        //        std::cout<<"\ndeltaScore dim\n" << deltaScore<<"\n";
        //        print_size(deltaScore);
        //        std::cout<<"\n weights \n" << weights<<"\n";
        //        print_size(weights);
        //        std::cout<<"\nlayer.transpose() * deltaScore \n" << layer.transpose() * deltaScore<<"\n";
        *dlayer.weight = layer.transpose() * deltaScore + weights * reg;
        *dlayer.bias = deltaScore.colwise().sum();
    }
    
    
    /**
     Update the weigths on this layer

     @param dlayer dB and dW for this layer
     @param step_size step size
     @param layer this layer that should have its weights and bias updated
     */
    template<typename... DeltaLayer, typename... Layer>
    void update_weights(const nn_layer<DeltaLayer...>& dlayer,
                        double step_size,
                        nn_layer<Layer...>& layer,const training_conf& conf)
    {
//        *layer.weight += -step_size * (*dlayer.weight);
//        *layer.bias += -step_size * (*dlayer.bias);
        *layer.weight += -conf.step_size * (*dlayer.weight);
        *layer.bias += -conf.step_size * (*dlayer.bias);
    }
    
    
    /**
     Fully connected neural net forward pass

     @param trainingSet Targetset so score can  be computed
     @param layer1 layer1
     @param layer2 layer2
     @param layer3 layer3
     @param scoreOut the result of the forward pass
     */
    template<typename Input, typename... Layer1, typename... Layer2, typename... Layer3, typename Output>
    void neural_net_forward_pass(Eigen::MatrixBase<Input>& trainingSet,
                                 nn_layer<Layer1...>& layer1,
                                 nn_layer<Layer2...>& layer2,
                                 nn_layer<Layer3...>& layer3,
                                 Eigen::MatrixBase<Output>& scoreOut)
    {
        //        // forward pass
        Eigen::MatrixXd hidden_layer1 = (trainingSet * (*layer1.weight)).rowwise() + (*layer1.bias);
        hidden_layer1 = hidden_layer1.cwiseMax(0); //relu , max(0,element)
        //    Eigen::MatrixXd hidden_layer2 = (hidden_layer1 * weight2).rowwise() + bias2;
        Eigen::MatrixXd hidden_layer2 = (hidden_layer1 * (*layer2.weight)).rowwise() + (*layer2.bias);
        hidden_layer2 = hidden_layer2.cwiseMax(0);
        Eigen::MatrixXd output_layer = (hidden_layer2 * (*layer3.weight)).rowwise() + (*layer3.bias);
        // score calculation and delta score
        probability_score(output_layer, scoreOut);
    }
    
    namespace detail
    {
        // training a fc nn one epoch
        template<typename Input,typename Target, typename... Layer1, typename... Layer2, typename... Layer3, typename... DeltaLayer1, typename... DeltaLayer2, typename... DeltaLayer3, typename Prob,typename DScore,typename Output>
        void neural_net_train(Eigen::MatrixBase<Input>& trainingSet,
                              Eigen::MatrixBase<Target>& targetSet,
                              nn_layer<Layer1...>& layer1,
                              nn_layer<Layer2...>& layer2,
                              nn_layer<Layer3...>& layer3,
                              nn_layer<DeltaLayer1...>& dlayer1,
                              nn_layer<DeltaLayer2...>& dlayer2,
                              nn_layer<DeltaLayer3...>& dlayer3,
                              Eigen::MatrixBase<Prob>& probs,
                              Eigen::MatrixBase<DScore>& dscore,
                              Eigen::MatrixBase<Output>& scoreOut,
                              const training_conf& conf)
        {
            
            /////
            
            
            ///////
            const double step_size = 0.3;
            
            Eigen::MatrixXd hidden_layer1= (trainingSet * (*layer1.weight)).rowwise() + *layer1.bias;
            
            hidden_layer1 = hidden_layer1.cwiseMax(0);
            Eigen::MatrixXd hidden_layer2 = (hidden_layer1 * (*layer2.weight)).rowwise() + (*layer2.bias);
            
            hidden_layer2 = hidden_layer2.cwiseMax(0);
            Eigen::MatrixXd output_layer = (hidden_layer2 * (*layer3.weight)).rowwise() + (*layer3.bias);
            
            probability_score(output_layer, probs);
            
            deltascore(probs, targetSet, dscore);
            
            delta_weights(hidden_layer2, dscore, *layer3.weight, dlayer3,conf);
            Eigen::MatrixXd deltaHidden2 = hidden_layer2.binaryExpr( dscore *( *layer3.weight).transpose(),BackPropOnMin<>());
            
            delta_weights(hidden_layer1, deltaHidden2, *layer2.weight, dlayer2,conf);
            Eigen::MatrixXd deltaHidden1 = hidden_layer1.binaryExpr(deltaHidden2 *( *layer2.weight).transpose(),BackPropOnMin<>());

            
            delta_weights(trainingSet, deltaHidden1, *layer1.weight, dlayer1,conf);
            
            update_weights(dlayer1, step_size, layer1, conf);
            update_weights(dlayer2, step_size, layer2, conf);
            update_weights(dlayer3, step_size, layer3, conf);
        }
    }
    
    
    /**
     neural_net_classifier

     @param trainingSet input data
     @param targetSet target data
     @param layer1 layer1
     @param layer2 layer2
     @param layer3 layer3
     @param scoreOut output
     @param conf how long it should train what the loss goals are and stop parameters
     */
    template<typename Input,typename Target, typename... Layer1, typename... Layer2, typename... Layer3, typename Output>
    void neural_net_classifier(Eigen::MatrixBase<Input>& trainingSet,
                               Eigen::MatrixBase<Target>& targetSet,
                               nn_layer<Layer1...>& layer1,
                               nn_layer<Layer2...>& layer2,
                               nn_layer<Layer3...>& layer3,
                               Eigen::MatrixBase<Output>& scoreOut,
                               const struct training_conf& conf)
    {
        using weight1_t = typename nn_layer<Layer1...>::WeightType;
        using weight2_t = typename nn_layer<Layer2...>::WeightType;
        using weight3_t = typename nn_layer<Layer3...>::WeightType;
        using bias1_t = typename nn_layer<Layer1...>::BiasType;
        using bias2_t = typename nn_layer<Layer2...>::BiasType;
        using bias3_t = typename nn_layer<Layer3...>::BiasType;
        
        Eigen::Matrix<typename weight1_t::Scalar, weight1_t::RowsAtCompileTime, weight1_t::ColsAtCompileTime> deltaWeigth1;
        Eigen::Matrix<typename weight2_t::Scalar, weight2_t::RowsAtCompileTime, weight2_t::ColsAtCompileTime> deltaWeigth2;
        Eigen::Matrix<typename weight3_t::Scalar, weight3_t::RowsAtCompileTime, weight3_t::ColsAtCompileTime> deltaWeigth3;
        Eigen::Matrix<typename bias1_t::Scalar, bias1_t::RowsAtCompileTime, bias1_t::ColsAtCompileTime> deltabias1;
        Eigen::Matrix<typename bias2_t::Scalar, bias2_t::RowsAtCompileTime, bias2_t::ColsAtCompileTime> deltabias2;
        Eigen::Matrix<typename bias3_t::Scalar, bias3_t::RowsAtCompileTime, bias3_t::ColsAtCompileTime> deltabias3;
        
        auto deltalayer1 = make_layer(deltaWeigth1, deltabias1);
        auto deltalayer2 = make_layer(deltaWeigth2, deltabias2);
        auto deltalayer3 = make_layer(deltaWeigth3, deltabias3);
        
        layer_setzero(deltalayer1);
        layer_setzero(deltalayer2);
        layer_setzero(deltalayer3);
        Eigen::Matrix<typename Input::Scalar, Input::RowsAtCompileTime, Target::ColsAtCompileTime> probs;
        Eigen::Matrix<typename Input::Scalar, Input::RowsAtCompileTime, Target::ColsAtCompileTime> deltascore;
        
        
        probs.setZero();
        deltascore.setZero();
        
        int loss_tracking = 0;
        double loss_old = 0.0;
        double loss = 0.0;
        //        std::cout<<"\nlayer 1 w = \n"<<*layer1.weight;
        //        std::cout<<"\nlayer 1 b = \n"<<*layer1.bias;
        //        std::cout<<"\nlayer 2 w = \n"<<*layer2.weight;
        //        std::cout<<"\nlayer 2 b = \n"<<*layer2.bias;
        //        std::cout<<"\nlayer 3 w = \n"<<*layer3.weight;
        //        std::cout<<"\nlayer 3 b = \n"<<*layer3.bias;
        for (int i = 0; i < conf.epoch_count; ++i) {
            probs.setZero();
            detail::neural_net_train(trainingSet, targetSet, layer1, layer2, layer3, deltalayer1, deltalayer2, deltalayer3, probs, deltascore, scoreOut,conf);
            
            if ((i + 1) % 1 == 0) {
                std::cout<<"\nprobs = {\n"<< probs<<"\n}\n";
                loss = compute_cross_entropy_loss_minibatch(probs, *layer3.weight, targetSet,conf);
                std::cout<<"loss = {"<< loss<<"}\tdiff = {"<< loss_old - loss<<"}\n";
                if (loss >= loss_old || loss < conf.acceptable_loss) {
                    loss_tracking++;
                }else {
                    loss_tracking = 0;
                }
                loss_old = loss;
                if (loss_tracking > conf.loss_growth_limit) {
                    std::cout<<"---\nTraining halted\n\tloss tracking = {"<< loss_tracking<<"}\n\tloop count = "<<i<<" \n";
                    break;
                }
            }
            
        }
        
        Eigen::Matrix<typename Input::Scalar, Output::RowsAtCompileTime, Output::ColsAtCompileTime> result;
        result.setZero();
        Eigen::MatrixXd inputset = trainingSet;
        std::cout<<"\nresult / inputset --- \n";
        print_size(result);
        print_size(inputset);
        neural_net_forward_pass(inputset, layer1, layer2, layer3, result);
        //    linear_classfier_prob(inputset, weights, bias, result);
        Eigen::Matrix<double, 1, Input::ColsAtCompileTime> in;
        
        Eigen::Matrix<double, 1,Output::ColsAtCompileTime> res = Eigen::MatrixXd::Zero(1, Output::ColsAtCompileTime);
        in << 1,0;
        neural_net_forward_pass(in, layer1, layer2, layer3, res);
        //    linear_classfier_prob(in, weights, bias, res);
        std::cout<< "Result: in = " << in <<" o = "<<res<<" t = 1 0"<<"\n-------\n";
        for (int i = 0; i < inputset.rows(); ++i) {
            
            
            std::cout<< "Result: in = " << inputset.row(i)<<" o = "<<result.row(i)<<" t = "<<targetSet.row(i)<<"\n";
        }
        std::cout<<"\nlayer 1 w = \n"<<*layer1.weight;
        std::cout<<"\nlayer 1 b = \n"<<*layer1.bias;
        std::cout<<"\nlayer 2 w = \n"<<*layer2.weight;
        std::cout<<"\nlayer 2 b = \n"<<*layer2.bias;
        std::cout<<"\nlayer 3 w = \n"<<*layer3.weight;
        std::cout<<"\nlayer 3 b = \n"<<*layer3.bias;
    }
}
#endif /* classifiers_hpp */
