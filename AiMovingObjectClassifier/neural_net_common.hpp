//
//  neural_net_common.hpp
//  AiMovingObjectClassifier
//
//  Created by o_0 on 2017-01-20.
//  Copyright © 2017 Magnus. All rights reserved.
//

#ifndef neural_net_common_h
#define neural_net_common_h
#include <Eigen/Dense>

/**
 Used to turn a non-template parm into a template parm
 */
template <int N>
struct NetSize{
    enum{value = N};
};

namespace classfier {
    /**
     training_conf has the parameters to be used for training
     */
    class training_conf {
    public:
//        const int epoch_count = 500;
//        const double acceptable_loss = 0.08;
//        const int loss_growth_limit = 6;
//        double reg = 0.001;
//        double step_size = 0.3;
        int epoch_count = 500;
        double acceptable_loss = 0.08;
        int loss_growth_limit = 6;
        double step_size = 0.7;
        double reg = 0.001;
        double reg_speedup = 1.0;
        double reg_slowdown = 1.0;
        double target_loss = 0.0;
        ~training_conf() = default;
        training_conf() = default;
        training_conf(training_conf const& ) = default;
        training_conf(training_conf&& ) = default;
        training_conf& operator=(training_conf const& ) = default;
        training_conf& operator=(training_conf&& ) = default;
//        /**
//         training_conf Constructor
//         
//         @param epoch_count_parm how may epcohs this should train for at max
//         @param acceptable_loss_parm if this loss score is achived the training stops, even if more epoch are left
//         @param loss_growth_limit_parm if the loss score gets wors this many times in a row the training is halted
//         
//         */
//        training_conf(const int epoch_count_parm,double acceptable_loss_parm,int loss_growth_limit_parm) : epoch_count(epoch_count_parm), acceptable_loss(acceptable_loss_parm),loss_growth_limit(loss_growth_limit_parm){}
//        
        
        /**
         training_conf Constructor , used to config the diffrent learning rates, epoh and so on.

         @param epoch_count_parm how may epcohs this should train for at max
         @param acceptable_loss_parm if this loss score is achived the training stops, even if more epoch are left
         @param loss_growth_limit_parm if the loss score gets wors this many times in a row the training is halted
         @param step_size_parm learning rate,
         @param reg_parm regulsaztion
         @param reg_speedup_parm change of reg, when score getts better
         @param reg_slowdown_parm change of reg, when score getts worse
         */
        training_conf(const int epoch_count_parm,double acceptable_loss_parm,int loss_growth_limit_parm,double step_size_parm = 0.07, double reg_parm = 0.001,double reg_speedup_parm = 1.00,
                      double reg_slowdown_parm = 1.0)
        : epoch_count(epoch_count_parm)
        , acceptable_loss(acceptable_loss_parm)
        , loss_growth_limit(loss_growth_limit_parm)
        , step_size(step_size_parm)
        , reg(reg_parm)
        , reg_speedup(reg_speedup_parm)
        , reg_slowdown(reg_slowdown_parm)
        {}
    };

    /**
     avrage cross entropy loss
     
     @param probability_score normilized propabilitys,
     @param weights the last layers weigth
     @param targetSet the desired target output
     @return avrage cross entropy loss
     */
    template<typename Input,typename Weights,typename Target>
    double compute_cross_entropy_loss_minibatch(const Eigen::MatrixBase<Input>& probability_score,
                                                const Eigen::MatrixBase<Weights>& weights,
                                                const Eigen::MatrixBase<Target>& targetSet,
                                                const training_conf& conf,
                                                const double maxVal = 0.0)
    {
        // the target set is 1 if it is correct other wise it is 0, this means only the correct propbs get ussed
        Eigen::MatrixXd correct_probs = (probability_score.cwiseProduct(targetSet)).rowwise().sum();
        //        std::cout<<"\ncorrect_prop = \n"<<correct_probs;
        // log on all the correct outputs
        Eigen::MatrixXd lg_probs = -(correct_probs.array().log());
        double reg = conf.reg;//0.001;
        double data_loss = lg_probs.sum();
        // the loss due to the weigths and regulizesion factor
        double reg_loss = 0.5*reg*( weights.array() * weights.array()).sum();
        return data_loss + reg_loss;
    }
    /**
     avrage cross entropy loss
     
     @param probability_score normilized propabilitys,
     @param weights the last layers weigth
     @param targetSet the desired target output
     @return avrage cross entropy loss
     */
    template<typename Input,typename Weights,typename Target>
    double compute_cross_entropy_loss(const Eigen::MatrixBase<Input>& probability_score,
                                      const Eigen::MatrixBase<Weights>& weights,
                                      const Eigen::MatrixBase<Target>& targetSet,
                                      const training_conf& conf,
                                      bool miniBatch = false,
                                      const double maxVal = 0.0)
    {
        if(miniBatch) {
            return compute_cross_entropy_loss_minibatch(probability_score ,weights, targetSet, conf, maxVal);
        }
        
        // the target set is 1 if it is correct other wise it is 0, this means only the correct propbs get ussed
        double correct_probs = (probability_score.cwiseProduct(targetSet)).sum();
        //        std::cout<<"\ncorrect_prop = \n"<<correct_probs;
        // log on all the correct outputs
        double lg_probs = (correct_probs > 0.0) ? -std::log(correct_probs) : 99999.0;
        double reg = conf.reg;//0.001;
        double data_loss = lg_probs;//lg_probs.sum();
        // the loss due to the weigths and regulizesion factor
        double reg_loss = 0.5*reg*( weights.array() * weights.array()).sum();
        return data_loss + reg_loss;
    }
    
    template<typename Scalar = double,int N = 0>
    struct ValueInvers {
        EIGEN_EMPTY_STRUCT_CTOR(ValueInvers)
        using result_type = Scalar;
        Scalar operator()(const Scalar& a) const
        {
            return (a > 0.0) ? 1.0/a : 0;
        }
    };
    
    
    
    
    /**
     normilized probability score,
     
     @param layerResult the network unnormilized output
     @param scoreOut normilized probability score
     @param miniBatch if the layerResult is minibatch and every row is a diffrent result
     */
    template<typename LayerResult, typename Prob>
    double probability_score_v2(const Eigen::MatrixBase<LayerResult>& layerResult, Eigen::MatrixBase<Prob>& scoreOut,bool miniBatch = false)
    {
        
        double maxVal = layerResult.maxCoeff();
        Eigen::MatrixXd exp_score = (layerResult.array() - maxVal).exp().matrix();
        if(miniBatch) {
            scoreOut = ((exp_score.array()).colwise() / (exp_score.rowwise().sum()).array()).matrix();
        }else {
            double sum = 1.0/exp_score.sum();
            //std::cout<<"\n exp_score  sum = { "<< sum <<"} ";
            //print_size(exp_score);
            scoreOut =  exp_score * sum;
        }
        return maxVal;
    }
    
    /**
     normilized probability score,
     
     @param layerResult the network unnormilized output
     @param scoreOut normilized probability score
     */
    template<typename LayerResult, typename Prob>
    void probability_score(const Eigen::MatrixBase<LayerResult>& layerResult, Eigen::MatrixBase<Prob>& scoreOut)
    {
//        Eigen::MatrixXd exp_score = layerResult.array().exp().matrix();
//        
//        scoreOut = ((exp_score.array()).colwise() / (exp_score.rowwise().sum()).array()).matrix();
        
        probability_score_v2(layerResult,scoreOut,true);
    }
   
    
    /**
     Delta score for the output
     
     @param prob normilized probability score
     @param targetSet desired output
     @param deltaScore the diffrents between result and target
     */
    template<typename ProbScore,typename Target, typename Output>
    void deltascore(const Eigen::MatrixBase<ProbScore>& prob,const Eigen::MatrixBase<Target>& targetSet,Eigen::MatrixBase<Output>& deltaScore)
    {
        deltaScore = (prob.array() - targetSet.array()) / targetSet.rows();
        //        deltaScore = (targetSet.array() - prob.array()) / targetSet.rows();
    }
    
    /**
     BackPropOnMin backprop a value dependent on other value
     res = a.binaryExpr(b,MyWiseOp<double,0>());
     elementwise:
     element in res will be set to the element in b if corrosponding element a > 0  otherwise set to zero
     @param Scalar type , default set to double
     @param N the threshold default set to zero
     */
    template<typename Scalar = double,int N = 0>
    struct BackPropOnMin {
        EIGEN_EMPTY_STRUCT_CTOR(BackPropOnMin)
        using result_type = Scalar;
        Scalar operator()(const Scalar& a,const Scalar& b) const
        {
            return ( a > N) ? b : 0.0;
        }
    };
    
    template<typename Scalar = double,int N = 0>
    struct ForwardProp_Relu {
        EIGEN_EMPTY_STRUCT_CTOR(ForwardProp_Relu)
        using result_type = Scalar;
        Scalar operator()(const Scalar& a) const
        {
            return ( a > N) ? a : 0.0;
        }
    };
    
    template<typename Scalar = double,int N = 0>
    struct BackProp_Relu {
        EIGEN_EMPTY_STRUCT_CTOR(BackProp_Relu)
        using result_type = Scalar;
        Scalar operator()(const Scalar& a,const Scalar& b) const
        {
            return ( a > N) ? b : 0.0;
        }
    };
    //-1.0*(std::exp(a) -1 )*b
    
    template<typename Scalar = double,int N = 0>
    struct ForwardProp_ELU {
        EIGEN_EMPTY_STRUCT_CTOR(ForwardProp_ELU)
        using result_type = Scalar;
        Scalar operator()(const Scalar& a) const
        {
            return ( a > N) ? a : -1.0*(std::exp(a) -1 );
        }
    };
    
    template<typename Scalar = double,int N = 0>
    struct BackProp_ELU {
        EIGEN_EMPTY_STRUCT_CTOR(BackProp_ELU)
        using result_type = Scalar;
        Scalar operator()(const Scalar& a,const Scalar& b) const
        {
            return ( a > N) ? b : ForwardProp_ELU<>()(a)*b;
        }
    };
    
    template<typename Scalar = double,int N = 0>
    struct ForwardProp_Sig {
        EIGEN_EMPTY_STRUCT_CTOR(ForwardProp_Sig)
        using result_type = Scalar;
        Scalar operator()(const Scalar& a) const
        {
            return 1.0/(1+std::exp(-a));
        }
    };
    
    template<typename Scalar = double,int N = 0>
    struct BackProp_Sig {
        EIGEN_EMPTY_STRUCT_CTOR(BackProp_Sig)
        using result_type = Scalar;
        Scalar operator()(const Scalar& a,const Scalar& b) const
        {
            return a*(1-a)*b;
        }
    };

}

inline std::ostream & operator<<(std::ostream & str, classfier::training_conf const & v) {
    // print something from v to str, e.g: Str << v.getX();
    str << v.epoch_count <<", ";
    str << v.acceptable_loss <<", ";
    str << v.loss_growth_limit <<", ";
    str << v.step_size <<", ";
    str << v.reg <<", ";
    str << v.reg_speedup <<", " ;
    str << v.reg_slowdown;
    return str;
}

#endif /* neural_net_common_h */
