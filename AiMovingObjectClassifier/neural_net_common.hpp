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
        const int epoch_count = 500;
        const double acceptable_loss = 0.08;
        const int loss_growth_limit = 6;
        double reg = 0.001;
        double step_size = 0.3;
        
        /**
         training_conf Constructor
         
         @param epoch_count_parm how may epcohs this should train for at max
         @param acceptable_loss_parm if this loss score is achived the training stops, even if more epoch are left
         @param loss_growth_limit_parm if the loss score gets wors this many times in a row the training is halted
         
         */
        training_conf(const int epoch_count_parm,double acceptable_loss_parm,int loss_growth_limit_parm) : epoch_count(epoch_count_parm), acceptable_loss(acceptable_loss_parm),loss_growth_limit(loss_growth_limit_parm){}
    };

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
                                      const Eigen::MatrixBase<Target>& targetSet,const training_conf& conf)
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
     normilized probability score,
     
     @param layerResult the network unnormilized output
     @param scoreOut normilized probability score
     */
    template<typename LayerResult, typename Prob>
    void probability_score(const Eigen::MatrixBase<LayerResult>& layerResult, Eigen::MatrixBase<Prob>& scoreOut)
    {
        Eigen::MatrixXd exp_score = layerResult.array().exp().matrix();
        
        scoreOut = ((exp_score.array()).colwise() / (exp_score.rowwise().sum()).array()).matrix();
        
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

}

#endif /* neural_net_common_h */
