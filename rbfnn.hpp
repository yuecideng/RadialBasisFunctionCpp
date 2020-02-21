#ifndef RBFNN_H
#define RBFNN_H

#include <opencv2/opencv.hpp>

using namespace std;

/**
 * @brief Radial Basis Function (RBF) neural network 
 *        The data calculated by the function should be normalized 
 *        within [0,1]
 *        This network is normally used in image warpping 
 */
class RBFRegression
{
private:
    typedef cv::Mat data;

    /**
     * @brief dimension of an training data, also can be seen as features number 
     */
    const int din;

    /**
     * @brief dimension of target data
     */
    const int dout;

    /**
     * @brief number of clustering center of K-means algorithm. 
     * The center is computed using the training data
     */
    const int numCenter;

    /**
     * @brief clustered centers using K-means 
     */
    data centers;

    /**
     * @brief The covariance matrixes of each cneters.
     * Also can be seen as the activation area of the rbf unit. 
     */
    vector<data> covariances;

    /**
     * @brief weights of the rbf network
     */
    data weights;

    /**
     * @brief RBF basis function
     * 
     * @param traindata: one of training data 
     * @param cnetersdata: one of clustering center
     * @param vardata: one of the variance corresponding to the center 
     * @return float
     */
    double basisFunc(data traindata, data cnetersdata, data vardata);

    /**
     * @brief compute activation of RBF net using input data
     * 
     * @param inputdata
     * @return vector<data> 
     */
    cv::Mat calcActivation(data inputdata);

public:
    /**
     * @brief method to solve the weights of rbf network 
     * if method = 0, use pseudo inverse
     * w = pinv(G) * Y, where G is the activation output, Y is the target
     * if method = 1, use LMS close-form solution
     * w = inv(G.T * G) * G.T * Y, T means the transpose of the matrix
     */
    int method;

    /**
     * @brief Construct a new RBFRegression object
     * 
     * @param din: dimension of input data  
     * @param numCenter: number of clustering canter 
     * @param dout: dimension of output data
     * @param method: method to solve the weights of rbf network
     */
    RBFRegression(const int din, const int numCenter, const int dout);
    RBFRegression(const int din, const int numCenter, const int dout, int methed);
    RBFRegression();

    /**
     * @brief train the RBF net using training data and target data
     * the input data should be normalized within [0, 1]
     * @param: traindata 
     * @param: targetdata 
     */
    void train(data &traindata, data &targetdata);

    /**
     * @brief predict the test data using trained RBF net
     * 
     * @param: testdata 
     * @param: dstdata
     */
    cv::Mat predict(data testdata);
};

#endif