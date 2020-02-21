#include <math.h>

#include "rbfnn.hpp"

double RBFRegression::basisFunc(data traindata, data centersdata, data vardata)
{
    assert(traindata.cols == din);
    
    data normVarData;
    data diff;
    cv::subtract(centersdata.t(), traindata.t(), diff);
    cv::normalize(vardata, normVarData);

    double out = (diff.t() * normVarData.inv(cv::DECOMP_SVD)).dot(diff.t());

    return exp(-out);
}

cv::Mat RBFRegression::calcActivation(data inputdata) 
{
    data actOut(cv::Size(numCenter, inputdata.rows), CV_64F);

    for (size_t i = 0; i < inputdata.rows; i++)
    {
        for (size_t j = 0; j < numCenter; j++)
        {
            actOut.at<double>(i, j) = basisFunc(inputdata.row(i), centers.row(j), covariances[j]);
        } 
    }
    
    return actOut;
}

RBFRegression::RBFRegression()
    :din(), numCenter(), dout(), method() {}

RBFRegression::RBFRegression(const int din, const int numCenter, const int dout)
    : din(din), numCenter(numCenter), dout(dout), method(0) {}

RBFRegression::RBFRegression(const int din, const int numCenter, const int dout, int method)
    : din(din), numCenter(numCenter), dout(dout), method(method) {}

void RBFRegression::train(data &traindata, data &targetdata)
{
    assert(traindata.rows == targetdata.rows);
    assert(traindata.cols == targetdata.cols);
    assert(traindata.cols == din);

    const cv::TermCriteria criteria = cv::TermCriteria(cv::TermCriteria::EPS, 10, 0.01);
    data bestLabels;

    // using k-means to cluster the RBF centers
    cv::kmeans(traindata, numCenter, bestLabels, criteria, 10, cv::KMEANS_PP_CENTERS, centers); 

    // convert datatype to double
    traindata.convertTo(traindata, CV_64F);
    targetdata.convertTo(targetdata, CV_64F);
    centers.convertTo(centers, CV_64F);

    // compute covariance matrix
    for (size_t i = 0; i < numCenter; i++)
    {
        data features;

        for (size_t j = 0; j < traindata.rows; j++)
        {
            // group traindata based on centers
            if (bestLabels.at<int>(j, 0) == i)
            {
                features.push_back(traindata.row(j));
            }
        }

        data covar;
        
        cv::calcCovarMatrix(features, covar, centers.row(i), cv::COVAR_NORMAL | cv::COVAR_ROWS);
        covariances.push_back(covar);
    }

    // compute RBF activation output
    data actOut = calcActivation(traindata);
    
    // compute weights using pseudo inverse
    if (method == 0)
    {
        weights = actOut.inv(cv::DECOMP_SVD) * targetdata;
    }
    // compute weights using LMS close-form solution
    else if (method == 1)
    {
        weights = ((actOut.t() * actOut).inv() * actOut.t()) * targetdata;
    }   
}

cv::Mat RBFRegression::predict(data testdata)
{
    assert(testdata.cols == din);

    return calcActivation(testdata) * weights;
}
