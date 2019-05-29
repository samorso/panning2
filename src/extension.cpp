// [[Rcpp::depends(RcppEigen,RcppNumerical)]]
#include <RcppEigen.h>
#include <RcppNumerical.h>
#include <numeric>
#include <math.h>
#include <random>
#include <algorithm>
#include <vector>


// --------------
// Misc
// --------------
struct Sigmoid {
  Sigmoid(){}
  const double operator()(const double& x) const {return 0.1e1 / (0.1e1 + std::exp(-x));}
};

// --------------
// Logistic regression
// --------------
class mle_logistic: public Numer::MFuncGrad
{
private:
  const Eigen::ArrayXd y;
  const Eigen::MatrixXd X;
  const unsigned int n = y.size();
  const unsigned int p = X.cols();

public:
  mle_logistic(const Eigen::ArrayXd& y_,const Eigen::MatrixXd& X_) : y(y_), X(X_) {}
  double f_grad(Numer::Constvec& beta, Numer::Refvec grad);
};

double mle_logistic::f_grad(
    Numer::Constvec& beta,
    Numer::Refvec grad
){
  // data storage
  Eigen::ArrayXd sig(n);
  Eigen::VectorXd v(n);

  // pre-computation
  v = X * beta;
  sig = v.unaryExpr(Sigmoid());

  // computation
  // objective function
  const double f = -(0.1e1 - sig).log().sum() / n - v.dot(y.matrix()) / n;

  // gradient
  grad = X.transpose() * (sig - y).matrix() / n;
  return f;
}

// --------------
// Cross-validation for logistic regression
// --------------

//'Cross-validation for logistic regression with l2-norm error
//'
//'@param X a n x p matrix of regressor
//'@param y a n-vector of response
//'@param seed an integer for setting the seed (reproducibility)
//'@param K number of splits; 10 by default
//'@param M number of repetitions; 10 by default
//'@export
// [[Rcpp::export]]
double cross_validation_logistic_l2(
    Eigen::MatrixXd& X,
    Eigen::ArrayXd& y,
    unsigned int seed,
    unsigned int K = 10,
    unsigned int M = 10
){
  // Storage
  double err(0.0);
  unsigned int n = X.rows();
  unsigned int p = X.cols();
  unsigned int nn = n;
  unsigned int n_train, n_test;
  std::vector<int> ivec(n);
  std::iota(ivec.begin(),ivec.end(),0);
  std::vector<int> n_fold(K);

  std::mt19937_64 engine(seed);  // Mersenne twister random number engine

  for(unsigned int i = 0; i<K; i++){
    n_fold[i] = std::ceil(nn/(K-i));
    nn -= n_fold[i];
  }

  for(unsigned int m = 0; m < M; m++){
    // Shuffle the index
    std::shuffle(ivec.begin(),ivec.end(),engine);

    // K-fold CV on logitstic classification
    for(unsigned int k = 0; k < K; k++){
      // Seperate training/test sets
      n_train = n-n_fold[k];
      n_test = n_fold[k];
      Eigen::MatrixXd X_train(n_train,p),X_test(n_test,p);
      Eigen::VectorXd y_train(n_train),y_test(n_test),pred(n_test);

      unsigned int ii(0),jj(0),ind;

      for(unsigned int i = k+1; i < n+k+1; i++){
        if(i%K == 0){
          ind = ivec[i-k-1];
          X_test.row(ii) = X.row(ind);
          y_test(ii) = y(ind);
          ii++;
        }else{
          ind = ivec[i-k-1];
          X_train.row(jj) = X.row(ind);
          y_train(jj) = y(ind);
          jj++;
        }
      }

      // Regress
      double fopt;
      Eigen::VectorXd beta(p);
      beta.setZero();
      mle_logistic f(y_train,X_train);
      Numer::optim_lbfgs(f,beta,fopt);

      // Get the predictions
      pred = (X_test * beta).unaryExpr(Sigmoid());

      // Classification error
      y_test -= pred;
      ii++;
      err += y_test.dot(y_test) / ii;
    }
  }

  return err / K / M;
}
