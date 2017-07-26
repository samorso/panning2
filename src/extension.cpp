// [[Rcpp::depends(RcppMLPACK)]]
#include <RcppMLPACK.h>
#include <RcppArmadillo.h>
#include <numeric>
#include <math.h>

// Enable C++11 via the Rcpp plugin
// [[Rcpp::plugins("cpp11")]]

// [[Rcpp::export]]
arma::vec log_reg(
  arma::mat X,
  arma::vec y
){
  // Regress
  mlpack::regression::LogisticRegression<> lr(X,y);
  // Get the parameters
  arma::vec parameters = lr.Parameters();
  
  return parameters;
}

// [[Rcpp::export]]
arma::vec pred_log_reg(
    arma::mat X,
    arma::vec y,
    arma::mat X_new
){
  // Regress
  mlpack::regression::LogisticRegression<> lr(X,y);
  // Get the predictions
  arma::vec prediction;
  lr.Predict(X_new,prediction);
  
  return prediction;
}

// [[Rcpp::export]]
double cross_validation_logistic(
  arma::mat& X,
  arma::vec& y,
  unsigned int K,
  unsigned int M
){
  double err(0.0);
  unsigned int n;
  unsigned int p;
  unsigned int nn;
  unsigned int n_train;
  unsigned int n_test;
  
  n = y.n_rows;
  nn = n;
  p = X.n_rows;
  
  arma::uvec ivec(n);
  arma::uvec n_fold(K);
  
  std::iota(ivec.begin(), ivec.end(), 0);
  
  for(unsigned int i = 0; i<K; i++){
    n_fold[i] = std::ceil(nn/(K-i));
    nn -= n_fold[i];
  }
  
  for(unsigned int m = 0; m < M; m++){
    // Shuffle the index
    ivec = shuffle(ivec);
    
    // K-fold CV on logitstic classification
    for(unsigned int k = 0; k < K; k++){
      // Seperate training/test sets
      n_train = n-n_fold[k];
      n_test = n_fold[k];
      arma::uvec i_train(n_train);
      arma::uvec i_test(n_test);
      arma::uvec index_train(n_train);
      arma::uvec index_test(n_test);
      arma::mat X_train(p,n_train);
      arma::mat X_test(p,n_test);
      arma::vec y_train(n_train);
      arma::vec y_test(n_test);
      arma::vec prediction(n_test);
      
      unsigned int ii = 0;
      unsigned int jj = 0;
      
      for(unsigned int i = k+1; i < n+k+1; i++){
        if(i%K == 0){
          index_test[ii] = i-k-1;
          ii++;
        }else{
          index_train[jj] = i-k-1;
          jj++;
        }
      }
      
      i_train = ivec.elem(index_train);
      i_test = ivec.elem(index_test);
      
      X_train = X.cols(i_train);
      X_test = X.cols(i_test);
      y_train = y.elem(i_train);
      y_test = y.elem(i_test);
      
      // Regress
      mlpack::regression::LogisticRegression<> lr(X_train,y_train);
      // Get the predictions
      lr.Predict(X_test,prediction);
      
      // Classification error
      y_test -= prediction;
      ii++;
      err += sum(arma::abs(y_test))/ii;
    }
  }
  
  return err / K / M;
}
