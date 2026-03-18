
data {
int<lower=0> N; //sample size with a minimum value of 0
int k; //data denoting whether a girl or boy is born
}
transformed data {
}
parameters {
real<lower=0, upper=1> theta; //probability of having a girl
}
transformed parameters {
}
model {
// Priors
theta ~ beta(1,1); //the same as uniform!
// Likelihood
k ~ binomial(N, theta);
}
generated quantities{
}
