---
layout: post
title: Volatility Models and Machine Learning
tags:
  - machine-learning
  - finance
---
{% include katex.html %}

{% katexmm %}

Uncertainty in the markets can be represented as a function of price fluctuation. The greater the price of an asset can change between any two periods of time, the less certain one can be about the future value of that asset. 

> The statistical measure of price fluctuation of an asset is known as volatility.

Volatility is therefore of the utmost importance in financial risk management in calculations such as the "Value at Risk" (VaR) which estimates how much a set of investments might lose (with a given probability)[^1].

[^1]: https://en.wikipedia.org/wiki/Value_at_risk

The aim of this post is to introduce to two classical volatility forecasting models and understand how can we make use of machine learning to derive an alternative model for volatility forecasting.

## GARCH(1,1)
A Generalized AutoRegressive Conditional Heteroskedasticity (GARCH) model assumes that an asset $S$ is has a constant rate of return $\mu$ and a volatility $\sigma^2_t$ that depends on the volatility and the return of the asset at time $t-1$ plus a constant term.

More formally, let $S_t$ be the value of the asset at time $t$, and $r_t = \log S_t / S_{t-1}$ the  log-return of the asset from time $t-1$ to $t$. We assume that $r_t \sim \mathcal{N}(0, \sigma^2_t)$ with 

$$
	\sigma^2_t = \omega + \alpha r_{t-1}^2 + \beta\sigma^2_{t-1}
$$

![return-history](https://i.imgur.com/hWy1LDn.png)

As with any other statistical model, we would like to find the values $\omega$, $\alpha$ and $\beta$ that best represent the data given the model. From a probabilistic perspective, finding $\omega$, $\alpha$,   and $\beta$ results in finding the *posterior* distribution of the parameters, i.e.,

$$
	p(\omega, \alpha, \beta \vert R) \propto p(\omega, \alpha, \beta)p(R \vert \omega, \alpha, \beta)
$$

In this post, we will not work with the fully-probabilistic model; we will obtain $(\omega, \alpha, \beta)$ via maximization of the likelihood. Consider

$$
\begin{aligned}
p(R \vert \omega, \alpha, \beta) &= \prod_{t=0}^Tp(r_t|\omega, \alpha, \beta)\\
&= p(r_0|\omega, \alpha, \beta) \prod_{t=1}^Tp(r_t|\omega, \alpha, \beta)\\
&= \mathcal{N}(r_0 \vert 0, \sigma_0^2)\prod_{t=1}^T\mathcal{N}(r_t\vert 0, \sigma^2_t)
\end{aligned}
$$

Note that we separated $r_0$ from the rest of the terms since it does not depend on the triplet $(\omega, \alpha, \beta)$, i.e., $\sigma_0$ is a constant with respect to the parameters.

Let $(\omega^*, \alpha^*, \beta^*)$ be the parameters that maximize $p(R\vert \omega, \alpha, \beta)$. Then,


$$
\begin{aligned}
(\omega^*, \alpha^*, \beta^*) &= \argmax_{(\omega, \alpha, \beta)} p(R\vert \omega, \alpha, \beta) \\
&= \argmax_{(\omega, \alpha, \beta)} \log p(R\vert \omega, \alpha, \beta) \\
&= \argmax_{(\omega, \alpha, \beta)} \sum_{t=1}^T
\log \mathcal{N}(r_t\vert 0, \sigma^2_t)\\
&= \argmax_{(\omega, \alpha, \beta)} -\frac{1}{2}\sum_{t=1}^T\left(\log \sigma^2_t + \frac{r_t^2}{\sigma_t^2}\right)\\
&= \argmin_{(\omega, \alpha, \beta)} \frac{1}{2}\sum_{t=1}^T\left(\log \sigma^2_t + \frac{r_t^2}{\sigma_t^2}\right)
\end{aligned}
$$

Where we denote the term $\frac{1}{2}\sum_{t=1}^T\left(\log \sigma^2_t + \frac{r_t^2}{\sigma_t^2}\right)$ as the negative log-likelihood of a GARCH(1,1) model.

Finally, we arrive at an optimization of the form

$$
\begin{aligned}
		\min_{(\omega, \alpha, \beta)} \quad &\frac{1}{2}\sum_{t=1}^T\left(\log \sigma^2_t + \frac{r_t^2}{\sigma_t^2}\right)\\
		s.t. \quad & \omega, \alpha, \beta \geq 0
\end{aligned}
$$

Note that each parameter is set to be nonnegative so as to ensure that no volatility term becomes negative. That is, we want to ensure that $\forall t.\sigma_t^2 \geq 0$.

### Implementation
The necessary setup to minimize the likelihood is not 
readily straightforward since every $t$ is completely dependent on $t- 1$. Thus, the optimization procedure must compute $\{\sigma_t\}_{t=1}^T$ sequentially for a given triplet $(\omega, \alpha, \beta)$ before taking a step towards a minimum.

A simple, yet supoptimal, way to compute the set $\{\sigma_t\}_{t=1}^T$ in Python is via a class such as the `GARCHSigmas` which we define below. This class takes fixed values `omega`, `alpha`, `beta`  as the parameters; `R` as the history of log-returns; and a parameter `sigma_init` that represents the initial volatility $\sigma_0^2$ which we will heuristically set it to be the historical variance of the dataset, i.e., $\sigma_0^2 = \mathbb{V}[R]$.

````python
class GARCHSigmas:
    """
    Class to estimate the sigma_t values
    of a GARCH(1, 1)
    """
    def __init__(self, omega, alpha, beta, sigma_init, R):
        self.omega = omega
        self.alpha = alpha
        self.beta = beta````
        self.R = R
        self.sigma_hist = [sigma_init]
        self.initialize_sigmas()
    
    def get_sigma1(self, t):
        rt0 = 0 if t == 0 else self.R.iloc[t - 1, 0]
        sigma0 = self.sigma_hist[-1]
        sigma1 = self.omega + self.alpha * rt0 ** 2 + self.beta * sigma0
        self.sigma_hist.append(sigma1)
    
    def initialize_sigmas(self):
        N, M = self.R.shape
        for t in range(N - 1, -1, -1):
            self.get_sigma1(t)
````

Once computed the set $\{\sigma_t\}_{t=1}^T$ we can evaluate the negative log-likelihood and optimize with respect to $(\omega, \alpha, \beta)$ for one step before repeating the processes all over again. This iterative procedure can be handled by a library such as _scipy_ using the function `scipy.optimize.minimize`.

Below we show the result of fitting a GARH(1,1) to the [NASDAQ-100 Index](https://www.bloomberg.com/quote/NDX:IND) using the Python class defined above.

![Imgur](https://i.imgur.com/Wu3Gls1.png)

## GJR-GACRH
The Glosten-Jagannathan-Runkle GARCH (GJR-GARCH) model is an alternative to a GARCH(1, 1) which considers an additional factor $\gamma$ that models the assymetric response of the market to the sign of the return. 

More formally,  a GJR-GARCH(1,1) model assumes that $r_t \sim \mathcal{N}(0, \sigma^2_t)$ with 

$$
	\sigma^2_t = \omega + (\alpha + \gamma \mathbb{I}_{r_{t-1 < 0}})  r_{t-1}^2 + \beta\sigma^2_{t-1} \tag{1}
$$

In $(1)$, $\mathbb{I}_{r_{t-1 < 0}}$ represents the indicator function of a negative return. That is 
$$
	\mathbb{I}_{r_{t-1 < 0}} = \begin{cases} 
			1 & r_{t-1} < 0 \\
	  0 & r_{t-1} \geq  0 
	\end{cases}
$$

Under the GJR-GARCH model, the log-likelihood function remains the same, but the estimation of the set $\{\sigma^2_t\}_{t=1}^T$ must change. To consider this changes, we slightly modify the `GARCHSigmas` class defined above to take account of the new term $\gamma$ in a class called `GJRSigma` which we define below:

````python
class GJRSigmas:
    """
    Class to estimate the sigma^2_t values
    of a GJR-GARCH
    """
    def __init__(self, omega, alpha, gamma, beta, sigma_init, R):
        self.omega = omega
        self.alpha = alpha
        self.gamma = gamma
        self.beta = beta
        self.R = R
        self.sigma_hist = [sigma_init]
        self.initialize_sigmas()
        
    def get_sigma1(self, t):
        rt0 = 0 if t == 0 else self.R.iloc[t - 1, 0]
        rt1 = self.R.iloc[t, 0]
        ind = 1 if rt0 < 0 else 0
        sigma0 = self.sigma_hist[-1]
        sigma1 = self.omega + (self.alpha + self.gamma * ind) * rt0 ** 2 + self.beta * sigma0
        self.sigma_hist.append(sigma1)
    
    def initialize_sigmas(self):
        N, M = self.R.shape
        for t in range(N - 1, -1, -1):
            self.get_sigma1(t)

```` 

The main difference between the  `GARCHSigmas` and `GJRSigmas` lies in the extra term `self.gamma * ind` in defining `sigma1`, which corresponds to the definition of $\sigma_t^2$ for the GJR-GARCH model.

The optimization problem for the GJR-GARCH closely resembles that of a GARCH model, that is,  we require $\omega, \alpha, \beta$ to be nonnegative. Furthermore, consider two more constraints:

1. that $\alpha + \gamma \geq 0$ so as to ensure positive volatility; and
2. the [covariance stationary](https://www.statlect.com/glossary/covariance-stationary) constraint given by $\alpha + \frac{1}{2}\gamma + \beta < 1$.


Thus, our optimization problem is written as follows:
$$
\begin{aligned}
		\min_{(\omega, \alpha, \beta)} \quad &\frac{1}{2}\sum_{t=1}^T\left(\log \sigma^2_t + \frac{r_t^2}{\sigma_t^2}\right)\\
		s.t. \quad & \omega, \alpha, \beta \geq 0\\
		& \alpha + \gamma\geq 0\\
		&\alpha + \frac{1}{2}\gamma + \beta < 1
\end{aligned}
$$


As previously done with the GARCH model, we can fit a GJR-GARCH to the NASDAQ-100 Index. We show its result below with an exponentially-weighted moving average with decay of around 0.9.

![ewma-nasdaq100-gjr-garch](https://i.imgur.com/JMMIoGK.png)



## The Machine Learning Approach
A main drawback of either the GJR-GARCH or GARCH model is the fact that its forecast can only be used to estimate the volatility one day ahead. To overcome this limitation, [Monfared and Enke (2014)](https://www.sciencedirect.com/science/article/pii/S1877050914013362?via%3Dihub) proposed a hybrid model to estimate the volatility of a GJR-GARCH(1,1) model over the next 40 days. In their paper, they consider 10 Nasdaq indices[^2] and fit a GJR-GARCH(1,1) over each index. Then, they fit a feed-forward neural network with 9 inputs; 20 hidden layers and 40 output units; one for each forecast day of the GJR-GARCH model. A sketch of the network is presented below.

![NN-GJR](https://i.imgur.com/TndAlB0.png)

What is interesting about their proposal is the use of pre-trained GJR-GARCH models as preprocessed input to a machine learning algorithm. 

Although training the model and testing it under current market conditions did not yield results that resembled that of the research, the idea of a classical model as input to learn signals is what is remarkable.

As an example of further use the work of  of this would be to combine the the work of Monfared and Enke (2014) with that of  [Bontempi, Gianluca & Ben Taieb, Souhaib & Le Borgne, Yann-Aël. (2013)]([https://www.researchgate.net/publication/236941795_Machine_Learning_Strategies_for_Time_Series_Forecasting](https://www.researchgate.net/publication/236941795_Machine_Learning_Strategies_for_Time_Series_Forecasting))  in which they present a general framework to adjust any time series dataset into a supervised learning algorithm. For example, by considering more than one day as input to the model.

In conclusion, classical volatility models, although limited in their predictive power, can be put to use to find signal in the markets. 

[^2]: Bank (CBNK), Biotechnology (CCMP), Composite (CFIN), Computer (CIND), Industrial (CINS), Insurance (CTRN), Other Finance (non-Bank) (CUTL), Telecommunication (IXK), Transportation (NBI), NASDAQ 100 (NDX)


{% endkatexmm %}
