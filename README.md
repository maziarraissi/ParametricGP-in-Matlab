# ParametricGP

This work introduces the concept of parametric Gaussian processes (PGPs), which is built upon the seemingly self-contradictory idea of making Gaussian processes parametric. Parametric Gaussian processes, by construction, are designed to operate in "big data" regimes where one is interested in quantifying the uncertainty associated with noisy data. The proposed methodology circumvents the well-established need for stochastic variational inference, a scalable algorithm for approximating posterior distributions. The effectiveness of the proposed approach is demonstrated using an illustrative example with simulated data and a benchmark dataset in the airline industry with approximately 6 million records.

For more details, please refer to the following: (https://arxiv.org/abs/1704.03144)

  - Raissi, Maziar. "Parametric Gaussian Process Regression for Big Data." arXiv preprint arXiv:1704.03144 [stat.ML] (2017).

## Citation

    @Misc{gpy2014,
      author =   {{GPy}},
      title =    {{GPy}: A Gaussian process framework in python},
      howpublished = {\url{http://github.com/SheffieldML/GPy}},
      year = {since 2012}
    }
