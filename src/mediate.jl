using DataFrames, StatsModels, LinearAlgebra, Statistics

# ppred takes a fitted regression model m and a dataframe x that
# contains all the variables in m.  Then the regression parameters in
# m are perturbed using a Gaussian approximation with covariance equal
# to their sampling covariance matrix.  Next the variables in x are
# used to form predicted values of the linear predictor using the
# perturbed model parameters.  Finally, draws from the predictive
# distribution of the model are made, conditionally on the linear
# predictor.
function ppred(
    m::StatsModels.TableRegressionModel,
    xl::Array{DataFrame,1},
)::Array{Float64,2}

    n = size(xl[1], 1)

    # Get the perturbed parameters
    c = vcov(m)
    r = cholesky(c)
    par0 = coef(m)
    p = length(par0)
    par = par0 + r.L * randn(p)

    # Generate a dataframe using the formula
    xf = []
    for x in xl
        f = apply_schema(m.mf.f, schema(m.mf.f, x))
        _, xx = modelcols(f, x)
        push!(xf, xx)
    end

    # The perturbed linear predictor
    lp = [x * par for x in xf]

    # The distribution family
    d = Normal(0, 1)
    if typeof(m.model) <: GeneralizedLinearModel
        d = m.model.rr.d
    end

    # The scale parameter (standard deviation)
    di = GLM.dispersion(m.model)

    yy = zeros(n, length(xl))

    if typeof(d) <: Normal
        for (j, ll) in enumerate(lp)
            for i = 1:n
                yy[i, j] = ll[i] + di * randn()
            end
        end
    elseif typeof(d) <: Binomial
        for (j, ll) in enumerate(lp)
            for i = 1:n
                yy[i, j] = rand() < 1 / (1 + exp(-ll[i])) ? 1 : 0
            end
        end
    elseif typeof(d) <: Poisson
        for (j, ll) in enumerate(lp)
            for i in eachindex(y)
                yy[i, j] = rand(Poisson(exp(ll[i])))
            end
        end
    else
        error("Distribution not supported")
    end

    return yy

end


"""
    mediate(m1, m2, x, ex, md, ot; expvals, nrep)

Perform a mediation analysis based on the mediator model specified
by `m1` and the outcome model specified by `m2`.

# Keyword Arguments
- `m1::TableRegressionModel`: The mediator model (includes the exposure
as a covariate).
- `m2::TableRegressionModel`: The outcome model (includes the exposure
and mediator as covariates).
- `x::DataFrame`: A dataframe containing all covariates and outcomes
in the exposure and mediator models.
- `ex::Symbol`: The name of the exposure variable.
- `md::Symbol`: The name of the mediator variable.
- `ot::Symbol`: The name of the outcome variable.

# Keyword Arguments
- `expvals::Symbol`: Two reference points in the domain of the exposure variable.
- `nrep::Integer`: The number of Monte Carlo replications.
"""
function mediate(
    m1::StatsModels.TableRegressionModel,
    m2::StatsModels.TableRegressionModel,
    x::DataFrame,
    ex::Symbol,
    md::Symbol,
    ot::Symbol;
    expvals = [0, 1],
    nrep::Int = 1000,
)

    rslt = DataFrame(
        :Name => [
            "Indirect (t=1)",
            "Indirect (t=0)",
            "Direct (t=1)",
            "Direct (t=0)",
            "Total",
        ],
        :Estimate => zeros(5),
        :StdErr => zeros(5),
        :LCB => zeros(5),
        :UCB => zeros(5),
    )

    n = size(x, 1)
    xx = [copy(x), copy(x), copy(x), copy(x)]

    dir0, ind0, tot = [], [], []
    dir1, ind1 = [], []
    for k = 1:nrep

        # t/m = 0/0, 0/1, 1/0, 1/1
        for j in eachindex(expvals)
            xx[j][!, ex] .= expvals[j]
        end
        z = ppred(m1, xx[1:2])
        for j in eachindex(expvals)
            xx[j][!, md] = z[:, j]
            xx[j+2][!, md] = z[:, j]
        end
        for j in eachindex(expvals)
            xx[2*j-1][!, ex] .= expvals[j]
            xx[2*j][!, ex] .= expvals[j]
        end
        yy = ppred(m2, xx)

        # If an outcome is observed, we don't
        # need to simulate it.
        ii = x[:, ex] .== expvals[1]
        yy[ii, 1] = x[ii, ot]
        ii = x[:, ex] .== expvals[2]
        yy[ii, 4] = x[ii, ot]

        # Indirect effects
        push!(ind0, mean(yy[:, 2] - yy[:, 1]))
        push!(ind1, mean(yy[:, 4] - yy[:, 3]))

        # Direct effects
        push!(dir0, mean(yy[:, 3] - yy[:, 1]))
        push!(dir1, mean(yy[:, 4] - yy[:, 2]))

        # Total effects
        push!(tot, mean(yy[:, 4] - yy[:, 1]))

    end

    rslt[1, 2:end] = [mean(ind1), std(ind1), quantile(ind1, 0.025), quantile(ind1, 0.975)]
    rslt[2, 2:end] = [mean(ind0), std(ind0), quantile(ind0, 0.025), quantile(ind0, 0.975)]
    rslt[3, 2:end] = [mean(dir1), std(dir1), quantile(dir1, 0.025), quantile(dir1, 0.975)]
    rslt[4, 2:end] = [mean(dir0), std(dir0), quantile(dir0, 0.025), quantile(dir0, 0.975)]
    rslt[5, 2:end] = [mean(tot), std(tot), quantile(tot, 0.025), quantile(tot, 0.975)]

    return rslt

end
