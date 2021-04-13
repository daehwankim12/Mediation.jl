using DataFrames, StatsModels, LinearAlgebra, Statistics, Distributions, StatsBase

# Model wrappers borrowed from github.com/juliagehring/Bootstrap.jl.

abstract type Model end

struct SimpleModel{T,A,K} <: Model
    class::T
    args::A
    kwargs::K
end

Model(class, args...; kwargs...) = SimpleModel(class, args, kwargs)

struct FormulaModel{T,A,K} <: Model
    class::T
    formula::FormulaTerm
    args::A
    kwargs::K
end

Model(class, formula::FormulaTerm, args...; kwargs...) =
    FormulaModel(class, formula, args, kwargs)

abstract type ParameterPerturber end

mutable struct AsymptoticPerturber <: ParameterPerturber
    r::Cholesky
    par0::Array{Float64,1}
end

# Perturb the parameters as a draw from the asymptotic Gaussian approximation to
# the sampling distribution of the parameters.
function AsymptoticPerturber(m::StatsModels.TableRegressionModel)::ParameterPerturber
    c = vcov(m.model)
    r = cholesky(c)
    par0 = coef(m.model)
    return AsymptoticPerturber(r, par0)
end

# Return a draw from the approximate sampling distribution of the
# model parameters.
function sample(pt::AsymptoticPerturber)::Array{Float64,1}
    p = length(pt.par0)
    return pt.par0 + pt.r.L * randn(p)
end

mutable struct BootstrapPerturber <: ParameterPerturber
    m::FormulaModel
    y::Array{Float64,1}
    x::Array{Float64,2}
end

# Return a draw from the approximate sampling distribution of the
# model parameters using bootstrapping.
function sample(pt::BootstrapPerturber)::Array{Float64,1}
    n = length(pt.y)
    ii = collect(1:n)
    ix = StatsBase.sample(ii, n, replace = true)
    r = fit(pt.m.class, pt.x[ii, :], pt.y[ii], pt.m.args...; pt.m.kwargs...)
    return coef(r)
end


function getPerturber(
    ::Type{AsymptoticPerturber},
    m::FormulaModel,
    ft::FormulaTerm,
    f::RegressionModel,
    x::AbstractDataFrame,
)
    return AsymptoticPerturber(f)
end

function getPerturber(
    ::Type{BootstrapPerturber},
    m::FormulaModel,
    ft::FormulaTerm,
    f::RegressionModel,
    x::AbstractDataFrame,
)
    y, x = modelcols(ft, x)
    y = convert.(Float64, y)
    return BootstrapPerturber(m, y, x)
end


function genrand(
    d::Normal{Float64},
    lp::Array{Array{Float64,1}},
    di::Float64,
    yy::Array{Float64,2},
)
    for (j, ll) in enumerate(lp)
        for i in eachindex(ll)
            yy[i, j] = ll[i] + di * randn()
        end
    end
end

function genrand(
    d::Binomial{Float64},
    lp::Array{Array{Float64,1}},
    di::Float64,
    yy::Array{Float64,2},
)
    for (j, ll) in enumerate(lp)
        for i in eachindex(ll)
            yy[i, j] = rand() < 1 / (1 + exp(-ll[i])) ? 1 : 0
        end
    end
end

function genrand(
    d::Poisson{Float64},
    lp::Array{Array{Float64,1}},
    di::Float64,
    yy::Array{Float64,2},
)
    for (j, ll) in enumerate(lp)
        for i in eachindex(ll)
            yy[i, j] = rand(Poisson(exp(ll[i])))
        end
    end
end

# contains all the variables in m.  Then the regression parameters in
# ppred takes a fitted regression model m and a dataframe x that
# m are perturbed using a Gaussian approximation with covariance equal
# to their sampling covariance matrix.  Next the variables in x are
# used to form predicted values of the linear predictor using the
# perturbed model parameters.  Finally, draws from the predictive
# distribution of the model are made, conditionally on the linear
# predictor.
function ppred(
    f::StatsModels.TableRegressionModel,
    x::AbstractDataFrame,
    pt::ParameterPerturber,
    xl::Array{DataFrame,1},
)::Array{Float64,2}

    n = size(xl[1], 1)

    # Get the perturbed parameters
    par = sample(pt)

    # Generate dataframes using the formula
    xf = []
    for x in xl
        fs = apply_schema(f.mf.f, schema(f.mf.f, x))
        _, xx = modelcols(fs, x)
        push!(xf, xx)
    end

    # The perturbed linear predictor
    lp = [x * par for x in xf]

    # The distribution family
    d = Normal(0, 1)
    if typeof(f.model) <: GeneralizedLinearModel
        d = f.model.rr.d
    end

    # The scale parameter (standard deviation)
    di = GLM.dispersion(f.model)

    yy = zeros(n, length(xl))
    genrand(d, lp, di, yy)

    return yy

end


"""
    mediate(m_med, m_out, x, ex, md, ot; expvals, nrep)

Perform a mediation analysis based on the mediator model specified
by `m_med` and the outcome model specified by `m_out`.

- `m_med::FormulaModel`: The mediator model (includes the exposure
as a covariate).
- `m_out::FormulaModel`: The outcome model (includes the exposure
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
    m_med::FormulaModel,
    m_out::FormulaModel,
    x::AbstractDataFrame,
    ex::Symbol,
    md::Symbol,
    ot::Symbol;
    expvals = [0, 1],
    pert = AsymptoticPerturber,
    nrep::Int = 1000,
)

    # Fit both models with full data
    ft_med = apply_schema(m_med.formula, schema(m_med.formula, x), m_med.class)
    fit_med = fit(m_med.class, ft_med, x, m_med.args...; m_med.kwargs...)
    ft_out = apply_schema(m_out.formula, schema(m_out.formula, x), m_out.class)
    fit_out = fit(m_out.class, ft_out, x, m_out.args...; m_out.kwargs...)

    medpert = getPerturber(pert, m_med, ft_med, fit_med, x)
    outpert = getPerturber(pert, m_out, ft_out, fit_out, x)

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

    # Dataframes for setting counterfactual values
    # of treatment and/or mediator.
    xx = [copy(x), copy(x), copy(x), copy(x)]

    dir0, ind0, tot = [], [], []
    dir1, ind1 = [], []
    for k = 1:nrep

        # Simulate potential outcomes for all exposure/mediator combinations
        # in this order: t/m = 0/0, 0/1, 1/0, 1/1
        for j in eachindex(expvals)
            xx[j][!, ex] .= expvals[j]
        end
        z = ppred(fit_med, x, medpert, xx[1:2])
        for j in eachindex(expvals)
            xx[j][!, md] = z[:, j]
            xx[j+2][!, md] = z[:, j]
        end
        for j in eachindex(expvals)
            xx[2*j-1][!, ex] .= expvals[j]
            xx[2*j][!, ex] .= expvals[j]
        end
        yy = ppred(fit_out, x, outpert, xx)

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
