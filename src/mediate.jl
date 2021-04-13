using DataFrames, StatsModels, LinearAlgebra, Statistics, Distributions

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


# Return a draw from the approximate sampling distribution of the
# model parameters.
function pert_asymp(m::StatsModels.TableRegressionModel)::Array{Float64,1}

    c = vcov(m.model)
    r = cholesky(c)
    par0 = coef(m.model)
    p = length(par0)
    par = par0 + r.L * randn(p)
    return par

end

# Return a draw from the approximate sampling distribution of the
# model parameters using bootstrapping.
function pert_boot(m::FormulaModel, y::Array{Float64,1}, x::Array{Float64,2})::Array{Float64,1}

    n = length(y)
    ii = collect(1:n)
    ix = sample(ii, n, replace=true)
    r = fit(m.class, x[ii, :], y[ii], m.args...; m.kwargs...)
    return coef(r)

end


function genrand(d::Normal{Float64}, lp::Array{Array{Float64,1}}, di::Float64, yy::Array{Float64,2})
    for (j, ll) in enumerate(lp)
        for i in eachindex(ll)
            yy[i, j] = ll[i] + di * randn()
        end
    end
end

function genrand(d::Binomial{Float64}, lp::Array{Array{Float64,1}}, di::Float64, yy::Array{Float64,2})
    for (j, ll) in enumerate(lp)
        for i in eachindex(ll)
            yy[i, j] = rand() < 1 / (1 + exp(-ll[i])) ? 1 : 0
        end
    end
end

function genrand(d::Poisson{Float64}, lp::Array{Array{Float64,1}}, di::Float64, yy::Array{Float64,2})
    for (j, ll) in enumerate(lp)
        for i in eachindex(ll)
            yy[i, j] = rand(Poisson(exp(ll[i])))
        end
    end
end

# ppred takes a fitted regression model m and a dataframe x that
# contains all the variables in m.  Then the regression parameters in
# m are perturbed using a Gaussian approximation with covariance equal
# to their sampling covariance matrix.  Next the variables in x are
# used to form predicted values of the linear predictor using the
# perturbed model parameters.  Finally, draws from the predictive
# distribution of the model are made, conditionally on the linear
# predictor.
function ppred(
    m::FormulaModel,
    f::StatsModels.TableRegressionModel,
    x::AbstractDataFrame,
    y_m::Array{Float64,1},
    x_m::Array{Float64,2},
    xl::Array{DataFrame,1},
    pertmeth::Symbol
)::Array{Float64,2}

    n = size(xl[1], 1)

    # Get the perturbed parameters
    if pertmeth == :asymptotic
        par = pert_asymp(f)
    elseif pertmeth == :bootstrap
        par = pert_boot(m, y_m, x_m)
    else
        error("Unknown parameter perturbation method")
    end

    # Generate a dataframe using the formula
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
    pertmeth = :asymptotic,
    nrep::Int = 1000,
)

    # Fit both models with full data
    fml_med = apply_schema(m_med.formula, schema(m_med.formula, x), m_med.class)
    f_med = fit(m_med.class, fml_med, x, m_med.args...; m_med.kwargs...)
    fml_out = apply_schema(m_out.formula, schema(m_out.formula, x), m_out.class)
    f_out = fit(m_out.class, fml_out, x, m_out.args...; m_out.kwargs...)

    y_med, x_med, y_out, x_out = zeros(0), zeros(0, 0), zeros(0), zeros(0, 0)
    if pertmeth == :bootstrap
        y_med, x_med = modelcols(fml_med, x)
        y_med = convert.(Float64, y_med)
        y_out, x_out = modelcols(fml_out, x)
        y_out = convert.(Float64, y_out)
    end

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
        z = ppred(m_med, f_med, x, y_med, x_med, xx[1:2], pertmeth)
        for j in eachindex(expvals)
            xx[j][!, md] = z[:, j]
            xx[j+2][!, md] = z[:, j]
        end
        for j in eachindex(expvals)
            xx[2*j-1][!, ex] .= expvals[j]
            xx[2*j][!, ex] .= expvals[j]
        end
        yy = ppred(m_out, f_out, x, y_out, x_out, xx, pertmeth)

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
