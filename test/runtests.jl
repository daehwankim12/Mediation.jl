using Test, GLM, DataFrames, Mediation, Random

@testset "linear1" begin

    # Generate data where the true indirect,
    # direct, and total effects are 1.5, 0.5,
    # and 2, respectively.
    Random.seed!(3324)
    n = 1000
    x = [rand() < 0.5 ? 1 : 0 for _ = 1:n]
    m = x + randn(n)
    y = 1.5 * m + 0.5 .* x + randn(n) # partial mediation
    da = DataFrame(:y => y, :m => m, :x => x)

    m1 = lm(@formula(m ~ x), da)
    m2 = lm(@formula(y ~ m + x), da)

    r = mediate(m1, m2, da, :x, :m, :y, expvals = [0, 1])

    @test isapprox(r[1, 2], 1.5, atol = 0.05)
    @test isapprox(r[2, 2], 1.5, atol = 0.05)
    @test isapprox(r[3, 2], 0.5, atol = 0.05)
    @test isapprox(r[4, 2], 0.5, atol = 0.05)
    @test isapprox(r[5, 2], 2, atol = 0.05)

end

@testset "linear_logit1" begin

    Random.seed!(3324)
    n = 10000
    x = [rand() < 0.5 ? 1 : 0 for _ = 1:n]
    m = x + randn(n)
    pr = 1 ./ (1 .+ exp.(-x)) # no mediation
    y = [rand() < pr[i] ? 1 : 0 for i = 1:n]
    da = DataFrame(:y => y, :m => m, :x => x)

    m1 = lm(@formula(m ~ x), da)
    m2 = glm(@formula(y ~ m + x), da, Binomial())

    r = mediate(m1, m2, da, :x, :m, :y, expvals = [0, 1])

    @test isapprox(r[1, 2], 0.0, atol = 0.02)
    @test isapprox(r[2, 2], 0.0, atol = 0.02)
    @test isapprox(r[3, 2], 0.23, atol = 0.02)
    @test isapprox(r[4, 2], 0.23, atol = 0.02)
    @test isapprox(r[5, 2], 0.23, atol = 0.02)

end
