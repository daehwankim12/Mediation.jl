using DataFrames, CSV, GLM, Mediation

da = open("framing.csv") do io
    CSV.read(io, DataFrame)
end

med = Model(LinearModel, @formula(emo ~ treat + age + educ + gender + income))

out = Model(GeneralizedLinearModel, @formula(cong_mesg ~ emo + treat + age + educ + gender + income), Binomial())

r = mediate(med, out, da, :treat, :emo, :cong_mesg; pertmeth=:bootstrap)
