using DataFrames, CSV, GLM, Mediation

da = open("framing.csv") do io
    CSV.read(io, DataFrame)
end

m1 = lm(@formula(emo ~ treat + age + educ + gender + income), da)
m2 = glm(@formula(cong_mesg ~ emo + treat + age + educ + gender + income), da, Binomial())

r = mediate(m1, m2, da, :treat, :emo, :cong_mesg)
