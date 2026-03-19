#Cohort Component Model of Projections (female dominant perspective, one or two sexes, closed vs open population), by JE Romero-Prieto, PhD#
#https://github.com/Romero-Prieto/teaching#
rm(list = ls())                                                                 #Clear all generated data if any.
#install.packages("stringr")
#install.packages("Matrix")
#install.packages("expm")
library(stringr)                                                                #For string manipulation.
library(Matrix)                                                                 #For matrix operations. 
library(expm)                                                                   #For exponential matrix calculation.

GitHub            = "https://raw.githubusercontent.com/Romero-Prieto/teaching/main/Population%20Dynamics%20and%20Projections/Preston_6.csv"
input             = read.csv(GitHub)                                            #Reads the data of Preston et al. 2001. "Population projections." Pp. 117-137 in Demography: measuring and modeling population processes. Malden, MA: Blackwell Publishers.
input             = as.matrix(input)                                            #Reads the data as a matrix.
female            = input[,c("age","nLx_f","nFx","nNx_f")] 
View(female)

# Definitions:
# It is useful to define some parameters before beginning the calculations. These parameters are not strictly necessary, but they make the code more readable and easier to modify if needed. Those parameters are: the baseline year, the number of age groups, the radix of the life table, the length of the age intervals—which is the same for all—and the sex ratio at birth.
year              = 1993.0
ages              = nrow(female)
radix             = 100000
n                 = 5
SRB               = 1.05

# Construction of the Leslie matrix:
# The first sub-diagonal of the Leslie matrix contains the survivorship ratios, which are calculated from the nLx column of a life table. We can generate a variable nLx and calculate the survivorship ratios nSx, keeping in mind that different formulas are used for the first age interval, the open‑ended age interval, and all intermediate age intervals. 
nLx               = female[,"nLx_f"]
nSx               = c(nLx[1]/(n*radix),nLx[2:(ages-1)]/nLx[1:(ages-2)],nLx[ages]/sum(nLx[(ages-1):ages])) 

# To calculate the projected number of births, the cohort component model makes specific assumptions about the fertility schedule of the population. We can define a function nGx to incorporate these assumptions.
nFx               = female[,"nFx"]
nGx               = (nFx + c(nFx[2:ages]*nSx[2:ages],0))*n/2

# We define the Leslie matrix as a square matrix whose order is equal to the number of age groups. Note that, before the matrix is populated, all entries are set to zero. The first row contains a set of multipliers related to reproduction, determined by the survivorship ratio of the first age interval, the fertility schedule, and the sex ratio at birth, as shown:
lESliE            = matrix(0,ages,ages)
lESliE[1,]        = nGx*1/(1 + SRB)*nSx[1]

# We then populate the sub‑diagonal with the survivorship ratios for all intermediate age intervals, and place the survivorship ratio for the open‑ended age interval in the last entry of the last column, as shown:
diag(lESliE[-1,]) = nSx[2:ages]
lESliE[ages,ages] = nSx[ages]

# Projecting the female population:
# Matrix multiplication simplifies the projection to a single operation. We simply multiply the Leslie matrix by the population vector at time t to obtain the projected population at time t + n. Note that R uses a very specif notation for matrix multiplication (i.e., the operator %*%), which is different from the notation for the element-wise multiplication (i.e., the operator *).
current           = female[,"nNx_f"]
projected         = lESliE%*%current

# Let’s produce a simple table of results by projecting the population for three consecutive periods while holding fertility and mortality constant (i.e., using the same Leslie matrix throughout). We first retain only the relevant information and rename the current population using the reference year.
female            = input[,c("age","nNx_f")] 
colnames(female)[colnames(female) == "nNx_f"] <- str_c("nNx_",sprintf("%.1f", year))

# We then use a loop to project the population iteratively, given the most recent data at each step, as shown:
for (t in seq(year, year + 2*n, n)) {
  projected = lESliE%*%female[,str_c("nNx_",sprintf("%.1f", t))] #Projects
  colnames(projected)[1] <- str_c("nNx_",sprintf("%.1f", t + n)) #Gives a name to the projected data.
  female = cbind(female,projected)                               #Concatenates the table.
}

# Finally, we print the resulting table:
print(female)


# Example Questions:
# a. What would the reference year be if the input were the mid‑year population for 1993 (i.e., July 1st, 1993)?
# The reference year would be 1993.5  
# b. How can we project the population for 100 years without using a loop?
# We can use the exponential of the Leslie matrix to project the population for multiple periods. For example, to project the population for 100 years (i.e., 20 periods of 5 years each), we can calculate the exponential of the Leslie matrix raised to the power of 20 and then multiply it by the initial population vector:
projected         = (lESliE%^%20)%*%female[,"nNx_1993.0"]
# c. Quantifying the projected number of births is no longer a necessary step when using a Leslie matrix. Since this quantity remains important for many research questions, how can we estimate the projected number of births by age of the mother when following the Leslie matrix approach?
# The projected number of births by age of the mother can be estimated by multiplying the age-specific fertility rates by the exposure to the risk of childbearing. Keeping the same assumptions of the cohort component model, the exposure is calculated as the average number of the population at the beginning and at the end of the projection interval.
B                 = nFx*n/2*(female[,"nNx_1993.0"] + lESliE%*%female[,"nNx_1993.0"])
# Which is the same as the following, but with a more compact code:
B                 = nFx*n/2*(diag(ages) + lESliE)%*%female[,"nNx_1993.0"]

# Compulsory Questions:
# a. Calculate the life expectancy at birth of this population.  
sum(nLx)/radix
# b. Calculate the Total Fertility Rate of this population.  
sum(nFx*n)
# c. Calculate the Net Reproduction Rate of this population.  
sum(nFx*nLx/radix)*(1/(1 + SRB))
# d. Calculate the projected number of births from 1993.0 to 1998.0 by age group of the mother.  
# Short answer   
current           = female[,"nNx_1993.0"]
B                 = n*nFx*(current + lESliE%*%current)/2
#Same answer with a fancy table
femaleS           = female[,c("age",str_c("nNx_",sprintf("%.1f", year)),str_c("nNx_",sprintf("%.1f", year + n)))]
B                 = matrix(rowSums(femaleS[,c(2,3)])/2*n*nFx,ages,1)
colnames(B)[1] <- str_c("B[",sprintf("%.1f", year),",",sprintf("%.1f", year + n),")")
femaleS           = cbind(femaleS,B)

# e. Calculate the projected total number of births from 1993.0 to 1998.0. 
sum(B)
# f. Calculate how many are projected to be male births.  
sum(B)*(SRB/(1 + SRB))
# g. Calculate the projected number of births from 1993.0 to 1998.0 by cohort of the mother.  
B                 = matrix(rowSums(femaleS[,c(2,3)])*n/2*nFx,ages,1)
x                 = femaleS[,"age"]
cohort            = matrix(paste(as.character(year-x-n), as.character(year-x-1), sep = "-"),ages)
colnames(cohort)[1] <- "Birth Cohort"
B                 = matrix(femaleS[,str_c("nNx_",sprintf("%.1f", year))]*nGx,ages,1)
colnames(B)[1] <- str_c("Bc[",sprintf("%.1f", year),",",sprintf("%.1f", year + n),")")
femaleC           = data.frame(cohort,B)
print(femaleS)
print(femaleC)


# Exercise 2: Cohort Component Model of Projections (two sexes, female dominant perspective, closed population)
# In this exercise, we extend the one‑sex model to project both sexes simultaneously. For now, we continue to assume a female‑dominant perspective and a closed population. We use the same data, so there is no need to modify the setup or the definition of key parameters. The only change is to read the data for each sex separately and to construct an augmented Leslie matrix that accounts for both sexes.
# The data:
female            = input[,c("age","nLx_f","nFx","nNx_f")]
male              = input[,c("age","nLx_m","nNx_m")]
# Construction of the augmented Leslie matrix:
# The Leslie matrix is consisting of four sub-matrices representing: (1) how the female population survives and is self-reproduced, i.e., to project female population using only female information, lESliE_ff; (2) how the male population survives, i.e., to project male population using only male information, lESliE_mm; (3) how the male population reproduces from the female population, i.e., to project the number of male births using female information, lESliE_fm; and (4) how the female population would reproduce from the male population, i.e., to project the number of female births using male information, lESliE_mf, which is not the case in a female dominant perspective.  
# Note that we already know how the female population survives and self‑reproduces, so we can simply use the Leslie matrix from the Exercise 1. We then need to construct the remaining (two) sub‑matrices, using as an initial condition that all entries are set to zero, as shown.
# Initial conditions
lESliE_ff         = lESliE 
lESliE_mm         = matrix(0,ages,ages)
lESliE_fm         = matrix(0,ages,ages)
lESliE_mf         = matrix(0,ages,ages)

# Male population
# The first step is to calculate the survivorship ratios using the nLx column.
nLx               = male[,"nLx_m"]
nSx               = c(nLx[1]/(n*radix),nLx[2:(ages-1)]/nLx[1:(ages-2)],nLx[ages]/sum(nLx[(ages-1):ages])) 

# The second step is to insert the multipliers needed to project the first age group of the male population, based on the fertility schedule and the age distribution of women in the reproductive ages. Note that we have used the same nGx values as in Exercise 1, but we now apply the male survivorship ratios nSx and multiply by the sex ratio at birth (SRB).
lESliE_fm[1,]     = nGx*SRB/(1 + SRB)*nSx[1]
# The third step is to insert the multipliers needed to project the remaining age groups, representing the survival of the male population. We simply use the same formulas as in Exercise 1, but applied to the male data.
diag(lESliE_mm[-1,]) = nSx[2:ages]
lESliE_mm[ages,ages] = nSx[ages]
# Once the four sub‑matrices are constructed (we didn't have to modify lESliE_mf), we can concatenate them to obtain the augmented Leslie matrix for projecting both sexes simultaneously, assuming female population on top. We also need to concatenate the population vectors so that we have a single input vector for the projection, as shown:
lESliE            = rbind(cbind(lESliE_ff,lESliE_mf),cbind(lESliE_fm,lESliE_mm))
current           = rbind(matrix(female[,"nNx_f"],ages),matrix(male[,"nNx_m"],ages))

# Projecting the two sexes simultaneously:
# We can now project the two sexes using one single operation, as we did in Exercise 1. The resulting (projected) population is a (column) vector of length 2*ages, where the first half corresponds to the projected female population.
projected         = lESliE%*%current
# Let's produce a simple table of results, projecting the population 3 consecutive periods holding fertility and mortality constant (i.e., the Leslie matrix is always the same). We first retain only the relevant information (pop) and rename the current population using the reference year. The function str_c() concatenates several strings, and we want the column names to display one digit of decimal precision using the function sprintf("%.1f",year), as shown:
pop               = input[,c("age","nNx_m","nNx_f")]
colnames(pop)[colnames(pop) == "nNx_f"] <- str_c("nNx_f",sprintf("%.1f",year))
colnames(pop)[colnames(pop) == "nNx_m"] <- str_c("nNx_m",sprintf("%.1f",year))

# We then use a loop to project the population iteratively, drawing on the most recent data at each step. The loop is defined over a sequence of years starting from the reference year and ending at the final projection year, with a step length equal to the width of the age intervals (i.e., seq(year, year + 2*n, n)). At each step, we define the input vector "current" from the most recent data, and then compute the "projected" output vector by multiplying the Leslie matrix by the current population. We then extract the male and female projections, assign appropriate names, and concatenate the resulting tables to include the new set of projections, as shown:
for (t in seq(year, year + 2*n, n)) {
  current   = rbind(matrix(pop[,str_c("nNx_f",sprintf("%.1f",t))],ages),matrix(pop[,str_c("nNx_m",sprintf("%.1f",t))],ages))
  projected = lESliE%*%current                                                  #Projects the population.
  female    = matrix(projected[1:ages,],ages)                                   #Extracts female population.
  colnames(female)[1] <- str_c("nNx_f",sprintf("%.1f",t + n))                   #Gives a name to the projected female population.
  male      = matrix(projected[(1 + ages):(2*ages),],ages)                      #Extracts male population.
  colnames(male)[1] <- str_c("nNx_m",sprintf("%.1f",t + n))                     #Gives a name to the projected male population.
  pop       = cbind(pop,male,female)                                            #Concatenates the table of results.
}
print(pop)

# Compulsory Questions:
# a. Calculate the life expectancy at birth for each sex.
sum(input[,"nLx_m"])/radix
sum(input[,"nLx_f"])/radix

# b. Calculate the resulting age-sex distribution of keeping constant the fertility and mortality regimes for a long period of time (i.e., stable-equivalent age distribution, ergodic solution.).
current           = rbind(matrix(input[,"nNx_f"],ages),matrix(input[,"nNx_m"],ages))
current           = current/sum(current)
projected         = (lESliE%^%100)%*%current
projected         = projected/sum(projected)

female            = matrix(projected[1:ages,],ages)
colnames(female)[1] <- "nNx_f*"
male              = matrix(projected[(1 + ages):(2*ages),],ages)
colnames(male)[1] <- "nNx_m*"
ergodic           = cbind(input[,"age"],male,female)
print(ergodic)



# Exercise 3: Cohort Component Model of Projections (one sex, female dominant perspective, open population)
# In this exercise, we extend the one‑sex model to project a population that is open to migration. Since the underlying data remain unchanged, the setup and the definition of key parameters do not require any modification. We have already calculated the Leslie matrix for a female population that is self-reproducing, lESliE_ff and we can keep this result representing a counterfactual scenario of a closed population. The only adjustment is to import the data containing the number of net migrants for the female population and to construct a modified Leslie matrix that explicitly incorporates migration.
# The data:
lESliE_C          = lESliE_ff
female            = input[,c("age","nLx_f","nFx","nNx_f","nIx_f")]

# Construction of the modified Leslie matrix to account for migration:
# We calculate again the survivorship ratios using the nLx column.
nLx               = female[,"nLx_f"]
nSx               = c(nLx[1]/(n*radix),nLx[2:(ages-1)]/nLx[1:(ages-2)],nLx[ages]/sum(nLx[(ages-1):ages]))

# Let’s define a vector of age‑specific net migration rates by dividing the number of net migrants by the population at time t. This is not to be consider an untenable assumption, given that half of all migrants are assumed to arrive (or depart) at the beginning of the projection interval. Thus, the total number of migrants could be inferred at the exact time t.
nKx               = c(female[,"nIx_f"]/female[,"nNx_f"])
# Migrants in reproductive ages will affect the projected number of births. The next step is to modify the function nGx to incorporate specific assumptions about fertility and migration. We assume that migrants reproduce at the same rates as the domestic population, but we also need to account for the fact that migrants arrive at different times during the projection interval increasing (reducing) the risk of childbearing. The modified function nGx is calculated as follows:
nGx               = ((1 + nKx)*nFx + (1 + nKx/2)*c(nFx[2:ages]*nSx[2:ages],0))*n/2
# We follow the same approach used in the previous models. We define the Leslie matrix as a square matrix whose order is equal to the number of age groups, and all entries are initially set to zero before the matrix is populated. We then insert the multipliers related to survivorship and reproduction, taking into account that migration interacts with both components. Let’s begin with the first row, which corresponds to the reproductive contribution of this open population, as shown:
lESliE            = matrix(0,ages,ages)
lESliE[1,]        = nGx*1/(1 + SRB)*nSx[1]

# As in the previous models, we populate the first sub‑diagonal using the survivorship ratios and, now, the effect of migration. All populations decrease through mortality, but open populations may increase (or decrease less) when net migration is positive. Conversely, when the number of out‑migrants exceeds the number of in‑migrants, the force of attrition is intensified by the effect of negative net migration.
diag(lESliE[-1,]) = (1 + nKx[1:ages-1]/2)*nSx[2:ages]
# We then populate the main diagonal of the Leslie matrix, which now accounts for the effect of migrants who arrive at the end of the projection period.
diag(lESliE[])    = diag(lESliE[]) +  nKx/2
# Finally, we adjust the survivorship ratio for the open‑ended age interval in the last entry of the last column, which has always required special attention. In this case, the adjustment accounts for the effect of net migration.
lESliE[ages,ages] = lESliE[ages,ages] + (1 + nKx[ages]/2)*nSx[ages]


# Projecting a female population open to migration:
# As in the previous models, projecting a population is reduced to a single operation: multiplying a column population vector by the corresponding Leslie matrix, as shown: 
current           = female[,"nNx_f"]
projected         = lESliE%*%current

# Let’s produce a simple table of results, projecting the population for three consecutive periods while holding fertility and mortality constant (i.e., the Leslie matrix remains unchanged). To quantify the effect of migration on population dynamics, let’s project the population twice—first assuming the population is open, and then assuming it is closed to migration.  
# We can create two structures female_O and female_C, one for each scenario. They are initially identical and retain only the essential information.
female_O          = input[,c("age","nNx_f")]
colnames(female_O)[colnames(female_O) == "nNx_f"] <- str_c("nNx_",sprintf("%.1f", year))
female_C          = female_O

# We then use a loop to project the population iteratively, as we have done in previous models, using as an input the most recent data at each step. The loop is defined over a sequence of years (i.e., seq(year, year + 2*n, n)). At each step, we define the input vector "current" from the most recent data, and then compute the "projected" output vector by multiplying the Leslie matrix by the current population. We then extract the male and female projections, assign appropriate names, and concatenate the resulting tables to include the new set of projections, as shown:
for (t in seq(year, year + 2*n, n)) {
  projected = lESliE%*%female_O[,str_c("nNx_",sprintf("%.1f", t))]              #Projects the open population.
  colnames(projected)[1] <- str_c("nNx_",sprintf("%.1f", t + n))                #Gives a name to the projected data.
  female_O  = cbind(female_O,projected)                                         #Concatenates the table of results.
  
  projected = lESliE_C%*%female_C[,str_c("nNx_",sprintf("%.1f", t))]            #Projects the open population.
  colnames(projected)[1] <- str_c("nNx_",sprintf("%.1f", t + n))                #Gives a name to the projected data.
  female_C  = cbind(female_C,projected)                                         #Concatenates the table of results.
}
print(female_O)
print(female_C)


# Compulsory Questions:
# a. Can we expect life expectancy to change because of migration?
# No, unless migrants are selected to survive more or less than the domestic population. We would need a life table of migrants to assess that problem.  

# b. Can we expect the TFR to change because of migration?
# No, we are assuming migrants reproduce at the same rates, but additional data would help to relax that assumption (i.e., nFx discriminated by migrants and nationals).  

# c. Can we expect the NRR to change because of migration?
# No, but in practice, populations reproduce more or less due to migration. We would need detailed nLx and nFx to quantify the effect.  

# d. Calculate the projected number of births from 1993.0 to 1998.0 by age group of the mother and discriminating by migrants and nationals.
femaleS           = female_O[,c("age",str_c("nNx_",sprintf("%.1f", year)),str_c("nNx_",sprintf("%.1f", year + n)))]
B_O               = matrix(rowSums(femaleS[,c(2,3)])/2*n*nFx,ages,1)

femaleS           = female_C[,c("age",str_c("nNx_",sprintf("%.1f", year)),str_c("nNx_",sprintf("%.1f", year + n)))]
B_nat             = matrix(rowSums(femaleS[,c(2,3)])/2*n*nFx,ages,1)

colnames(B_nat)[1] <- str_c("B_nat[",sprintf("%.1f", year),",",sprintf("%.1f", year + n),")")
B_mig             = B_O - B_nat
colnames(B_mig)[1] <- str_c("B_mig[",sprintf("%.1f", year),",",sprintf("%.1f", year + n),")")
BirthS            = cbind(matrix(input[,"age"],ages),B_nat,B_mig)
print(BirthS)

# e. What change can we expect in the projected number of births if net migration is negative (i.e., nIx < 0)?.
# The excess of births would be negative due to outmigration, and the projected number of births would be lower than the counterfactual.  

# f. Which line of this code would you modify if migrants brought their own age-specific fertility rates?
# The line calculating nGx.  

# g. Which line of this code would you modify if migrants imported their own mortality schedule (i.e., nLx)?
# All lines depending on nSx.