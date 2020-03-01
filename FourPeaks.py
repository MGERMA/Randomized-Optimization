#!/usr/bin/env python3

import mlrose
import numpy as np
import pandas as pd
from time import time
import os
from tqdm import tqdm

from itertools import product

# FourPeak analysis

## Saving parameters
dir_="results/plot/"

results_dir='./results'
plot_dir='./data/plot'
    
if not os.path.exists(dir_):
    os.makedirs(dir_)


max_iter_list=2**np.arange(5,8)
state_size = 50


## Hill climbing

nb_restarts_list=[0,5,10,15,20,25,30,35,40,45,50]
hill_climb_best_state=[]
hill_climb_best_fit=[]
hill_climb_curve=[]
hill_climb_time=[]
threshold_param_list=range(0,10)
print("Random Hill Climbing")

for threshold, nb_restarts in tqdm(product(threshold_param_list,nb_restarts_list)):
        for iters in max_iter_list:
            fn=mlrose.FourPeaks(t_pct=threshold/10)
            state=np.random.randint(0,2,size=state_size)
            problem_fit=mlrose.DiscreteOpt(state_size, fn)
            start=time()
            best_state, best_fitness, curve = mlrose.random_hill_climb(problem_fit, 
                                                                   restarts=nb_restarts,
                                                                   max_attempts = 500, 
                                                                   max_iters=int(iters), curve=True)
            end=time()
            hill_climb_time.append(end-start)
            hill_climb_best_state.append(best_state)
            hill_climb_best_fit.append(best_fitness)
            hill_climb_curve.append(curve)

fprhc=pd.DataFrame({'Restarts':nb_restarts_list, 'best_fit':hill_climb_best_fit, 'iteration':max_iter_list, 'Time':hill_climb_time, 't_pct':threshold_param_list})

hill_climb_best_state_df=pd.DataFrame(0.0, index=range(1,51), columns=range(1,len(hill_climb_best_state)+1))

for i in range(1,len(annealing_best_state)+1):
    hill_climb_best_state_df.loc[:,i]=hill_climb_best_state[i-1]
    
fprhc.to_csv(os.path.join(results_dir,'fp50rhc.csv'))
hill_climb_best_state_df.to_csv(os.path.join(results_dir,'fp50rhcstates.csv'))

for i in range(len(hill_climb_curve)):
    hill_climb_curvedf=pd.DataFrame(hill_climb_curve[i])
    hill_climb_curvedf.to_csv(os.path.join(plot_dir,'fp50hill_climb_curve{}_{}_{}.csv'.format(nb_restarts_list[i], max_iter_list[i], threshold_param_list[i])))
  




## Genetic Algorithm
population_list=[10,50, 100, 200]
mutation_probability_list=np.linspace(0.1,1,5)
genetic_best_state_list=[]
genetic_best_fit_list=[]
genetic_curve=[]
genetic_time=[]

print("Genetic algorithm")

for threshold, pops, pr in tqdm(product(threshold_param_list,population_list,mutation_probability_list)):
    for iters in max_iter_list:
        fn=mlrose.FourPeaks(t_pct=threshold/10)
        state=np.random.randint(0,2,size=state_size)
        problem_fit=mlrose.DiscreteOpt(state_size, fn)
        
        start=time()
        best_state, best_fitness, curve = mlrose.genetic_alg(problem_fit,
                                                               mutation_prob = pr,
                                                               max_attempts = 500,
                                                               pop_size=pops,
                                                               max_iters=int(iters),
                                                               curve=True)
        end=time()
        max_iter_list.append(int(iters))
        genetic_time.append(end-start)
        genetic_best_state_list.append(best_state)
        genetic_best_fit_list.append(best_fitness)
        genetic_curve.append(curve)


fpga=pd.DataFrame({'Mutation Probability':mutation_probability_list, 'best_fit':genetic_best_fit_list, 'iteration':max_iter_list, 'Time':genetic_time,'Population Size':population_list, 't_pct':threshold_param_list})
genetic_best_state_listdf=pd.DataFrame(0.0, index=range(1,51), columns=range(len(genetic_best_state_list)))
for i in range(len(genetic_curve)):
    genetic_curvedf=pd.DataFrame(genetic_curve[i])
    genetic_curvedf.to_csv(os.path.join(plot_dir,'fp50genetic_curve{}_{}_{}_{}.csv'.format(mutation_probability_list[i], population_list[i], max_iter_list[i], threshold_param_list[i])))
    


for i in range(1,len(genetic_best_state_list)+1):
    genetic_best_state_listdf.loc[:,i]=genetic_bes[i-1]
    
fpga.to_csv(os.path.join(results_dir,'fp50ga.csv'))
genetic_besdf.to_csv(os.path.join(results_dir,'fp50gastates.csv'))
   
    
print("Simulated Annealing")

annealing_best_state=[]
annealing_best_fit=[]
annealing_curve=[]
annealingtime=[]
decay_list = [0.9, 0.93, 0.96, 0.98,  0.99]

for threshold in tqdm(threshold_param_list):
    for iters in max_iter_list:
        for decay in decay_list:
            fn=mlrose.FourPeaks(t_pct=threshold/10)
            state=np.random.randint(0,2,size=state_size)
            problem_fit=mlrose.DiscreteOpt(state_size, fn)
            start=time()
            best_state, best_fitness, curve = mlrose.simulated_annealing(problem_fit,
                                                                           max_attempts = 500, 
                                                                           max_iters=int(iters),
                                                                           curve=True, 
                                                                           schedule=mlrose.GeomDecay(init_temp=1.0, decay=decay, min_temp=0.001))
           
            end=time()
            annealingtime.append(end-start)
            annealing_best_state.append(best_state)
            annealing_best_fit.append(best_fitness)
            annealing_curve.append(curve)

fpsa=pd.DataFrame({ 'best_fit':annealing_best_fit, 'iteration':max_iter_list, 'Time':annealingtime, 'decay':decay_list, 't_pct':threshold_param_list})
annealing_best_statedf=pd.DataFrame(0.0, index=range(1,51), columns=range(len(annealing_best_state)))
for i in range(len(annealing_curve)):
    annealing_curvedf=pd.DataFrame(annealing_curve[i])
    annealing_curvedf.to_csv(os.path.join(plot_dir,'fp50annealing_curve_{}_{}_{}.csv'.format( max_iter_list[i], CEl[i], threshold_param_list[i])))
    


for i in range(1,len(annealing_best_state)+1):
    annealing_best_statedf.loc[:,i]=annealing_best_state[i-1]
    
fpsa.to_csv(os.path.join(results_dir,'fp50sa.csv'))
annealing_best_statedf.to_csv(os.path.join(results_dir,'fp50sastates.csv'))


# MIMIC Algorithm

keep_list=np.linspace(0.1,1,10)
mimic_best_state=[]
mimic_best_fit=[]
mimic_curve=[]
mimic_time=[]
print("Mimic")

for threshold, keep_param, iters in tqdm(product(threshold_param_list,keep_list,max_iter_list)):
            fn=mlrose.FourPeaks(t_pct=threshold/10)
            state=np.random.randint(0,2,size=state_size)
            problem_fit=mlrose.DiscreteOpt(state_size, fn)
            start=time()
            best_state, best_fitness, curve = mlrose.mimic(problem_fit, 
                                                                   keep_pct=keep_param,
                                                                   max_attempts = 500, 
                                                                   max_iters=int(iters), curve=True)
            end=time()
            mimic_time.append(end-start)
            mimic_best_state.append(best_state)
            mimic_best_fit.append(best_fitness)
            mimic_curve.append(curve)    

fpmi=pd.DataFrame({'Keep percentage':keep_list, 'best_fit':mimic_best_fit, 'iteration':max_iter_list, 'Time':mimic_time,'t_pct': threshold_param_list})
mimic_best_statedf=pd.DataFrame(0.0, index=range(1,51), columns=range(1,len(mimic_best_state)+1))

for i in range(1,len(mimic_best_state)+1):
    mimic_best_statedf.loc[:,i]=mimic_best_state[i-1]
    
fpmi.to_csv(os.path.join(results_dir,'fp50mi.csv'))
mimic_best_statedf.to_csv(os.path.join(results_dir,'fp50mistates.csv'))

for i in range(len(mimic_curve)):
    mimic_curvedf=pd.DataFrame(mimic_curve[i])
    mimic_curvedf.to_csv(os.path.join(plot_dir,'fp50mimic_curve{}_{}_{}.csv'.format(keep_list[i], max_iter_list[i],threshold_param_list[i])))   

