#!/usr/bin/env python3

import mlrose
import numpy as np
import pandas as pd
from time import time
import os
from tqdm import tqdm

from itertools import product

dir_="results/plot/"
if not os.path.exists(dir_):
    os.makedirs(dir_)
results_dir='./results'
plot_dir='./data/plot'

iterlist=2**np.arange(5,10)
population_list=[]
mutation_probability_list=[]
genetic_best_state_list=[]
genetic_best_fit_list=[]
genetic_curve=[]
genetic_time=[]
index_list=[]
threshold_param_list=[]

print("Genetic algorithm")

# Solve problem using the genetic algorithm
for tp, pops, pr in tqdm(product(range(0,10),[10,20,30,50,75, 100, 150,200],np.linspace(0.1,1,10))):
    for iters in iterlist:
        np.random.seed(123)
        fn=mlrose.FourPeaks(t_pct=tp/10)
        state=np.random.randint(0,2,size=50)
        problem_fit=mlrose.DiscreteOpt(50, fn)
        
        start=time()
        best_state, best_fitness, curve = mlrose.genetic_alg(problem_fit, \
                                                               mutation_prob = pr,\
                                                               max_attempts = 500, \
                                                               pop_size=pops,
                                                               max_iters=int(iters), curve=True, random_state=123)
        end=time()
        index_list.append(int(iters))
        genetic_time.append(end-start)
        genetic_best_state_list.append(best_state)
        genetic_best_fit_list.append(best_fitness)
        mutation_probability_list.append(pr)
        genetic_curve.append(curve)
        population_list.append(pops)
        threshold_param_list.append(tp)


fpga=pd.DataFrame({'Mutation Probability':mutation_probability_list, 'best_fit':genetic_best_fit_list, 'iteration':index_list, 'Time':genetic_time,'Population Size':population_list, 't_pct':threshold_param_list})
genetic_best_state_listdf=pd.DataFrame(0.0, index=range(1,51), columns=range(len(genetic_best_state_list)))
for i in range(len(genetic_curve)):
    genetic_curvedf=pd.DataFrame(genetic_curve[i])
    genetic_curvedf.to_csv(os.path.join(plot_dir,'fp50genetic_curve{}_{}_{}_{}.csv'.format(mutation_probability_list[i], population_list[i], index_list[i], threshold_param_list[i])))
    


for i in range(1,len(genetic_best_state_list)+1):
    genetic_best_state_listdf.loc[:,i]=genetic_bes[i-1]
    
fpga.to_csv(os.path.join(results_dir,'fp50ga.csv'))
genetic_besdf.to_csv(os.path.join(results_dir,'fp50gastates.csv'))
   
    
print("Simulated Annealing")

annealing_best_state=[]
annealing_best_fit=[]
annealing_curve=[]
annealingtime=[]
index_list=[]
decay_list = [0.9, 0.93, 0.96, 0.98,  0.99]
threshold_param_list=[]

for tp in tqdm(range(0,10)):
    for iters in iterlist:
        for decay in decay_list:
            threshold_param_list.append(tp)
            fn=mlrose.FourPeaks(t_pct=tp/10)
            np.random.seed(123)
            state=np.random.randint(0,2,size=50)
            problem_fit=mlrose.DiscreteOpt(50, fn)
            start=time()
            best_state, best_fitness, curve = mlrose.simulated_annealing(problem_fit,\
                                                                           max_attempts = 500, \
                                                                           max_iters=int(iters),\
                                                                           curve=True, \
                                                                           schedule=mlrose.GeomDecay(init_temp=1.0, decay=decay, min_temp=0.001),
                                                                           random_state=123)
           
            end=time()
            annealingtime.append(end-start)
            annealing_best_state.append(best_state)
            annealing_best_fit.append(best_fitness)
            annealing_curve.append(curve)
            index_list.append(int(iters))
            print(tp)
            print(CE)
            print(int(iters))
            print(best_state)
            print(best_fitness)

fpsa=pd.DataFrame({ 'best_fit':annealing_best_fit, 'iteration':index_list, 'Time':annealingtime, 'decay':decay_list, 't_pct':threshold_param_list})
annealing_best_statedf=pd.DataFrame(0.0, index=range(1,51), columns=range(len(annealing_best_state)))
for i in range(len(annealing_curve)):
    annealing_curvedf=pd.DataFrame(annealing_curve[i])
    annealing_curvedf.to_csv(os.path.join(plot_dir,'fp50annealing_curve_{}_{}_{}.csv'.format( index_list[i], CEl[i], threshold_param_list[i])))
    


for i in range(1,len(annealing_best_state)+1):
    annealing_best_statedf.loc[:,i]=annealing_best_state[i-1]
    
fpsa.to_csv(os.path.join(results_dir,'fp50sa.csv'))
annealing_best_statedf.to_csv(os.path.join(results_dir,'fp50sastates.csv'))

nb_restarts_list=[0,5,10,15,20,25,30,35,40,45,50]
hill_climb_best_state=[]
hill_climb_best_fit=[]
hill_climb_curve=[]
hill_climb_time=[]
index_list=[]
threshold_param_list=[]
print("Random Hill Climbing")

for tp, rs in tqdm(product(range(0,10),nb_restarts_list)):
        for iters in iterlist:
            index_list.append(int(iters))
            nb_restarts_list.append(rs)
            threshold_param_list.append(tp)
            fn=mlrose.FourPeaks(t_pct=tp)
            np.random.seed(123)
            state=np.random.randint(0,2,size=50)
            problem_fit=mlrose.DiscreteOpt(50, fn)
            start=time()
            best_state, best_fitness, curve = mlrose.random_hill_climb(problem_fit, \
                                                                   restarts=rs,\
                                                                   max_attempts = 500, \
                                                                   max_iters=int(iters), curve=True, random_state=123)
            end=time()
            hill_climb_time.append(end-start)
            hill_climb_best_state.append(best_state)
            hill_climb_best_fit.append(best_fitness)
            hill_climb_curve.append(curve)
            print(int(iters))
            print(rs)
            print(best_state)
            print(best_fitness)

fprhc=pd.DataFrame({'Restarts':nb_restarts_list, 'best_fit':hill_climb_best_fit, 'iteration':index_list, 'Time':hill_climb_time, 't_pct':threshold_param_list})

hill_climb_best_state_df=pd.DataFrame(0.0, index=range(1,51), columns=range(1,len(hill_climb_best_state)+1))

for i in range(1,len(annealing_best_state)+1):
    hill_climb_best_state_df.loc[:,i]=hill_climb_best_state[i-1]
    
fprhc.to_csv(os.path.join(results_dir,'fp50rhc.csv'))
hill_climb_best_state_df.to_csv(os.path.join(results_dir,'fp50rhcstates.csv'))

for i in range(len(hill_climb_curve)):
    hill_climb_curvedf=pd.DataFrame(hill_climb_curve[i])
    hill_climb_curvedf.to_csv(os.path.join(plot_dir,'fp50hill_climb_curve{}_{}_{}.csv'.format(nb_restarts_list[i], index_list[i], threshold_param_list[i])))
    

kpl=[]
mibeststate=[]
mibestfit=[]
micurve=[]
mitime=[]
index_list=[]
threshold_param_list=[]
print("Mimic")

for tp in np.linspace(0.1,1,10):
    for kp in np.linspace(0.1,1,10):
        for iters in iterlist:
            threshold_param_list.append(tp)
            index_list.append(int(iters))
            np.random.seed(123)
            fn=mlrose.FourPeaks(t_pct=tp)
            state=np.random.randint(0,2,size=50)
            problem_fit=mlrose.DiscreteOpt(50, fn)
            start=time()
            best_state, best_fitness, curve = mlrose.mimic(problem_fit, \
                                                                   keep_pct=kp,\
                                                                   max_attempts = 500, \
                                                                   max_iters=int(iters), curve=True, random_state=123)
            end=time()
            mitime.append(end-start)
            mibeststate.append(best_state)
            mibestfit.append(best_fitness)
            kpl.append(kp)
            micurve.append(curve)    

fpmi=pd.DataFrame({'Keep percentage':kpl, 'best_fit':mibestfit, 'iteration':index_list, 'Time':mitime,'t_pct': threshold_param_list})
mibeststatedf=pd.DataFrame(0.0, index=range(1,51), columns=range(1,len(mibeststate)+1))

for i in range(1,len(mibeststate)+1):
    mibeststatedf.loc[:,i]=mibeststate[i-1]
    
fpmi.to_csv(os.path.join(results_dir,'fp50mi.csv'))
mibeststatedf.to_csv(os.path.join(results_dir,'fp50mistates.csv'))

for i in range(len(micurve)):
    micurvedf=pd.DataFrame(micurve[i])
    micurvedf.to_csv(os.path.join(plot_dir,'fp50micurve{}_{}_{}.csv'.format(kpl[i], index_list[i],threshold_param_list[i])))   

