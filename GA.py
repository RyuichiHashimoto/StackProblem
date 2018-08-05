import numpy as np
import pandas as pd
from copy import deepcopy
import numpy.random as random; 
from abc import ABCMeta, abstractmethod
import os;
import StackProblem
from time import time;
import sys
## This class assume a discrete problem (not permutation problem like a TSP problem). 
## if you modify this class for numerical problem, i recommend you add upper/lower limit of each variables.
class Solution():
    def __init__(self,nVal,div,nObj=1):
        self.nOfObjective = nObj;
        self.nOfVariables = nVal;
        self.division = div;
        self.objectives = [nObj];
        self.variables = [nVal];

    def getDivision(self):
        return self.division;

  
    def getNofVal(self):
        return self.nOfVariables;
    
    def getNofObj(self):
        return self.nOfObjective;

    def getFitness(self):
        return self.objectives[0];

    def getVariables(self):
        return self.variables;

    ## if you use
    def __lt__(self, other):
        return self.objectives[0] > other.objectives[0];


## this is for simple modification.
##
class Problem():
    __metaclass__ = ABCMeta 
    
    def __init__(self,nOfVal,nOfObj,trial_ = 0):
        self.nOfObjective = nOfObj;
        self.nOfVariables = nOfVal;
        self.division = [0]*nOfVal;
        self.trial =trial_;

    def getNofObjective(self):
        return self.nOfObjective;

    def getNofVariables(self):
        return self.nOfVariables;

    def getDivision(self):
        return self.division;

    @abstractmethod 
    def evaluate(self,solution):
        print("abstract")
        pass

    @abstractmethod
    def testTrial(self,solution):
        print("abstract")
        pass

    @abstractmethod 
    def initialize(self,initialize):
        print("abstract")
        pass


## This is a confication of my impelmentation.
class onemaxProblem(Problem):

    def __init__(self,nOfVal):
        super().__init__(nOfVal,1);  
        self.division = [2]*self.nOfVariables;

       
    def evaluate(self,solution:Solution):
        sum = np.sum(solution.getVariables());
        solution.objectives = np.array([sum]);

    def initialize(self,solution:Solution):       
        solution.variables = np.array([random.randint(0,solution.getDivision()[i]) for i in range(0,self.nOfVariables)]);        

    @abstractmethod
    def testTrial(self,solution):
        print("need not to implement");


def crossover(crossoverProbability,parent1:Solution,parent2:Solution):
    offspring = [];


    offspring.append(deepcopy(parent1));
    offspring.append(deepcopy(parent2));

    if(random.rand()<crossoverProbability):
        for i in range(0,parent1.getNofVal()):            
            if(random.rand()<0.5):
                offspring[0].variables[i] = parent2.variables[i]
                offspring[1].variables[i] = parent1.variables[i]

    return offspring;

def mutation(mutationProbability,sol:Solution):
    for i in range(0,sol.getNofVal()):
        if(random.rand() < mutationProbability):
            sol.variables[i] = random.randint(0,sol.division[i]);            

    

def parentsSelection(population:[]):
    parentsNumber = [];
    parentsNumber.append(random.randint(len(population)));
    parentsNumber.append(random.randint(len(population)));
    return parentsNumber;

def GA(problem:Problem):
    generation = 500;
    populationSize = 100;
    if populationSize %2 == 1 :   populationSize = populationSize+1; ## 今回は個体群サイズを偶数に設定しないとエラーを吐き出すよう実装しているため．
    
    
    start = time();
    
    population = initialize(populationSize,problem);
    bestTrainFitness  = []
    bestTestFitness = []
    bestTrainFitness.append([1,population[0].objectives[1]])
    bestTestFitness.append([1, problem.testTrial(population[0])[1]])
    
    gen = 1;    
    print(str(gen)+ "\tgen : Best Train Fitness " + str(population[0].getFitness())+"\t Best Test Fitness " +  str(bestTestFitness[-1][1]) + "\telapsed_time:{:.4f}".format(time()-start) + "\t[sec]");    

    while gen < generation:
        start = time();
        offspring = generateOffspring(population,problem);
        population = environmentalSelection(population,offspring);
        bestTrainFitness.append([gen+1,population[0].objectives[1]])
        bestTestFitness.append([gen + 1, problem.testTrial(population[0])[1]])
        gen = gen+1;
        if(gen %1 == 0): print(str(gen)+ "\tgen : Best Train Fitness " + str(population[0].getFitness())+"\t Best Test Fitness " +  str(bestTestFitness[-1][1]) + "\telapsed_time:{:.4f}".format(time()-start) + "\t[sec]");



    np.savetxt("result/Train/bestFitness_"+ str(problem.trial+1)+".dat",bestTrainFitness,delimiter= "\t");
    np.savetxt("result/Test/bestFitness_" + str(problem.trial + 1) + ".dat", bestTestFitness, delimiter="\t");
    return population;
        
def generateOffspring(population,problem):
    offspring =[];
    for i in range(0,int(len(population)/2)):
        parentsNumber = parentsSelection(population);
        children = crossover(1.0,population[parentsNumber[0]],population[parentsNumber[1]]);
    
        mutation(4.0/children[0].getNofVal(),children[0]);
        mutation(4.0/children[1].getNofVal(),children[1]);
        problem.evaluate(children[0]);
        problem.evaluate(children[1]);

        offspring.extend(children);

    return offspring;

def environmentalSelection(population,offspring):
    unionpopulation = [];
    unionpopulation.extend(population);
    unionpopulation.extend(offspring);
#    nondominatedSolution(unionpopulation);
    unionpopulation=sorted(unionpopulation);
    return unionpopulation[0:len(population)];

def initialize(populationSize:int,problem:Problem):
    population =  [ (Solution(problem.getNofVariables(),problem.getDivision(),problem.getNofObjective()))  for i in range(0,populationSize) ];
    for sol in population:
        problem.initialize(sol);
        problem.evaluate(sol);   

    return sorted(population)
            

    
if __name__ == "__main__":

    #problem = onemaxProblem(10); this is for debag
    os.makedirs("result/Train",exist_ok=True);
    os.makedirs("result/Test", exist_ok=True);
    df = pd.read_csv("Data/SonyData.csv");
    
    trial = int(sys.argv[1]);    
    print(str(trial + 1) + "th	start");
    
    random.seed(trial + 10022);
    problem = StackProblem.stackTradeProblem(df,1000000,trial);
    pop = GA(problem);
    problem.testTrial(pop[0]);
    print("benefit:"+str(pop[0].objectives[0]));





