import time
start_time = time.time()

import math
import random
import numpy as np
from matplotlib import pyplot as plt

class SolveTSPUsingACO:
    class Edge1:  
        def __init__(self, a, b, weight1, initial_pheromone1):
            self.a = a
            self.b = b 
            self.weight1 = weight1  
            self.pheromone1 = initial_pheromone1 

    class Ant1:
        def __init__(self, alpha1, beta1, num_nodes1, edges1):
            self.alpha1 = alpha1
            self.beta1 = beta1
            self.num_nodes1 = num_nodes1 
            self.edges1 = edges1 
            self.tour1 = None  
            self.distance = 0.0  

        def _select_node1(self):   
            roulette_wheel1 = 0.0  
            unvisited_nodes1 = [node for node in range(self.num_nodes1) if node not in self.tour1]  
            heuristic_total = 0.0  
            for unvisited_node in unvisited_nodes1:
                heuristic_total += self.edges1[self.tour1[-1]][unvisited_node].weight1  
            for unvisited_node in unvisited_nodes1:
                roulette_wheel1 += pow(self.edges1[self.tour1[-1]][unvisited_node].pheromone1, self.alpha1) * pow((heuristic_total / self.edges1[self.tour1[-1]][unvisited_node].weight1), self.beta1)

            random_value = random.uniform(0.0, roulette_wheel1)  
            wheel_position = 0.0
            for unvisited_node in unvisited_nodes1:
                wheel_position += pow(self.edges1[self.tour1[-1]][unvisited_node].pheromone1, self.alpha1) * pow((heuristic_total / self.edges1[self.tour1[-1]][unvisited_node].weight1), self.beta1)

                if wheel_position >= random_value:
                    return unvisited_node

        def find_tour1(self):
            self.tour1 = [random.randint(0, self.num_nodes1 - 1)]
            while len(self.tour1) < self.num_nodes1:
                self.tour1.append(self._select_node1())  
            return self.tour1

        def get_distance1(self):
            self.distance = 0.0
            for i in range(self.num_nodes1):
                self.distance += self.edges1[self.tour1[i]][self.tour1[(i + 1) % self.num_nodes1]].weight1
            return self.distance  

    class Edge2:  
        def __init__(self, a, b, weight2, initial_pheromone2):
            self.a = a
            self.b = b  
            self.weight2 = weight2  
            self.pheromone2 = initial_pheromone2  

    class Ant2:
        def __init__(self, alpha2, beta2, num_nodes2, edges2):
            self.alpha2 = alpha2
            self.beta2 = beta2
            self.num_nodes2 = num_nodes2 
            self.edges2 = edges2 
            self.tour2 = None  
            self.distance = 0.0  

        def _select_node2(self):   
            roulette_wheel2 = 0.0  
            unvisited_nodes2 = [node for node in range(self.num_nodes2) if node not in self.tour2]  
            heuristic_total = 0.0  
            for unvisited_node in unvisited_nodes2:
                heuristic_total += self.edges2[self.tour2[-1]][unvisited_node].weight2 
            for unvisited_node in unvisited_nodes2:
                roulette_wheel2 += pow(self.edges2[self.tour2[-1]][unvisited_node].pheromone2, self.alpha2) * pow((heuristic_total / self.edges2[self.tour2[-1]][unvisited_node].weight2), self.beta2)

            random_value = random.uniform(0.0, roulette_wheel2)  
            wheel_position = 0.0
            for unvisited_node in unvisited_nodes2:
                wheel_position += pow(self.edges2[self.tour2[-1]][unvisited_node].pheromone2, self.alpha2) * pow((heuristic_total / self.edges2[self.tour2[-1]][unvisited_node].weight2), self.beta2)

                if wheel_position >= random_value:
                    return unvisited_node

        def find_tour2(self):
            self.tour2 = [random.randint(0, self.num_nodes2 - 1)]
            while len(self.tour2) < self.num_nodes2:
                self.tour2.append(self._select_node2())  
            return self.tour2

        def get_distance2(self):
            self.distance = 0.0
            for i in range(self.num_nodes2):
                self.distance += self.edges2[self.tour2[i]][self.tour2[(i + 1) % self.num_nodes2]].weight2
            return self.distance  

    def __init__(self, colony_size1=50, elitist_weight1=1.0, min_scaling_factor1=0.001, alpha1=1.0, beta1=3.0,
                 rho1=0.1, pheromone_deposit_weight1=1.0, initial_pheromone1=1.0, nodes1=None, labels1=None,
                 colony_size2=50, elitist_weight2=1.0, min_scaling_factor2=0.001, alpha2=1.0, beta2=3.0,
                 rho2=0.1, pheromone_deposit_weight2=1.0, initial_pheromone2=1.0, nodes2=None, labels2=None, steps=500): 

        self.colony_size1 = colony_size1
        self.elitist_weight1 = elitist_weight1
        self.min_scaling_factor1 = min_scaling_factor1
        self.rho1 = rho1
        self.pheromone_deposit_weight1 = pheromone_deposit_weight1  
        self.steps = steps
        self.num_nodes1 = len(nodes1)
        self.nodes1 = nodes1
        if labels1 is not None:
            self.labels1 = labels1
        else:
            self.labels1 = range(1, self.num_nodes1 + 1)
        self.edges1 = [[None] * self.num_nodes1 for _ in range(self.num_nodes1)]
        for i in range(self.num_nodes1):
            for j in range(i + 1, self.num_nodes1):
                self.edges1[i][j] = self.edges1[j][i] = self.Edge1(i, j, math.sqrt(pow(self.nodes1[i][0] - self.nodes1[j][0], 2.0) + pow(self.nodes1[i][1] - self.nodes1[j][1], 2.0)),initial_pheromone1)
        self.ants1 = [self.Ant1(alpha1, beta1, self.num_nodes1, self.edges1) for _ in range(self.colony_size1)]
        self.global_best_tour1 = None
        self.global_best_distance1 = float("inf")


        self.colony_size2 = colony_size2
        self.elitist_weight2 = elitist_weight2
        self.min_scaling_factor2 = min_scaling_factor2
        self.rho2 = rho2
        self.pheromone_deposit_weight2 = pheromone_deposit_weight2  
        self.steps = steps
        self.num_nodes2 = len(nodes2)
        self.nodes2 = nodes2
        if labels2 is not None:
            self.labels2 = labels2
        else:
            self.labels2 = range(1, self.num_nodes2 + 1)
        self.edges2 = [[None] * self.num_nodes2 for _ in range(self.num_nodes2)]
        for i in range(self.num_nodes2):
            for j in range(i + 1, self.num_nodes2):
                self.edges2[i][j] = self.edges2[j][i] = self.Edge2(i, j, math.sqrt(pow(self.nodes2[i][0] - self.nodes2[j][0], 2.0) + pow(self.nodes2[i][1] - self.nodes2[j][1], 2.0)),initial_pheromone2)
        self.ants2 = [self.Ant2(alpha2, beta2, self.num_nodes2, self.edges2) for _ in range(self.colony_size2)]
        self.global_best_tour2 = None
        self.global_best_distance2 = float("inf")

    def _add_pheromone1(self, tour1, distance, weight1=1.0):  
        pheromone_to_add1 = self.pheromone_deposit_weight1 / distance 
        for i in range(self.num_nodes1):
            self.edges1[tour1[i]][tour1[(i + 1) % self.num_nodes1]].pheromone1 += weight1 * pheromone_to_add1 

    def _add_pheromone2(self, tour2, distance, weight2=1.0):  
        pheromone_to_add2 = self.pheromone_deposit_weight2 / distance  
        for i in range(self.num_nodes2):
            self.edges2[tour2[i]][tour2[(i + 1) % self.num_nodes2]].pheromone2 += weight2 * pheromone_to_add2 

    def mutate_list(self, lst):
        # 
        index1 = random.randint(0, len(lst) - 1)
        index2 = random.randint(0, len(lst) - 1)
        lst[index1], lst[index2] = lst[index2], lst[index1]
        return lst

   
    def order_crossover(self, parent1, parent2): 
       
        start = random.randint(0, len(parent1) - 2)
        end = random.randint(start + 1, len(parent1) - 1)
       
        child1 = parent1[start:end + 1]
     
        for city in parent2:
            if city not in child1:
                child1.append(city)
       
        return child1

   
    def order_crossover_low(self, parent1, parent2):  
       
        start = random.randint(0, len(parent1) - 2)
        end = random.randint(start + 1, len(parent1) - 1)

       
        child1 = [None] * len(parent1)
        child1[start:end + 1] = parent1[start:end + 1]

        
        list1 = []
        for city in parent2:
            if city not in child1 and city in parent1:
                list1.append(city)

        list1_index = 0
        for i in range(len(child1)):
            if child1[i] == None:
                child1[i] = list1[list1_index]
                list1_index += 1

       
        return child1

    
    def order_crossover_high(self, parent1, parent2):
        
        new_list = parent2.copy()
       
        list1 = []
        list1_index = 0
        for city in parent2:
            if city in parent1 and city in parent2:
                list1.append(city)
                new_list[list1_index] = None
            list1_index += 1
        child1_son = self.order_crossover_low(parent1, list1)
        list2_index = 0
        for i in range(len(new_list)):
            if new_list[i] == None:
                new_list[i] = child1_son[list2_index]
                list2_index += 1
       
        return new_list

    def get_distance1_GA(self, route_number):
        
        with open('eil51-426.txt', 'r') as file:
            cities = [list(map(float, line.split())) for line in file]

        
        num_cities = len(cities)
        distances = [[0.0] * num_cities for i in range(num_cities)]
        for i in range(num_cities):
            for j in range(num_cities):
                if i != j:
                    x1, y1 = cities[i][1], cities[i][2]
                    x2, y2 = cities[j][1], cities[j][2]
                    distances[i][j] = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

        
        total_distance = 0.0
        for i in range(len(route_number) - 1):
            start_city = route_number[i]
            end_city = route_number[i + 1]
            total_distance += distances[start_city][end_city]

       
        total_distance += distances[route_number[-1]][route_number[0]]

        return total_distance

    def get_distance2_GA(self, route_number):
        
        with open('berlin52-7542.txt', 'r') as file:
            cities = [list(map(float, line.split())) for line in file]

       
        num_cities = len(cities)
        distances = [[0.0] * num_cities for i in range(num_cities)]
        for i in range(num_cities):
            for j in range(num_cities):
                if i != j:
                    x1, y1 = cities[i][1], cities[i][2]
                    x2, y2 = cities[j][1], cities[j][2]
                    distances[i][j] = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

       
        total_distance = 0.0
        for i in range(len(route_number) - 1):
            start_city = route_number[i]
            end_city = route_number[i + 1]
            total_distance += distances[start_city][end_city]

       
        total_distance += distances[route_number[-1]][route_number[0]]

        return total_distance

    def SA_1(self, initial_solution1, initial_solution2):
        max_iter = 50
        temperature = 100
        
        current_solution1 = initial_solution1
        best_solution1 = initial_solution1
        
        for i in range(max_iter):
            
            neighbor_solution = self.order_crossover_low(initial_solution1, initial_solution2)
            
            delta = self.get_distance1_GA(neighbor_solution) - self.get_distance1_GA(current_solution1)
            
            if delta < 0:
                current_solution = neighbor_solution
                
                if self.get_distance1_GA(current_solution1) < self.get_distance1_GA(best_solution1):
                    best_solution1 = current_solution
           
            else:
                prob = math.exp(-delta / temperature)
                if random.random() < prob:
                    current_solution1 = neighbor_solution
            """
            if random.random() < 0.1:  # 变异
                neighbor_solution_ = self.mutate_list(best_solution1)
                alpha = self.get_distance1_GA(neighbor_solution_) - self.get_distance1_GA(best_solution1)
                if alpha < 0:
                    best_solution1 = neighbor_solution_
            """
            
            temperature *= 0.99
       
        return best_solution1

    def SA_2(self, initial_solution1, initial_solution2):
        max_iter = 100
        temperature = 100
       
        current_solution2 = initial_solution2
        best_solution2 = initial_solution2
       
        for i in range(max_iter):
           
            neighbor_solution = self.order_crossover_high(initial_solution1, initial_solution2)
            
            delta = self.get_distance2_GA(neighbor_solution) - self.get_distance2_GA(current_solution2)
            
            if delta < 0:
                current_solution = neighbor_solution
                
                if self.get_distance2_GA(current_solution2) < self.get_distance2_GA(best_solution2):
                    best_solution2 = current_solution
            
            else:
                prob = math.exp(-delta / temperature)
                if random.random() < prob:
                    current_solution2 = neighbor_solution
            """
            if random.random() < 0.1:  # 变异
                neighbor_solution_ = self.mutate_list(best_solution2)
                alpha = self.get_distance2_GA(neighbor_solution_) - self.get_distance2_GA(best_solution2)
                if alpha < 0:
                    best_solution2 = neighbor_solution_
            """
           
            temperature *= 0.99
        
        return best_solution2

    def _ACGA(self):
        iteration_list1 = []
        iteration_list2 = []
        variance1_list = []
        variance2_list = []
        for step in range(self.steps):
            if self.global_best_distance2 <= 7544.5:
                break
            tour1_list = []  
            tour2_list = []
            ant1_distance_list = []  
            ant2_distance_list = []
            iteration_best_tour1 = None
            iteration_best_distance1 = float("inf")
            variance1 = 0
            iteration_best_tour2 = None
            iteration_best_distance2 = float("inf")
            variance2 = 0
            for ant1 in self.ants1:
                ant1.find_tour1()
                tour1_list.append(ant1.tour1)
                ant1_distance_list.append(ant1.distance)
                # print(ant1.tour1)
                if ant1.get_distance1() < iteration_best_distance1: 
                    iteration_best_tour1 = ant1.tour1
                    iteration_best_distance1 = ant1.distance
            variance1 = np.var(ant1_distance_list)  
            variance1_list.append(variance1)
            for ant2 in self.ants2:
                ant2.find_tour2()
                tour2_list.append(ant2.tour2)
                ant2_distance_list.append(ant2.distance)
               
                if ant2.get_distance2() < iteration_best_distance2: 
                    iteration_best_tour2 = ant2.tour2
                    iteration_best_distance2 = ant2.distance
            variance2 = np.var(ant2_distance_list)  
            variance2_list.append(variance2)
            # print(sum(variance1_list) / len(variance1_list), sum(variance2_list) / len(variance2_list))
           
            if float(step + 1) / float(self.steps) <= 0.5:
            #if sum(variance1_list) / len(variance1_list) > 2100 or step == 0:
                self._add_pheromone1(iteration_best_tour1, iteration_best_distance1)
                max_pheromone1 = self.pheromone_deposit_weight1 / iteration_best_distance1  
            else:
               
                """
                for i in range(_colony_size1): 
                    child1 = self.SA_1(tour1_list[i], tour2_list[i])
                    child1_distance = self.get_distance1_GA(child1)
                    if child1_distance <= iteration_best_distance1:
                        iteration_best_distance1 = child1_distance
                        iteration_best_tour1 = child1
                        # print("1交互")
                """
                if variance1 <= sum(variance1_list) / len(variance1_list):
                   
                    child1 = self.SA_1(iteration_best_tour1, iteration_best_tour2)
                    child1_distance = self.get_distance1_GA(child1)
                    if child1_distance <= iteration_best_distance1:
                        iteration_best_distance1 = child1_distance
                        iteration_best_tour1 = child1
                       

                    if iteration_best_distance1 < self.global_best_distance1:
                        self.global_best_tour1 = iteration_best_tour1
                        self.global_best_distance1 = iteration_best_distance1
                else:
                    self._add_pheromone1(iteration_best_tour1, iteration_best_distance1)
                    max_pheromone1 = self.pheromone_deposit_weight1 / iteration_best_distance1  
                self._add_pheromone1(self.global_best_tour1, self.global_best_distance1)
                max_pheromone1 = self.pheromone_deposit_weight1 / self.global_best_distance1  
            min_pheromone1 = max_pheromone1 * self.min_scaling_factor1 
            for i in range(self.num_nodes1):
                for j in range(i + 1, self.num_nodes1):
                    self.edges1[i][j].pheromone1 *= (1.0 - self.rho1)
                    if self.edges1[i][j].pheromone1 > max_pheromone1:
                        self.edges1[i][j].pheromone1 = max_pheromone1
                    elif self.edges1[i][j].pheromone1 < min_pheromone1:
                        self.edges1[i][j].pheromone1 = min_pheromone1

            if float(step + 1) / float(self.steps) <= 0.5:
            #if sum(variance2_list) / len(variance2_list) > 2800 or step == 0:
                self._add_pheromone2(iteration_best_tour2, iteration_best_distance2)
                max_pheromone2 = self.pheromone_deposit_weight2 / iteration_best_distance2  
            else:
                # SGA退火交互
                '''
                for i in range(_colony_size2):
                    child2 = self.SA_2(tour1_list[i], tour2_list[i])
                    child2_distance = self.get_distance2_GA(child2)
                    if child2_distance <= iteration_best_distance2:
                        iteration_best_distance2 = child2_distance
                        iteration_best_tour2 = child2
                        # print("2交互")
                '''
                if variance2 <= sum(variance2_list) / len(variance2_list):
                    child2 = self.SA_2(iteration_best_tour1, iteration_best_tour2)
                    child2_distance = self.get_distance2_GA(child2)
                    if child2_distance <= iteration_best_distance2:
                        iteration_best_distance2 = child2_distance
                        iteration_best_tour2 = child2

                    if iteration_best_distance2 < self.global_best_distance2:
                        self.global_best_tour2 = iteration_best_tour2
                        self.global_best_distance2 = iteration_best_distance2
                    self._add_pheromone2(self.global_best_tour2, self.global_best_distance2)
                    max_pheromone2 = self.pheromone_deposit_weight2 / self.global_best_distance2  
                else:
                    self._add_pheromone2(iteration_best_tour2, iteration_best_distance2)
                    max_pheromone2 = self.pheromone_deposit_weight2 / iteration_best_distance2  
            min_pheromone2 = max_pheromone2 * self.min_scaling_factor2 
            for i in range(self.num_nodes2):
                for j in range(i + 1, self.num_nodes2):
                    self.edges2[i][j].pheromone2 *= (1.0 - self.rho2)
                    if self.edges2[i][j].pheromone2 > max_pheromone2:
                        self.edges2[i][j].pheromone2 = max_pheromone2
                    elif self.edges2[i][j].pheromone2 < min_pheromone2:
                        self.edges2[i][j].pheromone2 = min_pheromone2

            if self.global_best_distance1 > iteration_best_distance1:
                self.global_best_distance1 = iteration_best_distance1
                self.global_best_tour1 = iteration_best_tour1
            if self.global_best_distance2 > iteration_best_distance2:
                self.global_best_distance2 = iteration_best_distance2
                self.global_best_tour2 = iteration_best_tour2

            iteration_list1.append(self.global_best_distance1)
            iteration_list2.append(self.global_best_distance2)

            print("iter:{},（IB1）:{},（GB1）:{},（IB2）:{},（GB2）:{}".format(step, iteration_best_distance1, self.global_best_distance1, iteration_best_distance2, self.global_best_distance2))
            # print(self.global_best_distance1, self.global_best_tour1, self.global_best_distance2, self.global_best_tour2)
        with open("outout1.txt", "w") as file:
            for item in iteration_list1:
                file.write(str(item) + "\n")
        with open("outout2.txt", "w") as file:
            for item in iteration_list2:
                file.write(str(item) + "\n")
        """"""



    def run(self):
        self._ACGA()
        '''
        print()  # 输出最优解 
        '''
if __name__ == '__main__':
    _colony_size = 50
    _colony_size1 = _colony_size
    cities1 = []
    _colony_size2 = _colony_size
    cities2 = []
    _steps = 500
    with open('eil51-426.txt') as f:
        for line in f.readlines():
            city = line.split(' ')
            cities1.append([float(city[1]), float(city[2])])
    f.close()

    with open('berlin52-7542.txt') as f:
        for line in f.readlines():
            city = line.split(' ')
            cities2.append([float(city[1]), float(city[2])])
    f.close()

    _nodes1 = cities1
    _nodes2 = cities2

    ACGA = SolveTSPUsingACO(colony_size1=_colony_size1, steps=_steps, nodes1=_nodes1, colony_size2=_colony_size2, nodes2=_nodes2)
    ACGA.run()


end_time = time.time()

elapsed_time = end_time - start_time


print(f"代码运行时间: {elapsed_time} 秒")
