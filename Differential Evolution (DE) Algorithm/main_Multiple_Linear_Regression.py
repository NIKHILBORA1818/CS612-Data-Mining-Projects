from numpy import zeros
from sklearn import linear_model
import pandas as pd
import fitting_scoring
import process_input
from sklearn import svm
from sklearn import neural_network
import random
import numpy
import os
import csv
from numpy import *
import csv
import numpy as np
import pandas as pd
import math
from pandas import DataFrame

# ------------------------------------------------------------------------------------------------
class DrugDiscovery:
    input_process = process_input.InputProcessor()

    descriptors_file = "Practice_Descriptors.csv"
    targets_file = "Practice_Targets.csv"

# ------------------------------------------------------------------------------------------------
    def function_step1(self):
        self.descriptors = self.input_process.open_descriptor_matrix(self.descriptors_file)
        print('Original matrix dimensions : ', self.descriptors.shape)
        self.targets = self.input_process.open_target_values(self.targets_file)
        return self.descriptors, self.targets

# ------------------------------------------------------------------------------------------------
    def function_step2(self):
        self.descriptors, self.targets = self.input_process.removeInvalidData(self.descriptors, self.targets)
        print()
        print(self.targets)
        print()
        print('--------------------step 1 ends-------------------------')
        self.descriptors, self.active_descriptors = self.input_process.removeNearConstantColumns(self.descriptors)

        print('After removing invalid datas descriptor are as follows: ')
        print(self.descriptors)
        print()
        print('Now descriptor dimensions are ', self.descriptors.shape)
        # Rescale the descriptor data
        self.descriptors = self.input_process.rescale_data(self.descriptors)
        print('------------------------Rescaled matrix is below--------------------')
        print('Rescaled value of Xes is:')
        print(self.descriptors)

        print('Rescaled matrix dimenstions are:', self.descriptors.shape)
        print('------------------------------------------------------')
        return self.descriptors, self.targets, self.active_descriptors

# ------------------------------------------------------------------------------------------------
    def function_step3(self):
        self.descriptors, self.targets = self.input_process.sort_descriptor_matrix(self.descriptors, self.targets)
        return self.descriptors, self.targets

# ------------------------------------------------------------------------------------------------
    def function_step4(self):
        self.X_Train, self.X_Valid, self.X_Test, self.Y_Train, self.Y_Valid, self.Y_Test = self.input_process.simple_split(self.descriptors, self.targets)
        self.data = {'TrainX': self.X_Train, 'TrainY': self.Y_Train, 'ValidateX': self.X_Valid,
                     'ValidateY': self.Y_Valid,
                     'TestX': self.X_Test, 'TestY': self.Y_Test, 'UsedDesc': self.active_descriptors}
        print(str(self.descriptors.shape[1]) + " valid descriptors and " + str(self.targets.__len__()) + " molecules available.")
        return self.X_Train, self.X_Valid, self.X_Test, self.Y_Train, self.Y_Valid, self.Y_Test, self.data

# ------------------------------------------------------------------------------------------------
    def function_step5(self):
        self.binary_model = zeros((50, self.X_Train.shape[1]))

        L = (0.015 * 593)
        # ------------------------------------------------------------------------------------------------

        def ValidRow(binary_model):
            for i in range(50):
                cc = 0
                for j in range(593):
                    r = random.randint(0, 593)
                    if r < L:
                        binary_model[i][j] = 1
                        cc += 1
                if cc < 5 and cc > 25:
                    i -= 1
                else:
                    continue
            return binary_model

        self.binary_model = ValidRow(self.binary_model)

        return self.binary_model

# ------------------------------------------------------------------------------------------------
    def function_step6(self):
        regressor = linear_model.LinearRegression()

        instructions = {'dim_limit': 4, 'algorithm': 'None', 'MLM_type': 'MLR'}

        fitting_object = fitting_scoring.FittingScoringClass()
        directory = os.path.join(os.getcwd(), 'Outputs')
        output_filename = 'MLR_Output_Project3.csv'
        file_path = os.path.join(directory, output_filename)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        fileOut = open(file_path, 'w', newline='')  # create stream object for output file
        fileW = csv.writer(fileOut)
        fileW.writerow(
            ['Descriptor ID', 'Fitness', 'Algorithm', 'Dimen', 'R2_Train', 'R2_Valid', 'R2_Test', 'RMSE', 'MAE',
             'pred_Acc'
             ])
        print('This is MLR!')

        for gen in range(10000):
            print('Current generation is: ', gen)
            regressor.fit(self.X_Train, self.Y_Train)

            self.trackDesc, self.trackFitness, self.trackModel, \
            self.trackDimen, self.trackR2train, self.trackR2valid, \
            self.trackR2test, self.testRMSE, self.testMAE, \
            self.testAccPred = fitting_object.evaluate_population(model=regressor, instructions=instructions, data=self.data,
                                                              population=self.binary_model, exportfile=fileW)

            print('========================================================================')

            nik = []
            traill = []
            for i in range(50):
                traill.append(i)

            for key in self.trackDesc.keys():
                nik.append(self.trackFitness[key])

            df = pd.DataFrame(nik)
            df.columns = ['fitness']

            df1 = pd.DataFrame(traill)
            df1.columns = ['order']

            df['order'] = df1

            df2 = df.sort_values('fitness')

            order = []

            order = df2['order'].values.tolist()

            binary_model2 = self.binary_model.copy()

            for i in range(len(order)):
                # for j in range(593):
                a = order[i]
                binary_model2[i] = self.binary_model[a]

            pop = binary_model2

            Oldpop = pop

            for i in range(1, 50):
                V = numpy.arange(593)

                a = random.randint(1, 49)
                b = random.randint(1, 49)
                c = random.randint(1, 49)

                F = 0.7

                for j in range(0, 593):
                    V[j] = math.floor(abs(Oldpop[a, j] + (F * (Oldpop[b, j] - Oldpop[c, j]))))

                CV = 0.7
                CV = random.randint(0, 1)

                for k in range(0, 593):
                    Random = random.uniform(0, 1)
                    if (Random < CV):
                        pop[i, k] = V[k]
                    else:
                        continue

            def ValidRow(binary_model):
                L = (0.015 * 593)
                while True:
                    cc = 0
                    for j in range(593):
                        r = random.randint(0, 593)
                        if r < L:
                            binary_model[j] = 1
                            cc += 1
                    if cc < 5 or cc > 25:
                        continue
                    else:
                        break
                return binary_model

            for i in range(0, 50):
                check = 0
                for j in range(0, 593):
                    if pop[i, j] == 1:
                        check += 1

                if check < 5 or check > 25:
                    pop[i] = ValidRow(pop[i])

        return regressor, instructions, self.trackDesc, self.trackFitness, self.trackModel, self.trackDimen, self.trackR2train, self.trackR2valid, self.trackR2test, self.testRMSE, self.testMAE, self.testAccPred


    def function_step7(self):
        print('\n\nFitness\t\tAccuracy\t\t\tR_SquareTrain\tR_SquareValid\tR_SquareTest\tRMSE')
        print('========================================================================')
        for key in trackDesc.keys():
            print(str(trackFitness[key]) + '\t\t' + str(testAccPred[key]) + '\t\t\t' + str(trackR2train[key]) \
                  + '\t\t\t\t' + str(trackR2valid[key]) + '\t\t' + str(trackR2test[key]) + '\t\t' + str(testRMSE[key]))
# ------------------------------------------------------------------------------------------------

Alzheimer1 = DrugDiscovery()
descriptors1, targets1 = Alzheimer1.function_step1()
print()
print('Original descriptors are as follow:')
print()
print(descriptors1)
print()
print('Targets are as below:')
print()
print(targets1)
print()

print('______Function1 done_________')
descriptors, targets, active_descriptors = Alzheimer1.function_step2()
print()
print('------------------------step 2 ends-------------------------')

descriptors, targets = Alzheimer1.function_step3()
print('After sorting descriptor matrix is : ')
print(descriptors)
print()
print('after sorting targets are:')
print(targets)
print('------------------------step 3 ends-------------------------')
print()
X_Train, X_Valid, X_Test, Y_Train, Y_Valid, Y_Test, data = Alzheimer1.function_step4()
print()
print('------------------------step 4 ends-------------------------')

binary_model = Alzheimer1.function_step5()
print('------------------------step 5 ends-------------------------')

regressor, instructions, trackDesc, trackFitness, trackModel, trackDimen, trackR2train, trackR2valid, trackR2test, testRMSE, testMAE, testAccPred = Alzheimer1.function_step6()

print('------------------------step 6 ends-------------------------')
Alzheimer1.function_step7()
print('------------------------step 7 ends-------------------------')
