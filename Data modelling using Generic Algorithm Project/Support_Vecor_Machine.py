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
        a = []

        for i in range(594):
            a.append(i)
        print(a)
        a = numpy.array(a)
        print('After converting to numpy')
        print(a)

        self.active_descriptors = a

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

        return self.descriptors, self.targets, self.active_descriptors

# ------------------------------------------------------------------------------------------------
    def function_step3(self):
        self.descriptors, self.targets = self.input_process.sort_descriptor_matrix(self.descriptors, self.targets)
        return self.descriptors, self.targets

# ------------------------------------------------------------------------------------------------
    def function_step4(self):
        self.X_Train, self.X_Valid, self.X_Test, self.Y_Train, self.Y_Valid, self.Y_Test = self.input_process.simple_split(
            self.descriptors, self.targets)
        self.data = {'TrainX': self.X_Train, 'TrainY': self.Y_Train, 'ValidateX': self.X_Valid,
                     'ValidateY': self.Y_Valid,
                     'TestX': self.X_Test, 'TestY': self.Y_Test, 'UsedDesc': self.active_descriptors}
        print(str(self.descriptors.shape[1]) + " valid descriptors and " + str(
            self.targets.__len__()) + " molecules available.")
        return self.X_Train, self.X_Valid, self.X_Test, self.Y_Train, self.Y_Valid, self.Y_Test, self.data

# ------------------------------------------------------------------------------------------------
    def function_step5(self):

        self.binary_model = zeros((50, self.X_Train.shape[1]))
        newpopulation = zeros((4, self.X_Train.shape[1]))
        cc = 0
        L = (0.015 * 593)
        min1 = 20000

        min2 = 20000

        indexMin1 = indexMin2 = 0
        counter = 0

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

# ------------------------------------------------------------------------------------------------

        self.binary_model = ValidRow(self.binary_model)
        print(self.binary_model)

        regressor = svm.SVR()

        regressor.fit(self.X_Train, self.Y_Train)
        print('This is SVM!')
        instructions = {'dim_limit': 4, 'algorithm': 'GA', 'MLM_type': 'SVM'}

        fitting_object = fitting_scoring.FittingScoringClass()

        directory = os.path.join(os.getcwd(), 'Outputs')
        output_filename = 'SVM_Output.csv'
        file_path = os.path.join(directory, output_filename)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        fileOut = open(file_path, 'w', newline='')  # create stream object for output file
        fileW = csv.writer(fileOut)
        fileW.writerow(
            ['Descriptor ID', 'Fitness', 'Algorithm', 'Dimen', 'R2_Train', 'R2_Valid', 'R2_Test', 'RMSE', 'MAE',
             'pred_Acc'
             ])
        bestAcc = 0

        for k in range(10000):
            print('now k is: ', k)
            try:

                self.trackDesc, self.trackFitness, self.trackModel, \
                self.trackDimen, self.trackR2train, self.trackR2valid, \
                self.trackR2test, self.testRMSE, self.testMAE, \
                self.testAccPred = fitting_object.evaluate_population(model=regressor, instructions=instructions,
                                                                      data=self.data, population=self.binary_model,
                                                                      exportfile=fileW)

                counter = 0
                for key in self.trackDesc.keys():
                    if self.trackFitness[key] < min1:
                        min2 = min1
                        min1 = self.trackFitness[key]
                        indexMin2 = indexMin1
                        indexMin1 = counter
                        print('Previous fitness value is:', min2)
                        print('Updated Fitness value is:', min1)
                        print("Acceptable Predictions From Testing Set:")
                        print("\t" + str(100 * self.testAccPred[key]) + "% of predictions")
                        bestR2train = self.trackR2train[key]
                        bestR2test = self.trackR2test[key]
                        bestR2valid = self.trackR2valid[key]

                        trailacc = 100 * self.testAccPred[key]
                        if bestAcc < trailacc:
                            bestR2train = self.trackR2train[key]
                            bestR2test = self.trackR2test[key]
                            bestR2valid = self.trackR2valid[key]
                            bestAcc = trailacc
                        print('Current best Accuracy is: ', bestAcc)

                    counter += 1

                Oldpopulation = self.binary_model

                newpopulation[0] = Oldpopulation[indexMin1]
                newpopulation[1] = Oldpopulation[indexMin2]

                dad = newpopulation[0]
                mom = newpopulation[1]

# ------------------------------------------------------------------------------------------------
                def generateChildren(dad, mom):

                    child1 = numpy.zeros(593, dtype=int)
                    child2 = numpy.zeros(593, dtype=int)
                    n = random.randint(0, 593)
                    for i in range(0, n):
                        child1[i] = dad[i]

                    for i in range(n + 1, 592):
                        child1[i] = mom[i]

                    for i in range(0, n):
                        child2[i] = mom[i]

                    for i in range(n + 1, 592):
                        child2[i] = dad[i]

                    return child1, child2

                child1, child2 = generateChildren(dad, mom)

                newpopulation[2] = child1
                newpopulation[3] = child2

                self.binary_model = zeros((50, self.X_Train.shape[1]))

                self.binary_model = ValidRow(self.binary_model)

                self.binary_model = numpy.concatenate((newpopulation, self.binary_model[4:]), axis=0)

# ------------------------------------------------------------------------------------------------
                def mutation(binary_model):
                    for i in range(50):
                        for j in range(593):
                            X = random.randint(0, 100)

                            if (X <= 0.0005 and binary_model[i][j] == 1):
                                binary_model[i][j] = 0
                            elif (X <= 0.0005 and binary_model[i][j] == 0):
                                binary_model[i][j] == 1
                    return binary_model

# ------------------------------------------------------------------------------------------------

                self.binary_model = mutation(self.binary_model)

                print('===================================================')

                print('Ready for next generation..')

            except ValueError:
                print("")
        print('End of Generations.')
        print('Best recorded prediction model was: ')
        print('')
        print('R^2 Train: ', bestR2train)
        print('R^2 Test: ', bestR2test)
        print('R^2 Valid: ', bestR2valid)
        print('Best Accuracy in %: ', bestAcc)

        return regressor, instructions, min1, min2, self.trackDesc, self.trackFitness, self.trackModel, self.trackDimen, self.trackR2train, self.trackR2valid, self.trackR2test, self.testRMSE, self.testMAE, self.testAccPred

# ------------------------------------------------------------------------------------------------
    def function_step6(self):
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

regressor, instructions, lowestfitness, lower, trackDesc, trackFitness, trackModel, trackDimen, trackR2train, trackR2valid, trackR2test, testRMSE, testMAE, testAccPred = Alzheimer1.function_step5()

print('------------------------step 5,6 ends-------------------------')
Alzheimer1.function_step6()
print('------------------------step 7 ends-------------------------')
