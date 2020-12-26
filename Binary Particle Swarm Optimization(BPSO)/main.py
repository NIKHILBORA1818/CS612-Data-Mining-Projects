import numpy
from numpy import *
import pandas as pd
import csv
import hashlib
import re
from sklearn import *
import random
import numpy as np
import os
from math import sqrt
import math
import time
import sklearn.utils
from sklearn.exceptions import ConvergenceWarning
from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
from sklearn import linear_model, svm, neural_network


class Drug_Discovery:
    descriptors = None
    targets = None
    active_descriptors = None
    X_Train = None
    X_Valid = None
    X_Test = None
    Y_Train = None
    Y_Valid = None
    Y_Test = None
    Data = None
    binary_model = None

    def __init__(self, descriptors_file, targets_file):
        self.descriptors = self.open_descriptor_matrix(descriptors_file)
        self.targets = self.open_target_values(targets_file)

    # **********************************************************************************************
    def processData(self):
        self.descriptors, self.targets = self.removeInvalidData(self.descriptors, self.targets)
        self.descriptors, self.active_descriptors = self.removeNearConstantColumns(self.descriptors)
        # Rescale the descriptor data
        self.descriptors = self.rescale_data(self.descriptors)
        # sort data
        self.descriptors, self.targets = self.sort_descriptor_matrix(self.descriptors, self.targets)

    # **********************************************************************************************
    def splitData(self):
        self.X_Train, self.X_Valid, self.X_Test, self.Y_Train, self.Y_Valid, self.Y_Test = self.simple_split(
            self.descriptors, self.targets)
        self.data = {'TrainX': self.X_Train, 'TrainY': self.Y_Train, 'ValidateX': self.X_Valid,
                     'ValidateY': self.Y_Valid,
                     'TestX': self.X_Test, 'TestY': self.Y_Test, 'UsedDesc': self.active_descriptors}

        print(str(self.descriptors.shape[1]) + " valid descriptors and " + str(
            self.targets.__len__()) + " molecules available.")
        return self.X_Train, self.X_Valid, self.X_Test, self.Y_Train, self.Y_Valid, self.Y_Test, self.data

    # **********************************************************************************************
    # Set up the demonstration model
    def setUpDemoModel(self):
        binary_model = zeros((50, 593))
        c = 0
        for i in range(50):
            for j in range(593):
                r = random.randint(0, 593)
                L = int(0.015 * 593)
                if r < L:
                    binary_model[i][j] = 1
                    c += 1
            if c > 5 and c < 25:
                continue
            else:
                i -= 1

    # **********************************************************************************************
    # Create a Multiple Linear Regression object to fit our demonstration model to the data
    def runModel(self, regressor, instructions):
        trackDesc, trackFitness, trackModel, trackDimen, trackR2train, trackR2valid, trackR2test, testRMSE, testMAE, testAccPred = self.evaluate_population(
            model=regressor, instructions=instructions, data=self.data,
            population=self.binary_model, exportfile=None)
        self.outputModelInfo(trackDesc, trackFitness, trackModel, trackDimen, trackR2train, trackR2valid, trackR2test,
                             testRMSE, testMAE, testAccPred)

    # **********************************************************************************************

    def isValidRow(self, row):
        cc = 0
        for value in row:
            if value == 1:
                cc += 1
            if cc < 5 or cc > 25:
                return False

    def getValidRow(self):
        numberofDescriptors = self.X_Train.shape[1]
        rowValid = zeros((1, numberofDescriptors))
        count1 = 0
        while (count1 < 5) or (count1 > 25):

            for i in range(numberofDescriptors):
                rand1 = round(random.uniform(0, 100), 2)
                if rand1 < 1.5:
                    rowValid[0][i] = 1
                    count1 += 1
        return rowValid

    def BPSO(self, regressor, instructions, numGenerations, fileW, data):

        def step1(velocity):
            for i in range(50):
                for j in range(593):
                    velocity[i][j] = random.uniform(0, 1)

            return velocity

        def step2(population):
            L = (0.015 * 593)
            for i in range(50):
                cc = 0
                for j in range(593):
                    r = random.randint(0, 593)
                    if r < L:
                        population[i][j] = 1
                        cc += 1
                    if cc < 5 and cc > 25:
                        i -= 1
                    else:
                        continue
            return population

        def step3(population, fitness):

            fitness = list(fitness.values())
            local_best_matrix = population
            local_fitness = fitness
            return local_best_matrix, local_fitness

        def step4(init_local_best_matrix, init_local_fitness):
            global global_best_row
            global global_best_row_fitness

            global_best_row = np.zeros(593)
            global_best_row_fitness = 200.00
            return global_best_row, global_best_row_fitness

        def step5(population, fitness, local_best_matrix, local_fitness, trackDesc, i):

            fitness = list(fitness.values())
            for i in range(50):
                if fitness[i] < local_fitness[i]:
                    local_best_matrix[i] = population[i]
                    local_fitness[i] = fitness[i]

            return local_best_matrix, local_fitness

        def step6(local_best_matrix, local_fitness):
            global global_best_row
            global global_best_row_fitness

            idx = local_fitness.index(min(local_fitness))
            if local_fitness[idx] < global_best_row_fitness:
                global_best_row = local_best_matrix[idx]
                global_best_row_fitness = local_fitness[idx]

            return global_best_row, global_best_row_fitness

        def step7(velocity, population, local_best_matrix, global_best_row, c1=2, c2=2, inertia=0.9):
            new_velocity = np.zeros((50, 593))
            for i in range(50):
                for j in range(593):
                    t1 = c1 * numpy.random.random() * (local_best_matrix[i][j] - population[i][j])
                    t2 = c2 * numpy.random.random() * (global_best_row[j] - population[i][j])
                    new_velocity[i][j] = (inertia * velocity[i][j]) + t1 + t2

            return new_velocity

        def create_new_population(population, velocity, local_best_matrix, alpha):

            oldPopulation = population

            p = 0.5 * (1 + alpha)

            for i in range(50):
                for j in range(593):
                    if velocity[i][j] <= alpha:
                        population[i][j] = oldPopulation[i][j]
                    elif velocity[i][j] > alpha and velocity[i][j] <= p:
                        population[i][j] = local_best_matrix[i][j]
                    elif velocity[i][j] > p and velocity[i][j] <= 1:
                        population[i][j] = global_best_row[j]
                    else:
                        population[i][j] = oldPopulation[i][j]
                if self.isValidRow(population[i]) == False:
                    population[i] = self.getValidRow()
            return population

        def evolve_population(population, velocity, init_local_best_matrix, local_fitness, regressor, instructions,
                              data, fileW, trackDesc, numGenerations, global_best_row, global_best_row_fitness):
            alpha = 0.5

            for i in range(1, numGenerations):
                print(f'Currently in generation : {i}')
                population = create_new_population(population, velocity, init_local_best_matrix, alpha)
                self.trackDesc, trackFitness = self.evaluate_population(model=regressor, instructions=instructions,
                                                                        data=self.data, population=population,
                                                                        exportfile=fileW)

                init_local_best_matrix, local_fitness = step5(population, trackFitness,
                                                                                 init_local_best_matrix, local_fitness,
                                                                                 trackDesc, i)
                global_best_row, global_best_row_fitness = step6(init_local_best_matrix, local_fitness)

                velocity = step7(velocity, population, init_local_best_matrix, global_best_row)
                alpha = alpha - (0.17 / 10000)

                #print(f'End of generation number: {i}')

            print('End of generations...')

        fileW.writerow(
            ['Descriptor ID', 'Fitness', 'Algorithm', 'Dimen', 'R2_Train', 'R2_Valid', 'R2_Test', 'RMSE', 'MAE',
             'Pred Acc'])

        population = zeros((50, self.X_Train.shape[1]))
        velocity = zeros((50, self.X_Train.shape[1]))
        population = step2_createNewPopulation(population)
        velocity = step1_createInitialVelocity(velocity)

        self.trackDesc, self.trackFitness = self.evaluate_population(model=regressor, instructions=instructions,
                                                                     data=self.data, population=population,
                                                                     exportfile=fileW)

        global_best_row = np.zeros(593)
        global_best_row_fitness = 2000

        init_local_best_matrix, init_local_fitness = step3(population, self.trackFitness)
        global_best_row, global_best_row_fitness = step4(init_local_best_matrix,
                                                                                  init_local_fitness)
        # this is the main recurring function
        evolve_population(population, velocity, init_local_best_matrix, init_local_fitness, \
                          regressor, instructions, self.data, fileW, self.trackDesc, numGenerations, global_best_row,
                          global_best_row_fitness)

    # **********************************************************************************************
    def outputModelInfo(self, trackDesc, trackFitness, trackModel, trackDimen, trackR2train, trackR2valid, trackR2test,
                        testRMSE, testMAE, testAccPred):
        print("\n\nFitness\t\tDimension\t\t\tR_SquareTrain\t\tR_SquareValid\t\tRMSE\t\tDescriptors")
        print("========================================================================")

        for key in trackDesc.keys():
            print(str(trackFitness[key]) + "\t\t" + str(trackDimen[key]) + "\t\t\t\t\t" + str(
                trackR2train[key]) + "\t\t\t\t" + str(trackR2valid[key]) + "\t\t\t\t" + str(
                testRMSE[key]) + "\t\t" + str(trackDesc[key]))

    # **********************************************************************************************
    # try to optimize this code if possible
    def open_descriptor_matrix(self, fileName):
        preferred_delimiters = [';', '\t', ',', '\n']

        preferred_delimiters = [';', '\t', ',', '\n']
        file = fileName
        dataArray = np.genfromtxt(file, delimiter=',')

        if (min(dataArray.shape) == 1):  # flatten arrays of one row or column
            return dataArray.flatten(order='C')
        else:
            return dataArray

    # ************************************************************************************
    # Try to optimize this code if possible
    def open_target_values(self, fileName):
        preferred_delimiters = [';', '\t', ',', '\n']

        with open(fileName, mode='r') as csvfile:
            # Dynamically determining the delimiter used in the input file
            row = csvfile.readline()
            delimit = ','
            for d in preferred_delimiters:
                if d in row:
                    delimit = d
                    break

            csvfile.seek(0)
            datalist = csvfile.read().split(delimit)
            if ' ' in datalist:
                datalist = datalist[0].split(' ')

        for i in range(datalist.__len__()):
            datalist[i] = datalist[i].replace('\n', '')
            try:
                datalist[i] = float(datalist[i])
            except:
                datalist[i] = datalist[i]

        try:
            datalist.remove('')
        except ValueError:
            no_empty_strings = True

        return datalist
        # **********************************************************************************************

    def removeInvalidData(self, descriptors, targets):
        # Numpy to df and series
        descriptors_df = pd.DataFrame(descriptors)
        targets_series = pd.Series(targets)

        # Junk to NaN
        descriptors_df = descriptors_df.apply(pd.to_numeric, errors='coerce')

        # Get indexes of rows with any NaN values
        descriptor_rows_with_nan = [index for index, row in descriptors_df.iterrows() if row.isnull().any()]

        # Drop rows with any NaN values
        descriptors_df = descriptors_df.drop(descriptor_rows_with_nan)
        targets_series = targets_series.drop(descriptor_rows_with_nan)
        delCount = len(descriptor_rows_with_nan)
        print("Dropped ", delCount, " rows containing any junk values.")

        # Drop columns that have more than 20 junks
        numJunkPerCol_Series = descriptors_df.isna().sum()
        delCount = numJunkPerCol_Series[numJunkPerCol_Series > 20].count()
        descriptors_df = descriptors_df.drop(numJunkPerCol_Series[numJunkPerCol_Series > 20].index, axis=1)
        print("Dropped ", delCount, " columns containing more than 20 junk values.")

        # change NaN to 0
        print("Converting remaining junk values to 0...")
        descriptors_df = descriptors_df.fillna(0)

        # drop columns containing all zeros
        tempLen = len(descriptors_df.columns)
        descriptors_df = descriptors_df.loc[:, descriptors_df.ne(0).any(axis=0)]
        delCount = tempLen - len(descriptors_df.columns)
        print("Dropped ", delCount, " columns containing all zeros.")

        # df and series to numpy for return
        descriptors = descriptors_df.to_numpy()
        targets = targets_series.to_numpy()

        return descriptors, targets

    # **********************************************************************************************
    # Removes constant and near-constant descriptors.
    # But I think also does that too for real data.
    # So for now take this as it is

    def removeNearConstantColumns(self, data_matrix, num_unique=10):
        useful_descriptors = [col for col in range(data_matrix.shape[1])
                              if len(set(data_matrix[:, col])) > num_unique]
        filtered_matrix = data_matrix[:, useful_descriptors]

        remaining_desc = zeros(data_matrix.shape[1])
        remaining_desc[useful_descriptors] = 1

        return filtered_matrix, where(remaining_desc == 1)[0]

    # **********************************************************************************************
    def rescale_data(self, descriptor_matrix):
        # Statistics for dataframe
        df = pd.DataFrame(descriptor_matrix)
        rescaled_matrix = (df - df.values.mean()) / (df.values.std())
        print("Rescaled Matrix is: ")
        rescaled_matrix.to_csv("rescaledmatrix.csv")
        print(rescaled_matrix)
        return rescaled_matrix

    # **********************************************************************************************
    def sort_descriptor_matrix(self, descriptors, targets):
        # Placing descriptors and targets in ascending order of target (IC50) value.
        alldata = ndarray((descriptors.shape[0], descriptors.shape[1] + 1))
        alldata[:, 0] = targets
        alldata[:, 1:alldata.shape[1]] = descriptors
        alldata = alldata[alldata[:, 0].argsort()]
        descriptors = alldata[:, 1:alldata.shape[1]]
        targets = alldata[:, 0]

        return descriptors, targets

    # **********************************************************************************************
    # Performs a simple split of the data into training, validation, and testing sets.
    # So how does it relate to the Data Mining Prediction?

    def simple_split(self, descriptors, targets):

        testX_indices = [i for i in range(descriptors.shape[0]) if i % 4 == 0]
        validX_indices = [i for i in range(descriptors.shape[0]) if i % 4 == 1]
        trainX_indices = [i for i in range(descriptors.shape[0]) if i % 4 >= 2]

        TrainX = descriptors[trainX_indices, :]
        ValidX = descriptors[validX_indices, :]
        TestX = descriptors[testX_indices, :]

        TrainY = targets[trainX_indices]
        ValidY = targets[validX_indices]
        TestY = targets[testX_indices]

        return TrainX, ValidX, TestX, TrainY, ValidY, TestY

    # **********************************************************************************************
    def evaluate_population(self, model, instructions, data, population, exportfile):
        numOfPop = population.shape[0]
        fitness = zeros(numOfPop)
        predictive = 0

        TrainX = data['TrainX']
        TrainY = data['TrainY']
        ValidateX = data['ValidateX']
        ValidateY = data['ValidateY']
        TestX = data['TestX']
        TestY = data['TestY']
        UsedDesc = data['UsedDesc']

        trackDesc, trackFitness, trackModel, trackDimen, trackR2, trackR2PredValidation, \
        trackR2PredTest, trackRMSE, trackMAE, trackAcceptPred, trackCoefficients = self.InitializeTracks()

        unfit = 1000

        for i in range(numOfPop):

            xi = list(where(population[i] == 1)[0])

            idx = hashlib.sha1(array(xi)).digest()

            # Condenses binary models to a list of the indices of active features
            X_train_masked = TrainX.T[xi].T
            X_validation_masked = ValidateX.T[xi].T
            X_test_masked = TestX.T[xi].T

            try:
                model = model.fit(X_train_masked, TrainY)
            except:
                return unfit, fitness

            # Computed predicted values
            Yhat_training = model.predict(X_train_masked)
            Yhat_validation = model.predict(X_validation_masked)
            Yhat_testing = model.predict(X_test_masked)

            # Compute R2 scores (Prediction for Validation and Test set)
            r2_train = model.score(X_train_masked, TrainY)
            r2validation = model.score(X_validation_masked, ValidateY)
            r2test = model.score(X_test_masked, TestY)
            model_rmse, num_acceptable_preds = self.calculateRMSE(TestY, Yhat_testing)
            model_mae = self.calculateMAE(TestY, Yhat_testing)

            # Calculating fitness value
            if 'dim_limit' in instructions:
                fitness[i] = self.get_fitness(xi, TrainY, ValidateY, Yhat_training, Yhat_validation,
                                              dim_limit=instructions['dim_limit'])
            else:
                fitness[i] = self.get_fitness(xi, TrainY, ValidateY, Yhat_training, Yhat_validation)

            if predictive and ((r2validation < 0.5) or (r2test < 0.5)):
                # if it's not worth recording, just return the fitness
                # print("Ending program, fitness unacceptably low: ", predictive)
                continue

            # store stats
            if fitness[i] < unfit:
                # store stats
                trackDesc[idx] = (
                    re.sub(",", "_", str(xi)))  # Editing descriptor set to match actual indices in the original data.
                trackDesc[idx] = (re.sub(",", "_", str(
                    UsedDesc[xi].tolist())))  # Editing descriptor set to match actual indices in the original data.

                trackFitness[idx] = self.sigfig(fitness[i])
                trackModel[idx] = instructions['algorithm'] + ' with ' + instructions['MLM_type']
                trackDimen[idx] = int(xi.__len__())

                trackR2[idx] = self.sigfig(r2_train)
                trackR2PredValidation[idx] = self.sigfig(r2validation)
                trackR2PredTest[idx] = self.sigfig(r2test)

                trackRMSE[idx] = self.sigfig(model_rmse)
                trackMAE[idx] = self.sigfig(model_mae)
                trackAcceptPred[idx] = self.sigfig(float(num_acceptable_preds) / float(Yhat_testing.shape[0]))

            # For loop ends here.

        self.write(exportfile, trackDesc, trackFitness, trackModel, trackDimen, trackR2, trackR2PredValidation,
                   trackR2PredTest, trackRMSE, trackMAE, trackAcceptPred)

        return trackDesc, trackFitness

    # **********************************************************************************************
    def sigfig(self, x):
        return float("%.4f" % x)

    # **********************************************************************************************
    def InitializeTracks(self):
        trackDesc = {}
        trackFitness = {}
        trackAlgo = {}
        trackDimen = {}
        trackR2 = {}
        trackR2PredValidation = {}
        trackR2PredTest = {}
        trackRMSE = {}
        trackMAE = {}
        trackAcceptPred = {}
        trackCoefficients = {}

        return trackDesc, trackFitness, trackAlgo, trackDimen, trackR2, trackR2PredValidation, trackR2PredTest, \
               trackRMSE, trackMAE, trackAcceptPred, trackCoefficients

    # **********************************************************************************************
    def get_fitness(self, xi, T_actual, V_actual, T_pred, V_pred, gamma=3, dim_limit=None, penalty=0.05):
        n = len(xi)
        mT = len(T_actual)
        mV = len(V_actual)

        train_errors = [T_pred[i] - T_actual[i] for i in range(T_actual.__len__())]
        RMSE_t = sum([element ** 2 for element in train_errors]) / mT
        valid_errors = [V_pred[i] - V_actual[i] for i in range(V_actual.__len__())]
        RMSE_v = sum([element ** 2 for element in valid_errors]) / mV

        numerator = ((mT - n - 1) * RMSE_t) + (mV * RMSE_v)
        denominator = mT - (gamma * n) - 1 + mV
        fitness = sqrt(numerator / denominator)

        # Adjusting for high-dimensionality models.
        if dim_limit is not None:
            if n > int(dim_limit * 1.5):
                fitness += ((n - dim_limit) * (penalty * dim_limit))
            elif n > dim_limit:
                fitness += ((n - dim_limit) * penalty)

        return fitness

    # **********************************************************************************************
    def calculateMAE(self, experimental, predictions):
        errors = [abs(experimental[i] - predictions[i]) for i in range(experimental.__len__())]
        return sum(errors) / experimental.__len__()

    # **********************************************************************************************
    def calculateRMSE(self, experimental, predictions):
        sum_of_squares = 0
        errors_below_1 = 0
        for mol in range(experimental.__len__()):
            abs_error = abs(experimental[mol] - predictions[mol])
            sum_of_squares += pow(abs_error, 2)
            if abs_error < 1:
                errors_below_1 += 1
        return sqrt(sum_of_squares / experimental.__len__()), int(errors_below_1)

    # **********************************************************************************************

    simplefilter("ignore", category=ConvergenceWarning)

    def write(self, exportfile, descriptors, fitnesses, modelnames,
              dimensionality, r2trainscores, r2validscores, r2testscores, rmse, mae, acc_pred):

        if exportfile is not None:
            for key in fitnesses.keys():
                exportfile.writerow([descriptors[key], fitnesses[key], modelnames[key],
                                     dimensionality[key], r2trainscores[key], r2validscores[key],
                                     r2testscores[key], rmse[key], mae[key], acc_pred[key]
                                     ])


def main():
    Alzheimer_1 = Drug_Discovery("Practice_Descriptors.csv", "Practice_Targets.csv")
    Alzheimer_1.processData()
    X_Train, X_Valid, X_Test, Y_Train, Y_Valid, Y_Test, data = Alzheimer_1.splitData()
    directory = os.path.join(os.getcwd(), 'Outputs')

    # MLR
    print("\nMLR: ")
    output_filename = 'MLR_output_project4.csv'
    file_path = os.path.join(directory, output_filename)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    fileOut = open(file_path, 'w', newline='')
    fileW = csv.writer(fileOut)

    regressor = linear_model.LinearRegression()
    instructions = {'dim_limit': 4, 'algorithm': 'BPSO', 'MLM_type': 'MLR'}
    regressor.fit(X_Train, Y_Train)
    Alzheimer_1.BPSO(regressor, instructions, 10000, fileW, data)

    # #SVM
    print("\nSVM: ")
    output_filename = 'SVM_output_project4.csv'
    file_path = os.path.join(directory, output_filename)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    fileOut = open(file_path, 'w', newline='')
    fileW = csv.writer(fileOut)
    regressor = svm.SVR()
    regressor.fit(X_Train, Y_Train)
    instructions = {'dim_limit': 4, 'algorithm': 'BPSO', 'MLM_type': 'SVM'}
    Alzheimer_1.BPSO(regressor, instructions, 10000, fileW, data)

    # ANN
    print("\nANN: ")
    output_filename = 'ANN_output_project4.csv'
    file_path = os.path.join(directory, output_filename)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    fileOut = open(file_path, 'w', newline='')
    fileW = csv.writer(fileOut)
    regressor = neural_network.MLPRegressor(hidden_layer_sizes=(200, 100, 60, 40))
    regressor.fit(X_Train, Y_Train)
    instructions = {'dim_limit': 4, 'algorithm': 'BPSO', 'MLM_type': 'ANN'}
    Alzheimer_1.BPSO(regressor, instructions, 1000, fileW, data)


if __name__ == "__main__":
    main()
