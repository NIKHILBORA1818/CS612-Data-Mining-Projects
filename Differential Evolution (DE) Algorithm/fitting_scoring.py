from numpy import *
import hashlib
import re


class FittingScoringClass():
    # ---------------------------------------------------------------------------------------------
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

            if fitness[i] < unfit:
                # store stats
                trackDesc[idx] = (
                    re.sub(",", ",", str(xi)))  # Editing descriptor set to match actual indices in the original data.
                trackDesc[idx] = (re.sub(",", ",", str(
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

        return trackDesc, trackFitness, trackModel, trackDimen, trackR2, trackR2PredValidation, trackR2PredTest, trackRMSE, trackMAE, trackAcceptPred

    # ---------------------------------------------------------------------------------------------
    def sigfig(self, x):
        return float("%.4f" % x)

    # ---------------------------------------------------------------------------------------------
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

    # ---------------------------------------------------------------------------------------------
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

    # ---------------------------------------------------------------------------------------------
    def calculateMAE(self, experimental, predictions):
        errors = [abs(experimental[i] - predictions[i]) for i in range(experimental.__len__())]
        return sum(errors) / experimental.__len__()

    # ---------------------------------------------------------------------------------------------
    def calculateRMSE(self, experimental, predictions):
        sum_of_squares = 0
        errors_below_1 = 0
        for mol in range(experimental.__len__()):
            abs_error = abs(experimental[mol] - predictions[mol])
            sum_of_squares += pow(abs_error, 2)
            if abs_error < 1:
                errors_below_1 += 1
        return sqrt(sum_of_squares / experimental.__len__()), int(errors_below_1)

    # ---------------------------------------------------------------------------------------------
    def write(self, exportfile, descriptors, fitnesses, modelnames,
              dimensionality, r2trainscores, r2validscores, r2testscores, rmse, mae, acc_pred):

        if exportfile is not None:
            for key in fitnesses.keys():
                exportfile.writerow([descriptors[key], fitnesses[key], modelnames[key],
                                     dimensionality[key], r2trainscores[key], r2validscores[key],
                                     r2testscores[key], rmse[key], mae[key], acc_pred[key]
                                     ])