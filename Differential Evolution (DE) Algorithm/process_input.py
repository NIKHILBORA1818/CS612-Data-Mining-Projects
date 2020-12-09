from numpy import *
import csv
import numpy as np
import pandas as pd

class InputProcessor():
# ------------------------------------------------------------------------------------------------

    def rescale_data(self, descriptor_matrix):
        descriptor_matrix = descriptor_matrix.astype(np.float)
        XNormed = (descriptor_matrix - descriptor_matrix.mean()) / (descriptor_matrix.std())

        return XNormed


    # ------------------------------------------------------------------------------------------------
    # What do we need to sort the data?

    def sort_descriptor_matrix(self, descriptors, targets):
        # Placing descriptors and targets in ascending order of target (IC50) value.
        alldata = ndarray((descriptors.shape[0], descriptors.shape[1] + 1))
        alldata[:, 0] = targets
        alldata[:, 1:alldata.shape[1]] = descriptors
        alldata = alldata[alldata[:, 0].argsort()]
        descriptors = alldata[:, 1:alldata.shape[1]]
        targets = alldata[:, 0]

        return descriptors, targets


    # ------------------------------------------------------------------------------------------------

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


    # ------------------------------------------------------------------------------------------------

    # try to optimize this code if possible

    def open_descriptor_matrix(self, fileName):
        preferred_delimiters = [';', '\t', ',', '\n']

        with open(fileName, mode='r') as csvfile:
            # Dynamically determining the delimiter used in the input file
            row = csvfile.readline()

            delimit = ','
            for d in preferred_delimiters:
                if d in row:
                    delimit = d
                    break

            # Reading in the data from the input file
            csvfile.seek(0)
            datareader = csv.reader(csvfile, delimiter=delimit, quotechar=' ')
            dataArray = array([row for row in datareader if row != ''], order='C')

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
    # Removes constant and near-constant descriptors.
    # But I think also does that too for real data.
    # So for now take this as it is

    def removeNearConstantColumns(self,data_matrix, num_unique=10):
        useful_descriptors = [col for col in range(data_matrix.shape[1])
                              if len(set(data_matrix[:, col])) > num_unique]
        filtered_matrix = data_matrix[:, useful_descriptors]

        remaining_desc = zeros(data_matrix.shape[1])
        remaining_desc[useful_descriptors] = 1

        return filtered_matrix, where(remaining_desc == 1)[0]


    # ------------------------------------------------------------------------------------------------
    # part 1: Removes all rows with junk (ex: NaN, etc). Note that the corresponding IC50 value should bedeleted too
    # Part 2: Remove columns with 20 junks or more. Otherwise the junk should be replaced with zero
    # Part 3: remove all columns that have zero in every cell

    def removeInvalidData(self, descriptors, targets):
        # Write your code in here
        descriptors = pd.DataFrame(data=descriptors)

        descriptors.dropna(axis=0, how='all', inplace=True)
        print('After Removing rows with junk values: ', descriptors.shape)
        descriptors.dropna(axis=1, thresh=262, inplace=True)
        print('After Removing columns with 20 or more junk values: ', descriptors.shape)
        descriptors.replace(np.nan, 0, inplace=True)
        descriptors = descriptors.loc[:, (descriptors != 0).any(axis=0)]
        print('After Removing columns having 0 in every cell: ', descriptors.shape)
        descriptors = descriptors.to_numpy()
        print(descriptors)
        return descriptors, targets
