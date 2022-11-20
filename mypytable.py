import mysklearn.myutils as myutils
import copy
import csv 
from tabulate import tabulate


class MyPyTable:
    """Represents a 2D table of data with column names.
    Attributes:
        column_names(list of str): M column names
        data(list of list of obj): 2D data structure storing mixed type data. There are N rows by M columns.
    """

    def __init__(self, column_names=None, data=None):
        """Initializer for MyPyTable.
        Args:
            column_names(list of str): initial M column names (None if empty)
            data(list of list of obj): initial table data in shape NxM (None if empty)
        """
        if column_names is None:
            column_names = []
        self.column_names = copy.deepcopy(column_names)
        if data is None:
            data = []
        self.data = copy.deepcopy(data)

    def pretty_print(self):
         """Prints the table in a nicely formatted grid structure.
         """
         print(tabulate(self.data, headers=self.column_names))

    def get_shape(self):
        """Computes the dimension of the table (N x M).
        Returns:
            int: number of rows in the table (N)
            int: number of cols in the table (M)
        """
        return len(self.data), len(self.column_names)

    def get_column(self, col_identifier, include_missing_values=True):
        """Extracts a column from the table data as a list.
        Args:
            col_identifier(str or int): string for a column name or int
                for a column index
            include_missing_values(bool): True if missing values ("NA")
                should be included in the column, False otherwise.
        Returns:
            list of obj: 1D list of values in the column
        Notes:
            Raise ValueError on invalid col_identifier
        """
        colToReturn = [] # initialize empty list. will later store the values in the column
        
        if col_identifier not in self.column_names:
            raise ValueError # no column with col_identifier as its name
        
        # determine which column to return values from
        colIndex = self.column_names.index(col_identifier)

        for row in self.data: # loop through entire table
            if include_missing_values == False and row[colIndex] in ['NA', 'N/A', '']: # not keeping missing values
                pass
            else:
                colToReturn.append(row[colIndex]) # add to list to return


        return colToReturn # TODO: fix this

    def convert_to_numeric(self):
        """Try to convert each value in the table to a numeric type (float).
        Notes:
            Leave values as is that cannot be converted to numeric.
        """
        for row in self.data: # loop through all rows
            for i in range(len(row)): # go through all data in each row
                # try to convert value to numeric
                try: 
                    numericValue = float(row[i]) # conversion
                    row[i] = numericValue # reassign
                except ValueError:
                    pass

    def drop_rows(self, rows_to_drop):
        """Remove rows from the table data.
        Args:
            rows_to_drop(list of list of obj): list of rows to remove from the table data.
        """

        newTable = []
        checkedRows = []

        for row in self.data:
            if row in rows_to_drop and row in checkedRows:
                newTable.append(row)
            elif row not in rows_to_drop:
                newTable.append(row)
            checkedRows.append(row)
        
        self.data = newTable


    def load_from_file(self, filename):
        """Load column names and data from a CSV file.
        Args:
            filename(str): relative path for the CSV file to open and load the contents of.
        Returns:
            MyPyTable: return self so the caller can write code like: table = MyPyTable().load_from_file(fname)
        
        Notes:
            Use the csv module.
            First row of CSV file is assumed to be the header.
            Calls convert_to_numeric() after load
        """
        table = [] # initialize empty table

        with open(filename, newline= '') as csvFile: # create new csv reader object
            csvReader = csv.reader(csvFile)
        
            for row in csvReader: 
                table.append(row) # append data to new table
        
        self.column_names = table.pop(0) # get header row
        self.data = table # assign self the new table
        self.convert_to_numeric() # convert values in table to numeric

        return self 

    def save_to_file(self, filename):
        """Save column names and data to a CSV file.
        Args:
            filename(str): relative path for the CSV file to save the contents to.
        Notes:
            Use the csv module.
        """

        with open(filename, mode='w', newline = '') as write_file:
            fileWriter = csv.writer(write_file)

            fileWriter.writerow(self.column_names)
            
            for row in self.data:
                fileWriter.writerow(row)


    def getKeyValues(self, row, keyNames):
        """ Gets the values of attributes associated with keys
        Args:
            keyNames: names of the keys for particular table
            row: row to get values from
        Returns: list of the values associated with keys
        """
        indexList = []
        values = []

        for keys in keyNames: # find indices of keys
            indexList.append(self.column_names.index(keys))
        
        for index in indexList: # add value of key to list
            values.append(row[index])

        return values # return the list of the key values

    def find_duplicates(self, key_column_names):
        """Returns a list of duplicates. Rows are identified uniquely based on key_column_names.
        Args:
            key_column_names(list of str): column names to use as row keys.
        Returns: 
            list of list of obj: list of duplicate rows found
        
        Notes:
            Subsequent occurrence(s) of a row are considered the duplicate(s). The first instance is not
                considered a duplicate.
        """

        allKeyValues = []
        duplicates = []

        for row in self.data: 
            values = self.getKeyValues(row, key_column_names) # get values of keys
            if values in allKeyValues: # check if seen keys before
                duplicates.append(row) # if seen before, add entire row to duplicates
            else: # first time seeing key
                allKeyValues.append(values)
            

        return duplicates 

    def check_for_missing_values(self, row, missingVal):
        """ Returns True if the row is missing a value, false otherwise
        Args:
            row: the row to check for missing values
        """
        for i in range(len(row)):
            if row[i] in missingVal: # found a missing value, return true
                return True 
        return False # no missing values



    def remove_rows_with_missing_values(self):
        """Remove rows from the table data that contain a missing value ("NA").
        """

        newTable = []

        for rows in self.data:
            if self.check_for_missing_values(rows, ['NA', 'N/A', '']) == False: # no missing values in the row
                newTable.append(rows) # append row to the new table
        self.data = newTable

    def remove_missing(self, marker):
        """Remove rows from the table data that contain a missing value ("NA").
        """

        newTable = []

        for rows in self.data:
            if self.check_for_missing_values(rows, marker) == False: # no missing values in the row
                newTable.append(rows) # append row to the new table
        self.data = newTable
                

    def replace_missing_values_with_column_average(self, col_name):
        """For columns with continuous data, fill missing values in a column by the column's original average.
        Args:
            col_name(str): name of column to fill with the original average (of the column).
        """

        column = self.get_column(col_name, False) # get all values in column (not including missing values)
        colAvg = sum(column) / len(column) # calculate average of all values

       
        index = self.column_names.index(col_name)

        for rows in self.data: # change NA values to average
            if rows[index] == 'NA':
                rows[index] = colAvg 
        
    def find_median(self, values):
        """ Helper function that find median of a list
        Args:
            values: a list of numerical values 
        Returns:
            median of the list of values
        """
        values.sort()
        if len(values) % 2 == 0:
            index1 = len(values) // 2
            index2 = index1 - 1
            return (values[index1] + values[index2]) / 2
        else:
            return values[len(values) // 2]

    def compute_summary_statistics(self, col_names):
        """Calculates summary stats for this MyPyTable and stores the stats in a new MyPyTable.
        Args:
            col_names(list of str): names of the continuous columns to compute summary stats for.
        Returns:
            MyPyTable: stores the summary stats computed. 
            The column names and their order is as follows: ["attribute", "min", "max", "mid", "avg", "median"]
        """

        newTable = MyPyTable() 
        newTable.column_names = ["attribute", "min", "max", "mid", "avg", "median"]

        for column in col_names:
            row = []
            currCol = self.get_column(column, False) # get column without missing values
            if len(currCol) != 0: # make sure there are values in the column
                row.append(column)
                row.append(min(currCol)) # minimum
                row.append(max(currCol)) # maximum 
                row.append((max(currCol) + min(currCol)) / 2) # mid
                row.append(sum(currCol) / len(currCol)) # average
                row.append(self.find_median(currCol)) # median
                newTable.data.append(row)  # append new row to new table

        return newTable 

    def perform_inner_join(self, other_table, key_column_names):
        """Return a new MyPyTable that is this MyPyTable inner joined with other_table based on key_column_names.
        Args:
            other_table(MyPyTable): the second table to join this table with.
            key_column_names(list of str): column names to use as row keys.
        Returns:
            MyPyTable: the inner joined table.
        """
        newTable = MyPyTable()

        # add the rest of the attribute names to the joined table
        for name in self.column_names:
            if name not in newTable.column_names:
                newTable.column_names.append(name)
        for name in other_table.column_names:
            if name not in newTable.column_names:
                newTable.column_names.append(name)
        
        

        for row in self.data: # loop through first table
            newRow = []
            keyVals = self.getKeyValues(row, key_column_names) # get values of attributes assoc. with keys
            for otherRow in other_table.data: # loop through each row in other table
                otherKeys = other_table.getKeyValues(otherRow, key_column_names) # get key values
                
                if otherKeys == keyVals: # found a match 
                    for i in range(len(row)):
                        newRow.append(row[i])
                    for i in range(len(otherRow)):
                        if other_table.column_names[i] not in key_column_names:
                            newRow.append(otherRow[i])
                    
                    newTable.data.append(newRow) # add row to joined table
        return newTable

    def perform_full_outer_join(self, other_table, key_column_names):
        """Return a new MyPyTable that is this MyPyTable fully outer joined with other_table based on key_column_names.
        Args:
            other_table(MyPyTable): the second table to join this table with.
            key_column_names(list of str): column names to use as row keys.
        Returns:
            MyPyTable: the fully outer joined table.
        Notes:
            Pad the attributes with missing values with "NA".
        """

        # create new table 
        joinedTable = MyPyTable()

        # fix the header
        for name in self.column_names:
            joinedTable.column_names.append(name)
        for name in other_table.column_names:
            if name not in joinedTable.column_names:
                joinedTable.column_names.append(name)

        # add all of the values from the "left" table to the new table, pad missing values with NA
        for row in self.data:
            newRow = ["NA"] * len(joinedTable.column_names) # intialize row

            for i in range(len(row)):
                index = joinedTable.column_names.index(self.column_names[i])
                newRow[index] = row[i] # update position in list to correct item
            joinedTable.data.append(newRow) # add the new row to the table

        # find matching key values
        seenKeys = [] # list of keys already seen in table

        # find all matched (basiaclly like inner join)
        for row in joinedTable.data:
            keys = joinedTable.getKeyValues(row, key_column_names) # find the keys in the row
            seenKeys.append(keys) # add keys to the seen keys
            for otherRow in other_table.data:
                otherKeys = other_table.getKeyValues(otherRow, key_column_names) # get right table keys

                if keys == otherKeys: # matching keys
                    for i in range(len(otherRow)):
                        if other_table.column_names[i] not in key_column_names: # add values to the correct row
                            index = joinedTable.column_names.index(other_table.column_names[i])
                            row[index] = otherRow[i]

        # add all remaining rows from the "right" table
        for row in other_table.data:
            newRow = ["NA"] * len(joinedTable.column_names) # create new row full of NA
            keys = other_table.getKeyValues(row, key_column_names)
            if keys not in seenKeys: # row keys are not already included
                for i in range(len(row)): # add to row at correct index
                        index = joinedTable.column_names.index(other_table.column_names[i])
                        newRow[index] = row[i]
                joinedTable.data.append(newRow) # add row to table
        # joinedTable.pretty_print()  
        return joinedTable
