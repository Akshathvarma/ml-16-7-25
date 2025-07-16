import pandas as pd
import numpy as np

excel_file_path = "ml_lab1\Lab Session Data.xlsx"

df = pd.read_excel(excel_file_path,sheet_name='Purchase data')

print(df.head())
# Access a specific column
column_data = df['Customer']

print(column_data)

matrixA = df.iloc[0:11,1:4].values
print(matrixA)

matrixC=df.iloc[0:11,4:5]
print(matrixC)

print(matrixA.shape)

print(matrixC.shape)

print("the rank of matrix a is " , np.linalg.matrix_rank(matrixA))

print("the pinv of the matrix is")
print(np.linalg.pinv(matrixA))

