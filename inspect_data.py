import numpy as np
from tqdm import tqdm

def calculate_correlation(x, y):
    return np.corrcoef(x, y)[0, 1]

data = np.loadtxt('scirob_submission/Model_Learning/data/CRT/old/train_data_step1.csv', delimiter=',')
print('Shape of the training data: ', data.shape)

correlation_matrix = np.zeros((data.shape[0], 4))

for i in tqdm(range(data.shape[0] - 4)):
    for j in range(1, 5):
        correlation_matrix[i, j - 1] = calculate_correlation(data[i, :], data[i+j, :])

matrix = np.array(correlation_matrix)
np.savetxt('scirob_submission/Model_Learning/inspection.csv', matrix, delimiter=',')




