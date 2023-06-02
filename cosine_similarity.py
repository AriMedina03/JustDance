import numpy as np

def cosine_similarity(matrix1, matrix2):
    # Compute the dot product of the two matrices
    dot_product = np.dot(matrix1, matrix2.T)
    
    # Calculate the magnitudes of each matrix
    magnitude1 = np.sqrt(np.sum(np.square(matrix1), axis=1))
    magnitude2 = np.sqrt(np.sum(np.square(matrix2), axis=1))
    
    # Compute the cosine similarity
    similarity = dot_product / np.outer(magnitude1, magnitude2)
    
    return similarity

# Example usage
matrix1 = np.array([[1, 2], [4, 5]])
matrix2 = np.array([[7, 8], [10, 11]])

similarity_matrix = cosine_similarity(matrix1, matrix2)
print(similarity_matrix)
