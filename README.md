# K-Nearest Neighbors From Scratch

A complete implementation of the K-Nearest Neighbors (KNN) classification algorithm built from scratch using only NumPy. Features both 2D and 3D visualization capabilities with custom distance calculations and classification logic, demonstrating the fundamental concepts behind this popular machine learning algorithm.

## Author
**Macha Praveen**

## Overview

This project implements the K-Nearest Neighbors algorithm without using external machine learning libraries, providing a clear understanding of how KNN works under the hood. The implementation includes Euclidean distance calculations, majority voting for classification, and comprehensive visualizations in both 2D and 3D space.

## Features

- **Pure Python Implementation**: Built from scratch without scikit-learn or other ML libraries
- **Euclidean Distance Calculation**: Custom distance metric implementation
- **Flexible K Value**: Configurable number of nearest neighbors
- **2D and 3D Visualization**: Interactive plots with distance lines
- **Educational Focus**: Clear, commented code for learning purposes
- **Custom Dataset Support**: Easy to modify for different classification problems

## Algorithm Implementation

### Core KNN Class
```python
class KNearestNeighbors:
    def __init__(self, k=3):
        self.k = k
        self.points = None

    def fit(self, points):
        self.points = points

    def predict(self, new_point):
        distances = []
        
        for category in self.points:
            for point in self.points[category]:
                distance = euclidean_distance(point, new_point)
                distances.append([distance, category])
        
        categories = [category[1] for category in sorted(distances)[:self.k]]
        result = Counter(categories).most_common(1)[0][0]
        return result
```

### Distance Calculation
```python
def euclidean_distance(p, q):
    return np.sqrt(np.sum((np.array(p) - np.array(q)) ** 2))
```

## Example Usage

### 2D Classification Example
```python
# Sample data points
points = {'blue': [[2,4], [1,3], [2,3], [3,2], [2,1]],
          'orange': [[5,6], [4,5], [4,6], [6,6], [5,4]]}

# Point to classify
new_point = [3,3]

# Create and train classifier
clf = KNearestNeighbors(k=3)
clf.fit(points)

# Make prediction
prediction = clf.predict(new_point)
print(f"New point {new_point} is classified as: {prediction}")
```

### 3D Classification Example
```python
# 3D data points
points = {'blue': [[2, 4, 3], [1, 3, 5], [2, 3, 1], [3, 2, 3], [2, 1, 6]],
          'orange': [[5, 6, 5], [4, 5, 2], [4, 6, 1], [6, 6, 1], [5, 4, 6], [10, 10, 4]]}

new_point = [3, 3, 4]

clf = KNearestNeighbors(k=3)
clf.fit(points)
prediction = clf.predict(new_point)
```

## Visualization Features

### 2D Visualization
- **Data Points**: Blue and orange clusters with distinct colors
- **New Point**: Star marker showing the point to be classified
- **Distance Lines**: Dashed lines showing distances to all training points
- **Dark Theme**: Professional black background with colored points

### 3D Visualization  
- **3D Scatter Plot**: Three-dimensional representation of data points
- **3D Distance Lines**: Connections between new point and all training data
- **Interactive Rotation**: Full 3D visualization with matplotlib
- **Color Coding**: Consistent color scheme for different classes

## Algorithm Steps

1. **Distance Calculation**: Compute Euclidean distance from new point to all training points
2. **Sort by Distance**: Order all distances from smallest to largest
3. **Select K Neighbors**: Choose the K closest points
4. **Majority Vote**: Count class frequencies among K neighbors
5. **Classify**: Assign the most frequent class to the new point

## Mathematical Foundation

### Euclidean Distance Formula
For points p = (p₁, p₂, ..., pₙ) and q = (q₁, q₂, ..., qₙ):

```
d(p,q) = √[(p₁-q₁)² + (p₂-q₂)² + ... + (pₙ-qₙ)²]
```

### Classification Rule
Given K nearest neighbors with classes c₁, c₂, ..., cₖ:

```
predicted_class = argmax(count(cᵢ)) for i ∈ {1,2,...,k}
```

## Installation

### Prerequisites
- Python 3.7+
- NumPy
- Matplotlib
- Collections (built-in Python module)

### Dependencies
```bash
pip install numpy matplotlib
```

## Usage

### Running the Example
```bash
python main.py
```

This will:
1. Run 2D classification example
2. Display 2D visualization plot
3. Run 3D classification example  
4. Display 3D visualization plot

### Custom Implementation
```python
import numpy as np
from collections import Counter

# Your custom dataset
custom_points = {
    'class_A': [[1,1], [2,1], [1,2]],
    'class_B': [[4,4], [5,4], [4,5]]
}

# Initialize classifier
knn = KNearestNeighbors(k=3)
knn.fit(custom_points)

# Classify new points
test_points = [[2,2], [3,3], [5,5]]
for point in test_points:
    prediction = knn.predict(point)
    print(f"Point {point} -> Class: {prediction}")
```

## Project Structure

```
K-Nearest Neighbors From Scratch/
├── README.md
├── main.py                     # Complete implementation
├── sample_2d_data.py          # 2D example data
└── sample_3d_data.py          # 3D example data
```

## Algorithm Complexity

### Time Complexity
- **Training**: O(1) - just stores the training data
- **Prediction**: O(n×d) where n = training points, d = dimensions
- **Overall**: O(n×d) per prediction

### Space Complexity
- **Training Data Storage**: O(n×d)
- **Distance Calculations**: O(n) temporary storage
- **Overall**: O(n×d)

## Advantages and Limitations

### Advantages
- **Simple and Intuitive**: Easy to understand and implement
- **No Assumptions**: Makes no assumptions about data distribution  
- **Versatile**: Works for both classification and regression
- **Adaptive**: Automatically adjusts to local patterns

### Limitations
- **Computational Cost**: Expensive for large datasets
- **Sensitive to Scale**: Features with larger scales dominate
- **Curse of Dimensionality**: Performance degrades in high dimensions
- **Memory Requirements**: Must store entire training dataset

## Educational Value

This implementation is designed for learning purposes and demonstrates:

1. **Algorithm Fundamentals**: Core concepts without library abstractions
2. **Distance Metrics**: How similarity is measured in feature space
3. **Classification Logic**: Majority voting mechanism
4. **Visualization Techniques**: Both 2D and 3D plotting
5. **Code Organization**: Clean, readable Python structure

## Extensions and Improvements

### Possible Enhancements
- **Different Distance Metrics**: Manhattan, Minkowski, Cosine similarity
- **Weighted KNN**: Distance-weighted voting
- **Cross-Validation**: K-fold validation for optimal K selection
- **Feature Scaling**: Normalization and standardization
- **Efficient Search**: KD-trees or Ball trees for faster neighbor search

### Advanced Features
- **Regression Support**: Extend to KNN regression
- **Dynamic K Selection**: Automatic K optimization
- **Outlier Detection**: Identify anomalous data points
- **Dimensionality Reduction**: PCA integration for high-dimensional data

## License

This project is open-source and available under the MIT License.
"# KNN-Navigator-Applied-Machine-Learning-Tool" 
