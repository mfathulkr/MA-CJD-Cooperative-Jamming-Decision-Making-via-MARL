import numpy as np

def one_hot_encode(index, num_classes):
    """
    Creates a one-hot encoded vector.

    Args:
        index (int): The index to set to 1 (must be 0 <= index < num_classes).
        num_classes (int): The total number of classes (length of the output vector).

    Returns:
        np.ndarray: A numpy array of shape (num_classes,) with a 1 at the specified index and 0s elsewhere.
        
    Raises:
        ValueError: If index is out of bounds.
    """
    if not isinstance(index, int):
        raise TypeError(f"Index must be an integer, got {type(index)}")
    if not isinstance(num_classes, int):
        raise TypeError(f"num_classes must be an integer, got {type(num_classes)}")
    if num_classes <= 0:
         raise ValueError(f"num_classes must be positive, got {num_classes}")
    if not (0 <= index < num_classes): 
        raise ValueError(f'Index {index} out of bounds for {num_classes} classes')
        
    encoded = np.zeros(num_classes, dtype=np.float32) # Use float32 consistent with state vector
    encoded[index] = 1.0
    return encoded 