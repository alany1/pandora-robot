def point_aria_to_robot(point_aria, robot_aria_tf):
    """
    Transform a point from Aria to Robot coordinates.
    
    Args:
        point_aria (np.ndarray): The point in Aria coordinates.
        robot_aria_tf (np.ndarray): The transformation matrix from Robot to Aria coordinates.
        
    Returns:
        np.ndarray: The point in Robot coordinates.
    """
    # Convert the point to homogeneous coordinates
    point_aria_homogeneous = np.append(point_aria, 1)
    
    # Apply the transformation
    point_robot_homogeneous = robot_aria_tf @ point_aria_homogeneous
    
    # Convert back to 3D coordinates
    return point_robot_homogeneous[:3] / point_robot_homogeneous[3]
