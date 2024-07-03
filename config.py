CAMERA_DISTANCE_TO_TABLE = 100
CAMERA_WIDTH = 1024
CAMERA_HEIGHT = 576

TABLE_WIDTH = 1024
TABLE_HEIGHT = 576

GOAL_DEPTH = 20
GOAL_AREA_HEIGHT = 80

SATURATION_SCALE = 2.0
CONTRAST_SCALE = 2.0

LOWER_COLOR = [20, 100, 100]
UPPER_COLOR = [30, 255, 255] 

"""    
Parameters:
CAMERA_DISTANCE_TO_TABLE (float): The height of the camera above the table in centimeters.
CAMERA_WIDTH (int): The width of the resized camera in pixel.
CAMERA_HEIGHT (int): The height of the resized camera in pixel.

TABLE_WIDTH (int): The width of the table in pixels.
TABLE_HEIGHT (int): The height of the table in pixels.

GOAL_DEPTH (int): The depth/width of each goal on virtual table in pixel.
GOAL_AREA_HEIGHT (int): The height of each goal on virtual table in pixel.

SATURATION_SCALE (float): The multiplication factor of saturation.
CONTRAST_SCALE (float): The multiplication factor of contrast.

LOWER_COLOR (int array): The lower end of the desired color spectrum of the tracking object in the HSV color space.
UPPER_COLOR (int array): The upper end of the desired color spectrum of the tracking object in the HSV color space.
"""
