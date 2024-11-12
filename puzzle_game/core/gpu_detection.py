import cv2

def is_gpu_enabled():
    ''' Verifica que la ejecución se esté realizando por GPU, o se permita la ejecución por GPU'''
    return cv2.cuda.getCudaEnabledDeviceCount() > 0
