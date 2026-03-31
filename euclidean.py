import math


def euclidean_distance(x1,x2,y1,y2):
    return math.sqrt((x2-x1)**2 + (y2-y1)**2)
    
if __name__ == "__main__":
    x1 = 2
    x2 = 5
    y1 = 4
    y2 = 7

    result = euclidean_distance(x1,x2,y1,y2)
    print(result)