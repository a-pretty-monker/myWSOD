if __name__ == '__main__':
    matrix = [[1,2,3],[2,3,4],[4,5,6],[5,6,7]]

    num = matrix[:,:]
    x = num > 0.8

    print(x)