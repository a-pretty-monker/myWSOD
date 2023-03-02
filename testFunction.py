import numpy as np
if __name__ == '__main__':
    # matrix = np.array([[1,2,0],[2,3,4],[0,5,6],[0,6,7]]).astype(np.int32)
    #
    # num = matrix[:,:]
    # x = num > 0.8
    #
    # print(num[x])

    t = np.array([[1,0],[1,1],[0,1],[0,1]],dtype=np.int32)
    print(np.unique(t,axis=0))