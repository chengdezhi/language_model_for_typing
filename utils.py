import heapq
import numpy as np
class TopkHeap(object):
    def __init__(self, k):
        self.k = k
        self.data = []

    def Push(self, elem):
        if len(self.data) < self.k:
            heapq.heappush(self.data, elem)
        else:
            topk_small = self.data[0]
            if elem > topk_small:
                heapq.heapreplace(self.data, elem)

    def TopK(self):
       return [x for x in reversed([heapq.heappop(self.data) for x in xrange(len(self.data))])]


def get_topk_index(alist,k):
    th = TopkHeap(k)
    for i in alist:
        th.Push(i)
    res = th.TopK()
    sorted(res)
    index = []
    for j in res:
        index.append(np.where(alist==j)[0][0])
        #index.append(alist.index(j))
    return index
