from random import random
import torch
import numpy as np


# lst = [i for i in range(10)]
# print(lst)
# lst_iou = np.random.rand(10**2).reshape(10, 10)
# lst_iou = np.triu(lst_iou)
# lst_iou += lst_iou.T - np.diag(lst_iou.diagonal())

# lst_scr = np.random.rand(10)

# for i in range(len(lst)):
#     for j in range(i+1, len(lst)):
#         if lst_scr[i] < lst_scr[j]:
#             tmp = lst[j]
#             lst[j] = lst[i]
#             lst[i] = tmp
#             tmp = lst_scr[j]
#             lst_scr[j] = lst_scr[i]
#             lst_scr[i] = tmp

# print(lst)

# for i in range(len(lst)):
#     if lst[i] != -1:
#         for j in range(len(lst)):
#             if lst[j] != -1 and lst_iou[i, j] > 0.5 and j!=i:
#                 lst[j] = -1

# for i in lst: 
#     if i != -1:
#         print(i)

a = torch.Tensor([3,4,5,6])
b = a
 