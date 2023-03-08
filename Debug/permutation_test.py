import numpy as np
from copy import deepcopy


def permutation(given_idx: np.ndarray, max_order: int):
    # tmp_order is the 
    tmp_order = 1
    total_list = []
    pre_list = [[]]
    tmp_list = []
    while tmp_order <= max_order:
        for pre in pre_list:
            # print(f"The pre is {pre}")
            # Give a new memory to the pre
            pure_pre = deepcopy(pre)
            for idx in given_idx:
                # TODO Bug is here, First should prevent,
                if len(pre) > 0 and idx != pure_pre[-1]:
                    # This branch is for list length longer than 0, not include 0.
                    # print("Go to this branch")
                    # print(f"The idx is {idx}")
                    pre.append(idx)
                    # print(f"This time list to cache is {pre}")
                    # append or not need to verify
                    tmp_list.append(pre)
                    pre = deepcopy(pure_pre)
                elif len(pre) == 0:
                    # This branch is for those list length is 0.
                    # print(f"The idx is {idx}")
                    pre.append(idx)
                    # print(f"This time list to cache is {pre}")
                    tmp_list.append(pre)
                    pre = deepcopy(pure_pre)
        total_list.extend(tmp_list)
        pre_list = tmp_list
        tmp_list = []
        tmp_order += 1

    return total_list


if __name__ == '__main__':
    import time
    # plane index
    g_idx = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    # max order
    m_order = 5
    st_time = time.time()
    pm_list = permutation(g_idx, m_order)
    ed_time = time.time()
    print(f"The amount of possible route is {len(pm_list)}")
    print(f"Time Use: {ed_time-st_time}")
    print(pm_list[0:100])
