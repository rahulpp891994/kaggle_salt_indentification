import numpy as np
ps=np.array([10,10])
cs=np.array([6,6])
ch_mid_idx=np.array([int((ps[0]-cs[0])/2),int((ps[0]-cs[0])/2)+cs[0]])

child_1=np.zeros(cs)+1
child_2=np.zeros(cs)+1
child_3=np.zeros(cs)+1
child_4=np.zeros(cs)+1
child_mid=np.zeros(cs)+1


all_child_array=np.zeros(ps)


"""
here we can add child as output of final layer of auto encoder 
child_1= "prediction from child_1"
"all_child_array" can be used as count matrx for that speicific parent_shape i.e for averaging purposes
we can also create similar array from prediction 
"""


all_child_array[:cs[0],:cs[0]]=all_child_array[:cs[0],:cs[0]]+child_1
all_child_array[-cs[0]:,:cs[0]]=all_child_array[-cs[0]:,:cs[0]]+child_2
all_child_array[:cs[0],-cs[0]:]=all_child_array[:cs[0],-cs[0]:]+child_3
all_child_array[-cs[0]:,-cs[0]:]=all_child_array[-cs[0]:,-cs[0]:]+child_4

all_child_array[ch_mid_idx[0]:ch_mid_idx[1],ch_mid_idx[0]:ch_mid_idx[1]]=all_child_array[ch_mid_idx[0]:ch_mid_idx[1],
                                                                                         ch_mid_idx[0]:ch_mid_idx[1]]+child_mid

final_pred =all_child_array/all_child_array
