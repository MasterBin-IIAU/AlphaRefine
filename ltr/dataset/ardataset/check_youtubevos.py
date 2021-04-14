from ltr.dataset import Youtube_VOS
import random
dataset = Youtube_VOS()
num_inst = dataset.get_num_sequences()
idx = random.randint(0,num_inst)
dataset.get_sequence_info(idx)