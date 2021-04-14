import os

lasot_test_path = '/media/masterbin-iiau/EAGET-USB/LaSOT_Test'
result_path = 'lasot_test.txt'
f = open(result_path, 'w')
sequence_list = os.listdir(lasot_test_path)
for idx, seq in enumerate(sequence_list):
    f.write(seq+'\n')
f.close()