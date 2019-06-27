import json
import os
import re
from os.path import join
import numpy as np
import matplotlib.pyplot as plt
#plt.interactive(False)

results_path = "./models/Round One"
integer_regex = "^[-+]?[0-9]+$"
pattern = re.compile(integer_regex)

filesDirectory = os.listdir(results_path)

accuracies = np.zeros(len(filesDirectory), dtype=list)
losses = np.zeros(len(filesDirectory), dtype=list)
lres = np.zeros(len(filesDirectory), dtype=list)
fnames = np.zeros(len(filesDirectory), dtype=str)

for idx, filename in enumerate(sorted(filesDirectory)):
    if(filename.endswith('.json')) and pattern.match(filename.replace('.json','')):
        with open(join(results_path, filename), 'r') as reader:
            json_data = json.load(reader)
            # if json_data['test_accuracy'] >= 0.12987012987012986:
            #     print('#########################################')
            #     print('Test acurracy: ', json_data['test_accuracy'])
            #     for key, param in json_data['params'].items():
            #         print(key, '                ', param)
            accuracies[idx] = json_data['history']['acc']
            losses[idx] = json_data['history']['loss']
            lres[idx] = json_data['history']['lr']
            fnames[idx] = filename

# print(len(accuracies))
# for idx, acc in enumerate(accuracies):
#     plt.plot(acc, label=idx)
#     if idx == 23:
#         print(acc)
#         print(fnames[idx])

print(len(lres))
for idx, acc in enumerate(lres):
    plt.plot(acc, label=idx)


plt.title('lr')
plt.ylabel('ylabel')
plt.xlabel('xlabel')
plt.legend(prop={'size': 6})
# for loss in losses:
#     plt.plot(loss)

plt.show()

