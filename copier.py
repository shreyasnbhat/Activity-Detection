import os
import shutil

BASE = 'Test'

classes = ['Kicking', 'Riding-Horse', 'Running', 'SkateBoarding',
           'Swing-Bench', 'Lifting', 'Swing-Side', 'Walking',
            'Golf-Swing']

dest = {}
dest['Kicking-Side'] = 'Kicking-Front'
dest['Kicking-Front'] = 'Kicking-Front'
dest['Golf-Swing-Side'] = 'Golf-Swing-Back'

print(sorted(os.listdir(BASE)))

for i in ['Kicking-Side']:
    last_dir = sorted(os.listdir(BASE + '/Kicking-Front'))[-1]
    print(last_dir)
    to_move = sorted(os.listdir(BASE + '/' + i))
    desti = [str(0)*(3-len(str(int(last_dir) + i + 1))) + str(int(last_dir) + i + 1) for i in range(len(to_move))]
    print(desti)

    for k in range(len(to_move)):
        print(to_move[k])
        print(BASE + '/' + i + '/' + to_move[k])
        print(BASE + '/' + dest[i] + '/' + desti[k])
        shutil.copytree(BASE + '/' + i + '/' + to_move[k],BASE + '/' + dest[i] + '/' + desti[k])
