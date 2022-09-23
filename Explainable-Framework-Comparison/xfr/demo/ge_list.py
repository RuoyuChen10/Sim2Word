import os
import random

with open("./List.txt", "r") as f:
    datas = f.read().split('\n')

VGG_dir = "/home/cry/data2/VGGFace2/train_align/"

for data in datas:
    try:
        probe = data.split(" ")[0]
        probe_id = probe.split("/")[0]

        path = os.path.join(VGG_dir,probe_id)

        mates = os.listdir(path)

        while True:
            mate_name = random.choice(mates)
            mate_name = probe_id + '/' + mate_name

            if mate_name != probe:
                break
        
        with open("test.txt","a") as f:
            f.write(probe+" "+mate_name+" "+data.split(" ")[1]+"\n")
    except:
        pass
