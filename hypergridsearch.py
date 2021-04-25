import finetune
from tqdm import tqdm

if __name__ =='__main__':
    for i in tqdm([0.001,0.0001,0.000001,0.00001],desc="LR 3 4 6 5"):
        for j in tqdm([16,4,64,256],desc="BS 16 4 64 256"):
            finetune.RUN(j,i)