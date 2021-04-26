import finetune
from tqdm import tqdm

if __name__ =='__main__':
    for i in tqdm([0.0001,0.00001,0.001],desc="LR 4,5,3"):
        for j in tqdm([16,32],desc="BS 16 32,64"):
            finetune.RUN(j,i)