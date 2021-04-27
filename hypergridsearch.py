
import fc_train
from tqdm import tqdm

if __name__ =='__main__':
    for i in tqdm([0.001,0.0001,0.00001],desc="LR 3 4 6 5"):
        for j in tqdm([8,16],desc="BS 16 4 64 256"):
            fc_train.RUN(j,i)