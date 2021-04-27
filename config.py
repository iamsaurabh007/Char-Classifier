import string
symbols=list(string.printable[:94])
symbols.append(u"\u00A9")
symbols.append(u"\u2122")
symbols.append(" ")
num_classes=len(symbols)

data_dir_path="/home/ubuntu/data/ocr/out"
csv_path='/home/ubuntu/Character-Classifier/FC/hypergridcsv'
MODELCHECKPOINT_PATH="/home/ubuntu/data/ocr/InceptFCByParts"
device=None

#USED IN DATALOADER
batch_size=64
shuffle=True
num_workers=7
alpha=0.01         ####loss=cross entropy loss + alpha * similarity loss
#USED IN MODEL
learning_rate=0.001
num_epochs=500


#FOR INCEPTION ARCHITECTURE
channel=32

####MODELEVALUATION

#checkpath="/home/ubuntu/data/ocr/ModelResnetfinal/epoch-36.pt"
checkpath="/home/ubuntu/data/ocr/InceptFCByParts/CONVPART/epoch-20.pt"
pdfdata="/home/ubuntu/craft_benchmark/"
