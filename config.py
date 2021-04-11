import string
symbols=list(string.printable[:94])
symbols.append(u"\u00A9")
symbols.append(u"\u2122")
symbols.append(" ")
num_classes=len(symbols)

data_dir_path="/home/ubuntu/data/ocr/out"
csv_path='/home/ubuntu/Character-Classifier/FC/hypergridcsv'
MODELCHECKPOINT_PATH="/home/ubuntu/data/ocr/ModelInceptfinal"
device=None

#USED IN DATALOADER
batch_size=128
shuffle=True
num_workers=6

#USED IN MODEL
learning_rate=0.001
num_epochs=500


#FOR INCEPTION ARCHITECTURE
channel=32

####MODELEVALUATION

#checkpath="/home/ubuntu/data/ocr/ModelResnetfinal/epoch-36.pt"
checkpath="/home/ubuntu/data/ocr/ModelInceptfinal/epoch-161.pt"
pdfdata="/home/ubuntu/data/ocr/kdeval/good/"
