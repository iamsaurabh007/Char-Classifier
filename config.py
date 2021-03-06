import string
symbols=list(string.printable[:94])
symbols.append(u"\u00A9")
symbols.append(u"\u2122")
symbols.append(" ")
num_classes=len(symbols)

data_dir_path="/home/ubuntu/craft_benchmark/fine_tune/fine_tune_sampled_ds"
csv_path='/home/ubuntu/Character-Classifier/FC/hypergridcsv'
MODELCHECKPOINT_PATH="/home/ubuntu/data/ocr/ModelInceptfinal"
device=None

#USED IN DATALOADER
batch_size=128
shuffle=True
num_workers=6

#USED IN MODEL
learning_rate=0.0001
num_epochs=500


#FOR INCEPTION ARCHITECTURE
channel=32

####MODELEVALUATION

#checkpath="/home/ubuntu/data/ocr/ModelResnetfinal/epoch-36.pt"
checkpath="/home/ubuntu/data/ocr/Model_Weights/ModelInceptfinal/FineTune2/fine-epoch-37.pt"
pdfdata="/home/ubuntu/craft_benchmark/"
#pdfdata="/home/ubuntu/data/ocr/kdeval/good/"









####################################################################

####Requires a folder with images and json folder into it
#testfiles=["/home/ubuntu/data/ocr/kdeval/good/","/home/ubuntu/data/ocr/kdeval/bad/",\
#   "/home/ubuntu/data/ocr/kdeval/average/"] 


testfiles=["good","bad","average"]
testpath="/home/ubuntu/craft_benchmark/"
weightfilepath="/home/ubuntu/data/ocr/Model_Weights/ModelInceptfinal/FineTune2"
testweights=["fine-epoch-37.pt","fine-epoch-346.pt","fine-epoch-494.pt"]
