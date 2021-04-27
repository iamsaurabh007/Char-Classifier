import string
symbols=list(string.printable[:94])
symbols.append(u"\u00A9")
symbols.append(u"\u2122")
symbols.append(" ")
num_classes=len(symbols)

#data_dir_path="/home/ubuntu/data/ocr/out"
data_dir_path="/home/ubuntu/craft_benchmark/fine_tune/fine_tune_sampled_ds"
csv_path='/home/ubuntu/Character-Classifier/FC/hypergridcsv'
MODELCHECKPOINT_PATH="/home/ubuntu/data/ocr/ModelInceptTripletrun1"
device=None

#USED IN DATALOADER
batch_size=16
shuffle=True
num_workers=7
alpha=0.01         ####loss=cross entropy loss + alpha * similarity loss
#USED IN MODEL
learning_rate=0.00001
num_epochs=500


#FOR INCEPTION ARCHITECTURE
channel=32

####MODELEVALUATION

#checkpath="/home/ubuntu/data/ocr/ModelResnetfinal/epoch-36.pt"
#checkpath="/home/ubuntu/data/ocr/ModelInceptTripletrun1/epoch-100.pt"
checkpath="/home/ubuntu/data/ocr/ModelInceptTripletrun1/FineTune/fine-epoch-49.pt"
pdfdata="/home/ubuntu/craft_benchmark/fine_tune/"



####Requires a folder with images and json folder into it
#testfiles=["/home/ubuntu/data/ocr/kdeval/good/","/home/ubuntu/data/ocr/kdeval/bad/",\
#   "/home/ubuntu/data/ocr/kdeval/average/"] 

weightfilepath="/home/ubuntu/data/ocr/ModelInceptTripletrun1/FineTune/"


testfiles=["good","bad","average"]
testpath="/home/ubuntu/craft_benchmark/"
testweights=["fine-epoch-51.pt","fine-epoch-109.pt","fine-epoch-39.pt"]

