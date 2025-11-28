import torch, json, os
import numpy
from sklearn.cluster import KMeans
import random
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import AdamW
from tqdm import tqdm

""" This is the code used for training models, it has been used in many different forms,
    currently it uses the google mt5 base model, it teaches it to associate specific words with labels,
    the idea is that it will be able to generate sentences that contain certain traits which can be
    used to create targeted tasks.
"""


class MODELSETUP:
    def __init__(self):
        self.device = None
        self.saveDir = "TestSave5"
        self.loadedOldModel = False
        
        self.unprocessedDataDir = "Labeled_Unzipped"
        self.processedDataDir = "Processed_Data"

        self.finalClusterMap = dict()
        self.dataUseSize = 0.50 # We were only able to use half of our already limited size data, as training was a very long process even with our low end CUDA setup
        self.randomSeed = 100

        modelName = "google/mt5-base"
        
        if not os.path.isdir(self.saveDir):
            print(f"Loading fresh model: {modelName}")
            self.TOKENIZER = AutoTokenizer.from_pretrained(modelName, trust_remote_code=True, use_fast = False)
            self.MODEL = AutoModelForSeq2SeqLM.from_pretrained(modelName, trust_remote_code=True)
            self.MODEL.gradient_checkpointing_enable() 
        else:
            print(f"Loading model from {self.saveDir}...")
            self.TOKENIZER = AutoTokenizer.from_pretrained(self.saveDir)
            self.MODEL = AutoModelForSeq2SeqLM.from_pretrained(self.saveDir)
            self.MODEL.gradient_checkpointing_enable()
            self.loadedOldModel = True
            
            mapPath = os.path.join(self.saveDir, "cluster_map.json")
            if os.path.exists(mapPath):
                print("Loading saved Cluster Map...")
                with open(mapPath, "r", encoding="utf-8") as f:
                    self.finalClusterMap = json.load(f)
            else:
                print("No map found!")

    def cleanFiles(self): # This is a part of the preprocessing setup, used to clean the files at the early stage of the task
        os.makedirs(self.processedDataDir, exist_ok=True)
        print("\nStarting to clean files...")

        for fname in os.listdir(self.unprocessedDataDir):
            outputName = fname.rsplit(".")[0] + ".json"
            
            if not fname.endswith(".jsonl") or os.path.exists(os.path.join(self.processedDataDir, outputName)):
                continue

            cleanData = list()
            all_raw_data = []

            try:
                with open((self.unprocessedDataDir + "/" + fname), "r", encoding = "utf-8") as file:
                    for line in file:
                        raw_line_data = json.loads(line)
                        all_raw_data.append(raw_line_data) 
                
                for rawData in all_raw_data:
                    if "parsed" in rawData:
                        for item in rawData["parsed"]:
                            cleanData.append({
                                "sentence": item["input"],
                                "words": item["labels"]
                            })

                with open(os.path.join(self.processedDataDir, outputName), "w", encoding = "utf-8") as file:
                    json.dump(cleanData, file, ensure_ascii=False, indent=2)
                    print("Finished cleaning file ", fname)
            
            except Exception as e:
                print(f"Skipping file {fname} cause of error {e}")

        print("Finished cleaning all files...\n")

    def createClusters(self, lenClusters = 100, batchSize = 100): # This method creates a cluster map based on patterns found by the T5 in text
        wordSet = set()

        for fname in os.listdir(self.processedDataDir):
            if not fname.endswith(".json"):
                continue
            with open(os.path.join(self.processedDataDir, fname), "r", encoding = "utf-8") as f:
                cleanedData = json.load(f)
                for item in cleanedData:
                    for word in item["words"]:
                        if word.get("text"):
                            wordSet.add(word["text"])

        wordList = sorted(list(wordSet))
        if not wordList:
            print("\nError no wordlist\n")
            return

        if not self.device:
            self.deviceSelection(override=None)

        encoder = self.MODEL.get_encoder()
        self.MODEL.eval()
        allEmbeddings = list()

        print(f"Generating embeddings for {len(wordList)} unique words...")
        with torch.no_grad():
            for i in tqdm(range(0, len(wordList), batchSize)):
                wordsOfBatch = wordList[i:i + batchSize]
                inputs = self.TOKENIZER(
                    wordsOfBatch,
                    return_tensors = "pt",
                    padding = True,
                    truncation = True,
                    max_length = 512
                ).to(self.device)

                outputs = encoder(input_ids = inputs.input_ids, attention_mask = inputs.attention_mask)
                embeddings = outputs.last_hidden_state.mean(dim = 1)
                allEmbeddings.append(embeddings.cpu().numpy())

        allEmbeddings = numpy.vstack(allEmbeddings)

        print("Clustering embeddings...")
        Kmeans = KMeans(n_clusters = lenClusters, random_state = 42, n_init = 10)
        Kmeans.fit(allEmbeddings)

        self.finalClusterMap = dict(zip(wordList, Kmeans.labels_.tolist()))
        print("Finished clustering...")

    def makePairs(self): # This function makes training pairs. Essentially, one word can be associated with more than one label
        self.allTrainingPairs = []
        print("Generating training pairs...")

        for fname in os.listdir(self.processedDataDir):
            if not fname.endswith(".json"):
                continue

            with open(os.path.join(self.processedDataDir, fname), "r", encoding = "utf-8") as f:
                cleanedData = json.load(f)

            for item in cleanedData:
                wordsList = [w.get("text") for w in item["words"] if w.get("text")]
                targetSentence = " ".join(wordsList)
                
                if not targetSentence:
                    continue

                for word in item["words"]:
                    wordTXT = word.get("text")
                    
                    if wordTXT:
                        self.allTrainingPairs.append({
                            "input": f"construct sentence with: {wordTXT}",
                            "target": targetSentence
                        })

        totalPairsCount = len(self.allTrainingPairs)
        targetPairsCount = int(totalPairsCount * self.dataUseSize)

        if targetPairsCount > 0 and targetPairsCount < totalPairsCount:
            print(f"Reducing dataset from {totalPairsCount} to {targetPairsCount}...")
            
            random.seed(self.randomSeed)
            random.shuffle(self.allTrainingPairs)

            guaranteed_pairs = []
            remaining_pairs = []
            seen_words = set()

            for pair in self.allTrainingPairs:
                input_word = pair["input"].split("with: ")[-1]
                
                if input_word not in seen_words:
                    seen_words.add(input_word)
                    guaranteed_pairs.append(pair)
                else:
                    remaining_pairs.append(pair)

            print(f"Preserved {len(seen_words)} unique vocabulary words...")

            slots_needed = targetPairsCount - len(guaranteed_pairs)
            if slots_needed > 0:
                sampled_rest = random.sample(remaining_pairs, min(len(remaining_pairs), slots_needed))
                self.allTrainingPairs = guaranteed_pairs + sampled_rest
            else:
                self.allTrainingPairs = guaranteed_pairs

            random.shuffle(self.allTrainingPairs)
        
        print(f"Final training dataset size: {len(self.allTrainingPairs)}")

    def deviceSelection(self, override):
        if override is None:
            if torch.cuda.is_available():
                device = torch.device("cuda")
                print(f"Using GPU: {torch.cuda.get_device_name(0)}")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = torch.device("mps")
                print("Using MPS")
            else:
                device = torch.device("cpu")
                print("Using CPU")
        else:
            device = torch.device(override)
            print("Overide device:", device)
        
        self.device = device
        self.MODEL.to(self.device)

    def tokenizeProcDat(self): # Tokenization function
        print(f"Tokenizing {len(self.allTrainingPairs)} pairs...")

        inputData = [pair["input"] for pair in self.allTrainingPairs]
        targetData = [pair["target"] for pair in self.allTrainingPairs]

        self.tokenizedData = self.TOKENIZER(
            inputData,
            text_target=targetData,
            padding= "longest",
            truncation=True,
            return_tensors="pt"
        )

        labels = self.tokenizedData["labels"]
        labels[labels == self.TOKENIZER.pad_token_id] = -100
        
        print("Tokenization complete...")

    def batchNstuff(self, batch_size = 1, learning_rate = 2e-5): 
        if not self.device:
            self.deviceSelection(override=None)
        
        dataset = TensorDataset(
            self.tokenizedData["input_ids"],
            self.tokenizedData["attention_mask"],
            self.tokenizedData["labels"]
        )

        self.DL = DataLoader(dataset, shuffle = True, batch_size = batch_size)
        
        self.OPTIMIZER = AdamW(self.MODEL.parameters(), lr = learning_rate)
        
        print("Dataloader and optimizer ready...")

    def trainLoop(self, epochs= 5, accumulation_steps = 16): # Main training loop
        print("Starting training loop...")
        self.MODEL.train()
        self.OPTIMIZER.zero_grad()

        optimizer = self.OPTIMIZER

        for epoch in range(epochs):
            print(f"Epoch {epoch + 1} / {epochs}")
            totalLoss = 0

            for step, batchData in enumerate(tqdm(self.DL)):
                batch = {
                    "input_ids": batchData[0].to(self.device),
                    "attention_mask": batchData[1].to(self.device),
                    "labels": batchData[2].to(self.device)
                }

                outputs = self.MODEL(**batch)
                loss = outputs.loss
                loss = loss / accumulation_steps
                
                loss.backward()
                totalLoss += loss.item()

                if (step + 1) % accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.MODEL.parameters(), max_norm = 1.0) 
                    
                    optimizer.step()
                    
                    optimizer.zero_grad()

            if (len(self.DL) % accumulation_steps) != 0:
                torch.nn.utils.clip_grad_norm_(self.MODEL.parameters(), max_norm = 1.0)
                
                optimizer.step()
                
                optimizer.zero_grad()

            if len(self.DL) > 0:
                avgLoss = (totalLoss * accumulation_steps) / len(self.DL) 
                print(f"Epoch {epoch + 1} finished. Avg Loss: {avgLoss:.4f}")

        print("Training finished...")

    def saveModel(self):
        print("Saving model...")
        os.makedirs(self.saveDir, exist_ok = True)

        self.MODEL.save_pretrained(self.saveDir)
        self.TOKENIZER.save_pretrained(self.saveDir)

        mapPath = os.path.join(self.saveDir, "cluster_map.json")
        with open(mapPath, "w", encoding = "utf-8") as f:
            json.dump(self.finalClusterMap, f, indent = 4, ensure_ascii = False)
        
        print(f"Saved to {self.saveDir}")

def CreateObject(): # Creating the training object
    print("1 = Yes, else = No\n")
    choiceCLEANFILES = input("Clean raw labeled files? ")
    choiceOVERRIDE = input("Override device choice? ")

    modelObj = MODELSETUP()

    if choiceCLEANFILES == "1":
        modelObj.cleanFiles()

    if choiceOVERRIDE == "1":
        choiceDEVICE = input("1 = cuda, 2 = mps, 3 = cpu: ")
        devices = {"1": "cuda", "2": "mps", "3": "cpu"}
        device = devices.get(choiceDEVICE)
    else:
        device = None

    modelObj.deviceSelection(override=device)

    if not modelObj.loadedOldModel:
        modelObj.createClusters()
        modelObj.makePairs()
        modelObj.tokenizeProcDat()
        modelObj.batchNstuff(batch_size=1)
        modelObj.trainLoop(epochs=5, accumulation_steps=16)
        modelObj.saveModel()
    else:
        print("Skipping training, model is already loaded...")

if __name__ == "__main__":
    CreateObject()
