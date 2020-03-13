import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as op
import torchvision
from torch.utils.data import TensorDataset, DataLoader
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import FileManager
import Classifier
import nltk
from collections import Counter
import numpy as np

def main():
    # Loading in data from files for training and test
    fileManager = FileManager.FileMngr("Program/data/cleanedData.csv")
    textForTraining = fileManager.getText()

    fileManagerTest = FileManager.FileMngr("Program/data/testData.csv")
    textForTest = fileManagerTest.getText()

    fileManager.checkFile()
    fileManagerTest.checkFile()
    # chars = set(''.join(text["tweet"]))
    # int2charDict = dict(enumerate(chars))
    # char2intDict = {char: ind for ind, char in int2charDict.items()}

    

    # text['tweet'] = text['tweet'].apply(lambda x : padding(x, longestTweet))

    # dict_size = len(char2intDict)
    # seq_len = longestTweet - 1
    # batch_size = len(tweets)
    # features = np.zeros((batch_size, seq_len, dict_size), dtype=np.float32)
    # print(features)
    # for i in range(len(text)):
    #     tweets[i] = [char2intDict[character] for character in tweets[i]]
    # print(tweets)
    # for i in range(batch_size):
    #     for u in range(seq_len):
    #         features[i, u, tweets[i][u]] = 1
    # print(features)

    #Setting tweets and classes/labels to variables
    trainingText = textForTraining['tweet']
    trainingLabel = textForTraining['class']

    testText = textForTest['tweet']
    testLabel = textForTest['class']

   

    # Creating a word dictionary using training data. This gives each word a numeric value. Also removes any word that is used less than once as it is likely to be irrelevent
    # in classification. Also this helps with things such as mispelling

    words = createDictionary(trainingText)

    # WordToNo and NoToWord do vice versa, translates between user and program
    wordToNoDict = {o:i for i,o in enumerate(words)}
    noToWordDict = {i:o for i,o in enumerate(words)}

    # Mapping words to their relevent numbers within the dictionary
    trainingText = wordMap(trainingText, wordToNoDict)
    testText = wordMap(testText, wordToNoDict)

     #Finding longest tweet to pad other tweets for consistency for batch processing
    longestTweet = len(max(textForTraining['tweet'], key=len))
    trainingText = pad_input(trainingText, longestTweet)
    testText = pad_input(testText, longestTweet)

    #Splitting test data into test and validation data. Validation used to test accuracy during training.
    splitNo = int(0.2 * len(trainingText))
    valText, testText = testText[:splitNo], testText[splitNo:]
    valLabel, testLabel = testLabel[:splitNo], testLabel[splitNo:]
 
    # trainingData = torch.from_numpy(features) 

    # Check if a gpu is available and setting it for use if it is. This allows for faster processing when training if it is available
    gpuAvailable = torch.cuda.is_available()

    if gpuAvailable:
        device = torch.device("cuda")
        print("GPU is available")
    else:
        device = torch.device("cpu")
        print("GPU not available, CPU used")

    device = torch.device("cpu")


    trainingData = TensorDataset(torch.from_numpy(np.asarray(trainingText)), torch.from_numpy(np.asarray(trainingLabel)))
    validationData = TensorDataset(torch.from_numpy(np.asarray(valText)), torch.from_numpy(np.asarray(valLabel)))
    testingData = TensorDataset(torch.from_numpy(np.asarray(testText)), torch.from_numpy(np.asarray(testLabel)))

    batchSize = 250

    trainingLoad = DataLoader(trainingData, shuffle = True, batch_size = batchSize, drop_last=True)
    validationLoad = DataLoader(validationData, shuffle = True, batch_size = batchSize, drop_last=True)
    testingLoad = DataLoader(testingData, shuffle = True, batch_size = batchSize, drop_last=True)
    


    vocabularySize = len(wordToNoDict) + 1
    output = 3
    embedding = 400
    hiddenDimension = 512
    layers = 2

    classifierModel = Classifier.HateSpeechDetector(device, vocabularySize, output, embedding, hiddenDimension, layers)
    classifierModel.to(device)

    trainClassifier(classifierModel, trainingLoad, validationLoad, device, batchSize)
    path = './Program/data/state_dict.pt'
    weight = torch.tensor([15389/3407, 15389/15389, 15389/800])
    criterion = nn.CrossEntropyLoss(weight=weight)
    #test(classifierModel, path, testingLoad, batchSize, device, criterion)

def trainClassifier(model, trainingData, validationData, device, batchSize):
    weight = torch.tensor([1.2, 1.0, 1.8])
    epochs = 2
    counter = 0
    testWithValiEvery = 10
    clip = 5
    valid_loss_min = np.Inf

    lr=0.005
    weight = torch.tensor([15389/3407, 15389/15389, 15389/800])
    criterion = nn.CrossEntropyLoss(weight=weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
   

    model.train()
   
    for i in range(epochs):
        
        h = model.init_hidden(batchSize, device)
       
        for inputs, labels in trainingData:
            h = tuple([e.data for e in h])
            inputs, labels = inputs.to(device), labels.to(device)
            model.zero_grad()

            output, h = model(inputs, h)

            loss = criterion(output.squeeze(), labels.float())
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            counter += 1
            print(counter)
        
            if counter%testWithValiEvery == 0:
                print("validating")
                val_h = model.init_hidden(batchSize, device)
                val_losses = []
                model.eval()
                for inp, lab in validationData:
                    val_h = tuple([each.data for each in val_h])
                    inp, lab = inp.to(device), lab.to(device)
                    out, val_h = model(inp, val_h)
                    val_loss = criterion(out.squeeze(), lab.float())
                    val_losses.append(val_loss.item())
                
                model.train()
                print("Epoch: {}/{}...".format(i+1, epochs),
                    "Step: {}...".format(counter),
                    "Loss: {:.6f}...".format(loss.item()),
                    "Val Loss: {:.6f}".format(np.mean(val_losses)))
                if np.mean(val_losses) <= valid_loss_min:
                    torch.save(model.state_dict(), './Program/data/state_dict.pt')
                    print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min,np.mean(val_losses)))
                    valid_loss_min = np.mean(val_losses)

def test(HateSpeechModel, path, testData, batchSize, device, criterion):
    HateSpeechModel.load_state_dict(torch.load(path))
    testLosses = []
    numCorrect = 0
    h = HateSpeechModel.init_hidden(batchSize, device) 
    HateSpeechModel.eval()
    for tweets, labels in testData:
        h = tuple([each.data for each in h])
        tweets, labels = tweets.to(device), labels.to(device)
        output, h = HateSpeechModel(tweets, h)
        testLoss = criterion(output.squeeze(), labels.float())
        testLosses.append(testLoss.item())
        prediction = torch.round(output.squeeze())
        print(prediction)
        correctTensor = prediction.eq(labels.float().view_as(prediction))
        correct = np.squeeze(correctTensor.cpu().numpy())
        numCorrect += np.sum(correct)

    print("Test loss: {:.3f}".format(np.mean(testLosses)))
    testAccuracy = numCorrect/len(testData.dataset)
    print("Test accuracy: {:.3f}%".format(testAccuracy*100))
    print(testLosses)

def wordMap(text, wordToNoDict):
    
    for i, sentence in enumerate(text):
        text[i] = [wordToNoDict[word] if word in wordToNoDict else 0 for word in sentence]
    return text

def pad_input(sentences, seq_len):
    features = np.zeros((len(sentences), seq_len),dtype=int)
    for i, tweet in enumerate(sentences):
        if len(tweet) != 0:
            features[i, -len(tweet):] = np.array(tweet)[:seq_len]
    return features

def createDictionary(tweets):
    words = Counter()
    print("Creating Dictionary")
    for i, sentence in enumerate(tweets):
        tweets[i] = []
        for word in nltk.word_tokenize(sentence):  # Tokenizing the words
            words.update([word])  # Converting all the words to lowercase
            tweets[i].append(word)
        if i%20000 == 0:
            print(str((i*100)/len(tweets)) + "% done")
    print("100% done")
    words = {k:v for k,v in words.items() if v>1}
    words = sorted(words, key=words.get, reverse=True)
    words = ['_PAD','_UNK'] + words
    return words

    

def padding(text, longestTweet):
    while len(text)<longestTweet:
        text += ' '
    return text


def cleanDataFile():
    fileManager = FileManager.FileMngr("Program/data/labeled_data.csv")
    fileManager.loadFile()
    fileManager.checkFile()
    fileManager.chartFile()
    fileManager.cleanFile()
    fileManager.createFile()



main()
