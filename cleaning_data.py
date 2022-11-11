import pandas as pd
import os
import string

#further clean

schizoaffective = []
schizophrenia = []
bipolar = []
control = []


def importPodcast1(file, tag):
  text = open("raw_data/1 - Podcast/"+file, "r")
  if tag == 'sE':
    cleaned = open("cleaned_data/schizophrenia.txt", "w")
  else:
    cleaned = open("cleaned_data/schizoaffective.txt", "w")
  host = text.readline().split()[0]
  control = open("cleaned_data/control.txt", "w")
  control.write(text.readline().replace("\n", " "))
  text.readline()
  while True:
    line = text.readline() 
    if len(line) == 0:
      break
    line2 = text.readline().replace("\n", " ")
    
    if line.split()[0] == host:
      control.write(line2)
    elif tag == 'sE':
      cleaned.write(line2)
    else:
      cleaned.write(line2)
    text.readline()
    
  text.close()
  cleaned.close()
  control.close()
      

def main():
  try:
    os.remove("cleaned_data/schizoaffective.txt")
  except:
    pass

  try:
    os.remove("cleaned_data/schizophrenia.txt")
  except:
    pass

  try:
    os.remove("cleaned_data/control.txt")
  except:
    pass

  path = os.getcwd()
  files = os.listdir(path+"/raw_data/1 - Podcast")
  for file in files:
    tag = file[0:2]
    if tag != 'xx':
      importPodcast1(file, tag)

def importSZ_Disucssion():
  file = open('raw_data/schizophrenia.com/DX.txt')
  file_data = file.read()
  print(file_data)


importSZ_Disucssion()