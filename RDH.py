import cv2
import pandas as pd
import numpy as np

import sqlite3
import pandas as pd
from numba.typed import List

import pytesseract


"""# Embed Function"""

#https://www.tutorialspoint.com/sqlite/sqlite_create_database.htm


"""# Run Embed"""



class Embedding(object):
  def __init__(self, imgPath, id, number):
    self.img = cv2.imread(imgPath)
    self.id = id
    self.number = number
    self.block = []
    self.difpixels = []
    self.listL = []
    self.listS = []
    self.data = []
    self.h, self.w = 792, 1122


  def toModthree(value):
    if (value%3 == 1):
      value-=1
    elif(value%3 == 2):
      value+=1
    return value

  def SubBlockImage(self, height, width):
    for i in range(height):
      k=0
      for j in range(3,width+3, 3):
        self.block.append(self.img[i,k:j])
        k+=3
    self.block = np.asarray(self.block)

  def binarySearch (self, arr, l, r, x):
    if r >= l:
        mid = l + (r - l) // 2
        if arr[mid] == x:
            return True
        elif arr[mid] > x:
            return self.binarySearch(arr, l, mid-1, x)
        else:
            return self.binarySearch(arr, mid + 1, r, x)
    else:
        return False


  def OverUnder_Flow_Handle(self, idx):
    for j in range(len(self.block)):
      k = 0
      p = 0
      flag = self.binarySearch(idx, 0, len(idx)-1, j)
      for i in range(9):

        if(i%3==0 and i!=0):
          k+=1
        if flag:
          p+=self.block[j,k,i%3]/255

        if(self.block[j,k,i%3] == 0):
          self.block[j,k,i%3] = 1
          self.listL.append(1)
          
        elif(self.block[j,k,i%3] == 255):
          self.block[j,k,i%3] = 254
          self.listL.append(1)
          
        elif(self.block[j,k,i%3] == 1 or self.block[j,k,i%3] == 254):        
          self.listL.append(0)
          
      if flag:
        self.listS.append(int(p*100))


  def generateRandInd(self, lenSubBlock, seed=None):
    idx = []
    if not seed:
      seed = int(np.random.uniform(0,9999))
    rng = np.random.default_rng(seed)
    for i in range(lenSubBlock):
      if(rng.random() > 0.99):
        idx.append(i)
    return List(idx), seed

  def generateRandEmb(self, ids, lenSubBlock, seed):
    rng = np.random.default_rng(seed)
    a = [i for i in range(ids+1,lenSubBlock)]
    idx = rng.choice(a,len(a), replace=False)
    return List(idx)


  def differencePixel(self):
    self.difpixels = self.block.copy().astype(np.int16)
    for i in range(len(self.difpixels)):
      k = 0
      x = self.block[i,k+1,1]
      for j in range(9):
        # print(i, k , j)
        if(j%3==0 and j!=0):
          k+=1
        if(j%3==1 and k==1):
          self.difpixels[i,k,j%3] = x
        else:
          self.difpixels[i,k,j%3] -= x

  def InData(self):
    strS = ''.join([str(bin(x))[2:].zfill(10) for x in self.listS])
    strL = ''.join(map(str,self.listL))
    sep = ''.join(format(i, '08b') for i in bytearray("|", encoding ='utf-8')).zfill(10) + ''.join(format(i, '08b') for i in bytearray(">", encoding ='utf-8')).zfill(10)

    data =  self.number +"|"
    bindata = ''.join(format(i, '08b') for i in bytearray(data, encoding ='utf-8'))
    mergedata = bindata + strS + sep + strL
    self.data = List(list(map(int, [bit for bit in mergedata])))

  def stringTobit(self):
    string= self.id + '|'
    bindata = ''.join(format(i, '08b') for i in bytearray(string, encoding ='utf-8'))
    return List(list(map(int, [bit for bit in bindata])))

  def embeddingID(self, bitId):
    counterData = 0
    for i in range(len(self.block)):
      k = 0
      for j in range(9):
        if(j%3==0 and j!=0):
          k+=1

        if(j%3==1 and k==1):
            self.block[i,k,j%3] += 0  
        elif(counterData < len(bitId)):
          if(self.difpixels[i,k,j%3] < -1):
            self.block[i,k,j%3] -= 1
          elif(self.difpixels[i,k,j%3] == -1):
            self.block[i,k,j%3] -= bitId[counterData]
            counterData += 1
          elif(self.difpixels[i,k,j%3] == 0):
            self.block[i,k,j%3] += bitId[counterData]
            counterData += 1
          elif(self.difpixels[i,k,j%3] > 0):
            self.block[i,k,j%3] += 1
            
        elif(counterData >= len(bitId)):
          return i

  def embedding(self, embidx):
    counterData = 0
    for index in range(len(embidx)):
      i = embidx[index]
      k = 0
      for j in range(9):
        if(j%3==0 and j!=0):
          k+=1

        if(j%3==1 and k==1):
            self.block[i,k,j%3] += 0 
        elif(counterData < len(self.data)):

          if(self.difpixels[i,k,j%3] < -1):
            self.block[i,k,j%3] -= 1
          elif(self.difpixels[i,k,j%3] == -1):
            self.block[i,k,j%3] -= self.data[counterData]
            counterData += 1
          elif(self.difpixels[i,k,j%3] == 0):
            self.block[i,k,j%3] += self.data[counterData]
            counterData += 1
          elif(self.difpixels[i,k,j%3] > 0):
            self.block[i,k,j%3] += 1
            
        elif(counterData >= len(self.data)):
          if(self.difpixels[i,k,j%3] <= -1):
            self.block[i,k,j%3] -= 1
          elif(self.difpixels[i,k,j%3] >= 0):
            self.block[i,k,j%3] += 1
          counterData+=1

  def stackImage(self, height, width):
    index1 = 0
    index2 = width//3
    z = list()
    for i in range(height):
      z.append(np.reshape(self.block[index1:index2],(width,3)))
      index1 = index2
      index2 += width//3
    return np.array(z)

  def insertData(self, seed):
    db = sqlite3.connect("val.db")
    cek = "SELECT id FROM validation WHERE id='{}'".format(id)
    rs = db.execute(cek).fetchall()
    if len(rs) == 0:
      cmd = "insert into validation(id,seed) values('{}','{}')".format(self.id,seed)
    else:
      cmd = "UPDATE validation set seed = '{}' where id = '{}'".format(seed,self.id)
    db.execute(cmd)
    db.commit()

  def run(self):
    self.img = cv2.resize(self.img, (self.w, self.h))

    self.SubBlockImage(self.h, self.w)
    Lidx, seed = self.generateRandInd(len(self.block))
    
    self.OverUnder_Flow_Handle(Lidx)

    self.insertData(seed)

    bitID = self.stringTobit()
    self.InData()

    self.differencePixel()
    idx = self.embeddingID(bitID)

    idx = self.generateRandEmb(idx, len(self.block), seed)
    self.embedding(idx)

    embed_image = self.stackImage(self.h, self.w)
    return cv2.imwrite("Embedded.png", embed_image)


"""# Run Extraction"""

class Extract():
  def __init__(self, img):
    self.img = img
    self.id = ""
    self.number = ""
    self.block = []
    self.respixels = []
    self.listL = []
    self.listS = []
    self.data = []
    self.h, self.w = self.img.shape[0], self.img.shape[1]
  
  
  def SubBlockImage(self):
    for i in range(self.h):
      k=0
      for j in range(3,self.w+3, 3):
        self.block.append(self.img[i,k:j])
        k+=3
    self.block = np.asarray(self.block)

  def binarySearch (self, arr, l, r, x):
    if r >= l:
        mid = l + (r - l) // 2
        if arr[mid] == x:
            return True
        elif arr[mid] > x:
            return self.binarySearch(arr, l, mid-1, x)
        else:
            return self.binarySearch(arr, mid + 1, r, x)
    else:
        return False

  def bitExtractionID(self):
    self.respixels = self.block.copy().astype(np.int16)
    counter = 0
    c2=0
    bits = []
    id=""
    for i in range(len(self.respixels)):
      k = 0
      x = self.block[i,k+1,1]
      for j in range(9):
        # print(i, k , j)
        if(j%3==0 and j!=0):
          k+=1
        if(j%3==1 and k==1):
          self.respixels[i,k,j%3] = x
        else:
          value =self.respixels[i,k,j%3] - x

          if(value == 0 or value == -1):
            bits.append(0)
            counter+=1
          elif(value == 1 or value == -2):
            bits.append(1)
            counter+=1

          if(value > 0):
            self.respixels[i,k,j%3] -= 1
          elif(value<-1):
            self.respixels[i,k,j%3] += 1
            
        if(counter == 8 and len(bits)!=0):
          counter = 0
          c2+=1
          if(c2==11):
            self.id = bits[:len(bits)-8]
            return i

  def text_from_bits(self, bits, encoding='utf-8', errors='surrogatepass'):
    n = int(bits, 2)
    return n.to_bytes((n.bit_length() + 7) // 8, 'big').decode(encoding, errors) or '\0'

  def getSeed(self):
    db = sqlite3.connect("val.db")
    cmd = "SELECT * FROM validation WHERE id='{}'".format(self.id)
    rs = db.execute(cmd).fetchall()
    if len(rs) > 0:
      return rs[0][1]
    return None

  def generateRandInd(self, seed=None):
    idx = []
    if not seed:
      seed = int(np.random.uniform(0,9999))
    rng = np.random.default_rng(seed)
    for i in range(len(self.respixels)):
      if(rng.random() > 0.99):
        idx.append(i)
    return List(idx), seed

  def generateRandEmb(self, ids, seed):
    rng = np.random.default_rng(seed)
    a = [i for i in range(ids+1,len(self.block))]
    idx = rng.choice(a,len(a), replace=False)
    return List(idx)
    
  def bitExtraction(self, embidx):
    bits = []
    for index in range(len(embidx)):
      i = embidx[index]
      k = 0
      x = self.block[i,k+1,1]
      for j in range(9):
        if(j%3==0 and j!=0):
          k+=1
        if(j%3==1 and k==1):
          self.respixels[i,k,j%3] = x
        else:
          value = self.respixels[i,k,j%3] - x
          if(value == 0 or value == -1):
            bits.append(0)
          elif(value == 1 or value == -2):
            bits.append(1)
          if(value > 0):
            self.respixels[i,k,j%3] -= 1
          elif(value<-1):
            self.respixels[i,k,j%3] += 1
    self.data = bits

  def convertExtracted(self):
    data = {
        "number":""
    }
    sepCount=0
    copy_extract = self.data.copy()
  
    for i in self.data:

      if (self.text_from_bits(i) == "|"):

        copy_extract.pop(0)
        sepCount+=1
        if (sepCount==1):
          break
      else: 

        data[[*data][sepCount]]+=self.text_from_bits(i)
        copy_extract.pop(0)

    self.number = data
    self.data = copy_extract  

  def DivideMap(self):
    mapS = []
    extr_data = self.data
    extr_data = [extr_data[i:i+10] for i in range(0, len(extr_data), 10)]
    extr_data_new = extr_data.copy()
    for i in range(len(extr_data)):
      try:
        if (self.text_from_bits(extr_data[i])=="|" and self.text_from_bits(extr_data[i+1])==">"):
          extr_data_new.pop(0)
          extr_data_new.pop(0)
          mapL = ''.join(extr_data_new)
          mapL = list(map(int, [bit for bit in mapL]))
          break
      except:
        pass
      mapS.append(int(extr_data[i],2))
      extr_data_new.pop(0)

    self.listL = mapL
    self.listS = mapS

  def fullyRestored(self, idx):
    counter = 0
    counterS = 0
    valid=True
    for i in range(len(self.respixels)):
      k = 0
      p = 0
      flag = self.binarySearch(idx, 0, len(idx)-1, i)
      for j in range(9):
        if(j%3==0 and j!=0):
          k+=1

        if not valid:
          self.respixels[i,k,j%3]=np.random.uniform(0,255)

        if(self.respixels[i,k,j%3] == 254 and self.listL[counter]==1):
          self.respixels[i,k,j%3]=255
          counter+=1

        elif(self.respixels[i,k,j%3] == 1 and self.listL[counter]==1):
          self.respixels[i,k,j%3]=0
          counter+=1

        elif((self.respixels[i,k,j%3] == 254 or self.respixels[i,k,j%3] == 1) and self.listL[counter]==0):
          counter+=1
          
        if flag:
          p+=self.respixels[i,k,j%3]/255
      if flag and valid:
        valid = int(p*100)==self.listS[counterS]
        counterS+=1

  def stackImage(self):
    index1 = 0
    index2 = self.w//3
    z = list()
    for i in range(self.h):
      z.append(np.reshape(self.respixels[index1:index2],(self.w,3)))
      index1 = index2
      index2 += self.w//3
    return np.array(z).astype('uint8')

  def run(self):
    self.SubBlockImage()
    i = self.bitExtractionID()

    self.id = self.text_from_bits(''.join(map(str,self.id)))
    seed = self.getSeed()

    resdup = self.respixels.copy()
    try:
      residx = self.generateRandEmb(i ,int(seed))
      self.bitExtraction(residx)
    except:
      self.respixels = resdup.copy()
      del(resdup)
      residx = self.generateRandEmb(i+1 ,int(seed))
      self.bitExtraction(residx)

    self.data = ''.join(map(str,self.data))
    self.data = [self.data[j:j+8] for j in range(0, len(self.data), 8)]
    self.convertExtracted()

    self.data = ''.join(self.data)
    self.DivideMap()
    try:
      index,_ = self.generateRandInd(seed=int(seed))
      self.fullyRestored(index)
      return self.stackImage(), self.number['number']
    except:
      return "",False

class Validation():
  def __init__(self, img, number):
    self.img = img
    self.number = number

  def findNumber(self):
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'
    custom_config = r'--oem 3 --psm 6'
    text = pytesseract.image_to_string(self.img, config=custom_config)
    if text.find(self.number) != 1:
      return True
    else:
      return False

  def run(self):
    if self.number:
      result = self.findNumber()
      if result:
        return "Ijazah Valid"
      else:
        return "Ijazah Tidak Valid, Nomor Tidak Ditemukan"
    else:
      return "Ijazah Tidak Valid"



