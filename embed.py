import cv2
import numpy as np

from numba import njit

from numba.typed import List

import sqlite3


def SubBlockImage(img, height, width):
  block = []
  for i in range(height):
    k=0
    for j in range(3,width+3, 3):
      block.append(img[i,k:j])
      k+=3
  block = np.asarray(block)
  return block

@njit(fastmath=True)
def binarySearch (arr, l, r, x):
  
    if r >= l:
        mid = l + (r - l) // 2
        if arr[mid] == x:
            return True
        elif arr[mid] > x:
            return binarySearch(arr, l, mid-1, x)
        else:
            return binarySearch(arr, mid + 1, r, x)
    else:
        return False

@njit(fastmath=True)
def calculateBlock(pixels):
  s=0
  for i in range(len(pixels)):
    s+=pixels[i]
  return int(s*100)

@njit(fastmath=True)
def OverUnder_Flow_Handle(block, idx):
  L=[]
  S=[]
  for j in range(len(block)):
    k = 0
    p=0
    flag = binarySearch(idx, 0, len(idx)-1, j)
    for i in range(9):

      if(i%3==0 and i!=0):
        k+=1
      if flag:
        p+=block[j,k,i%3]/255

      if(block[j,k,i%3] == 0):
        block[j,k,i%3] = 1
        L.append(1)
        
      elif(block[j,k,i%3] == 255):
        block[j,k,i%3] = 254
        L.append(1)
        
      elif(block[j,k,i%3] == 1 or block[j,k,i%3] == 254):        
        L.append(0)
        
    if flag:
      S.append(int(p*100))
  return L, S, block

def generateRandInd(lenSubBlock, seed=None):
  idx = []
  if not seed:
    seed = int(np.random.uniform(0,9999))
  rng = np.random.default_rng(seed)
  for i in range(lenSubBlock):
    if(rng.random() > 0.99):
      idx.append(i)
  return List(idx), seed

def generateRandEmb(ids, lenSubBlock, seed):
  rng = np.random.default_rng(seed)
  a = [i for i in range(ids+1,lenSubBlock)]
  idx = rng.choice(a,len(a), replace=False)
  return List(idx)

@njit
def differencePixel(C1, subBlocks):
  C1 = C1.astype(np.int16)
  for i in range(len(C1)):
    k = 0
    x = subBlocks[i,k+1,1]
    for j in range(9):
      # print(i, k , j)
      if(j%3==0 and j!=0):
        k+=1
      if(j%3==1 and k==1):
        C1[i,k,j%3] = x
      else:
        C1[i,k,j%3] -= x
  return C1

def InData(number, S, L):
  strS = ''.join([str(bin(x))[2:].zfill(10) for x in S])
  strL = ''.join(map(str,L))
  sep = ''.join(format(i, '08b') for i in bytearray("|", encoding ='utf-8')).zfill(10) + ''.join(format(i, '08b') for i in bytearray(">", encoding ='utf-8')).zfill(10)

  data =  number +"|"
  bindata = ''.join(format(i, '08b') for i in bytearray(data, encoding ='utf-8'))

  mergedata = bindata + strS + sep + strL
  mergedataInt = List(list(map(int, [bit for bit in mergedata])))
  return mergedataInt

def stringTobit(string):
  string+='|'
  bindata = ''.join(format(i, '08b') for i in bytearray(string, encoding ='utf-8'))
  return List(list(map(int, [bit for bit in bindata])))

@njit
def embedding(C1, D, embidx, data):
  counterData = 0
  for index in range(len(embidx)):
    i = embidx[index]
    k = 0
    for j in range(9):
      if(j%3==0 and j!=0):
        k+=1

      if(j%3==1 and k==1):
          C1[i,k,j%3] += 0 
      elif(counterData < len(data)):

        if(D[i,k,j%3] < -1):
          C1[i,k,j%3] -= 1
        elif(D[i,k,j%3] == -1):
          C1[i,k,j%3] -= data[counterData]
          counterData += 1
        elif(D[i,k,j%3] == 0):
          C1[i,k,j%3] += data[counterData]
          counterData += 1
        elif(D[i,k,j%3] > 0):
          C1[i,k,j%3] += 1
          
      elif(counterData >= len(data)):
        if(D[i,k,j%3] <= -1):
          C1[i,k,j%3] -= 1
        elif(D[i,k,j%3] >= 0):
          C1[i,k,j%3] += 1
        counterData+=1
  return C1

@njit
def embeddingID(C1, D, data):
  counterData = 0
  for i in range(len(C1)):
    k = 0
    for j in range(9):
      if(j%3==0 and j!=0):
        k+=1

      if(j%3==1 and k==1):
          C1[i,k,j%3] += 0  
      elif(counterData < len(data)):
        if(D[i,k,j%3] < -1):
          C1[i,k,j%3] -= 1
        elif(D[i,k,j%3] == -1):
          C1[i,k,j%3] -= data[counterData]
          counterData += 1
        elif(D[i,k,j%3] == 0):
          C1[i,k,j%3] += data[counterData]
          counterData += 1
        elif(D[i,k,j%3] > 0):
          C1[i,k,j%3] += 1
          
      elif(counterData >= len(data)):
        return C1, i

def stackImage(C2_Block, height, width):
  index1 = 0
  index2 = width//3
  z = list()
  for i in range(height):
    z.append(np.reshape(C2_Block[index1:index2],(width,3)))
    index1 = index2
    index2 += width//3
  return np.array(z)

def updateData(id, seed):
  db = sqlite3.connect("val.db")
  cek = "SELECT id FROM validation WHERE id='{}'".format(id)
  rs = db.execute(cek).fetchall()
  if len(rs) == 0:
    cmd = "insert into validation(id,seed) values('{}','{}')".format(id,seed)
  else:
    cmd = "UPDATE validation set seed = '{}' where id = '{}'".format(seed,id)
  db.execute(cmd)

  db.commit()

def insertData(seed):
  db = sqlite3.connect("val.db")
  cek = "SELECT id FROM validation".format(id)
  rs = db.execute(cek).fetchall()
  if len(rs) == 0:
    ids = "000"
    cmd = "insert into validation(id,seed) values('{}','{}')".format(ids,seed)
  else:
    QueryLastRow = "SELECT id FROM validation WHERE id=(SELECT max(id) FROM validation)"
    LastId = db.execute(QueryLastRow).fetchall()[0][0]
    ids = str(int(LastId)+1).zfill(3)
    cmd = "insert into validation(id,seed) values('{}','{}')".format(ids,seed)

  db.execute(cmd)
  db.commit()
  return ids

def Embedding(img, number):
  img = cv2.resize(img, (1122,792))
  height, width = 792, 1122

  subBlock = SubBlockImage(img, height, width)
  Lidx, seed = generateRandInd(len(subBlock))
  mapL, mapS, C = OverUnder_Flow_Handle(subBlock, Lidx)
  id = insertData(seed)

  embid = stringTobit(id)
  data = InData(number, mapS, mapL)

  subCopy = C.copy()
  D = differencePixel(subCopy, C)
  
  C1s, idx = embeddingID(C.copy(), D, embid)
  idx = generateRandEmb(idx, len(subBlock), seed)

  C2 = embedding(C1s.copy(), D, idx, data)
  embed_image = stackImage(C2, height, width)
  return embed_image