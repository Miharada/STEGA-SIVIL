import cv2
import pandas as pd
import numpy as np


import pytesseract
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

@njit
def bitExtraction(C1, subBlock, embidx):
  bits = []
  C1 = C1.astype(np.int16)
  for index in range(len(embidx)):
    i = embidx[index]
    k = 0
    x = subBlock[i,k+1,1]
    for j in range(9):
      if(j%3==0 and j!=0):
        k+=1
      if(j%3==1 and k==1):
        C1[i,k,j%3] = x
      else:
        value = C1[i,k,j%3] - x

        if(value == 0 or value == -1):
          # print("a")
          bits.append(0)
        elif(value == 1 or value == -2):
          # print("b")
          bits.append(1)

        if(value > 0):
          C1[i,k,j%3] -= 1
        elif(value<-1):
          C1[i,k,j%3] += 1
          
  return bits, C1

@njit
def bitExtractionID(C1, subBlock):
  counter = 0
  c2=0
  bits = []
  id=""
  C1 = C1.astype(np.int16)
  for i in range(len(C1)):
    k = 0
    x = subBlock[i,k+1,1]
    for j in range(9):
      # print(i, k , j)
      if(j%3==0 and j!=0):
        k+=1
      if(j%3==1 and k==1):
        C1[i,k,j%3] = x
      else:
        value = C1[i,k,j%3] - x

        if(value == 0 or value == -1):
          bits.append(0)
          counter+=1
        elif(value == 1 or value == -2):
          bits.append(1)
          counter+=1

        if(value > 0):
          C1[i,k,j%3] -= 1
        elif(value<-1):
          C1[i,k,j%3] += 1
          
      if(counter == 8 and len(bits)!=0):
        counter = 0
        c2+=1

        if(c2==11):
          return bits[:len(bits)-8], C1, i

def text_from_bits(bits, encoding='utf-8', errors='surrogatepass'):
    n = int(bits, 2)
    return n.to_bytes((n.bit_length() + 7) // 8, 'big').decode(encoding, errors) or '\0'


def convertExtracted(extract_data):
  data = {
      "number":""
  }
  sepCount=0
  copy_extract = extract_data.copy()
 
  for i in extract_data:
    if (text_from_bits(i) == "|"):
      copy_extract.pop(0)
      sepCount+=1
      if (sepCount==1):
        break
    else: 
      data[[*data][sepCount]]+=text_from_bits(i)
      copy_extract.pop(0)

  return data, copy_extract  

def convertID(bits):
  text=""
  for i in bits:
    text+=text_from_bits(i)
  return text

def DivideMap(extr_data):
  mapS=[]
  extr_data = [extr_data[i:i+10] for i in range(0, len(extr_data), 10)]
  extr_data_new = extr_data.copy()
  for i in range(len(extr_data)):
    try:
      if (text_from_bits(extr_data[i])=="|" and text_from_bits(extr_data[i+1])==">"):
        extr_data_new.pop(0)
        extr_data_new.pop(0)
        mapL = ''.join(extr_data_new)
        mapL = list(map(int, [bit for bit in mapL]))
        break
    except:
      pass
    mapS.append(int(extr_data[i],2))
    extr_data_new.pop(0)
  return mapS, mapL
    

@njit(fastmath=True)
def fullyRestored(res_im, mapL, idx, resMapS):
  counter = 0
  counterS = 0
  valid=True
  for i in range(len(res_im)):
    k = 0
    p = 0
    flag = binarySearch(idx, 0, len(idx)-1, i)
    for j in range(9):
      if(j%3==0 and j!=0):
        k+=1

      if not valid:
        res_im[i,k,j%3]=np.random.uniform(0,255)

      if(res_im[i,k,j%3] == 254 and mapL[counter]==1):
        res_im[i,k,j%3]=255
        counter+=1

      elif(res_im[i,k,j%3] == 1 and mapL[counter]==1):
        res_im[i,k,j%3]=0
        counter+=1

      elif((res_im[i,k,j%3] == 254 or res_im[i,k,j%3] == 1) and mapL[counter]==0):
        counter+=1
        
      if flag:
        p+=res_im[i,k,j%3]/255
    if flag and valid:
      valid = int(p*100)==resMapS[counterS]
      counterS+=1
        
  return res_im

def getSeed(id):
  db = sqlite3.connect("val.db")
  cmd = "SELECT * FROM validation WHERE id='{}'".format(id)
  rs = db.execute(cmd).fetchall()
  if len(rs) > 0:
    return rs[0][1]
  return None
  
def Extraction(emb_img):
  size = (emb_img.shape[0], emb_img.shape[1])
  block_embed_image = SubBlockImage(emb_img, size[0], size[1])

  D_emb = block_embed_image.copy()

  id, D_embs, i = bitExtractionID(D_emb, block_embed_image)
  id = text_from_bits(''.join(map(str,id)))
  id_seed = getSeed(id)
  try:
    residx = generateRandEmb(i, len(block_embed_image),int(id_seed))

    extracted_data, restored = bitExtraction(D_embs, block_embed_image, residx)
    text_from_bits(''.join(map(str,extracted_data[:8])))

  except:
    residx = generateRandEmb(i+1, len(block_embed_image),int(id_seed))
    extracted_data, restored = bitExtraction(D_embs, block_embed_image, residx)


  extracted_data_str = ''.join(map(str,extracted_data))
  extracted_data_str = [extracted_data_str[j:j+8] for j in range(0, len(extracted_data_str), 8)]
  convert_data, extracted_data_new = convertExtracted(extracted_data_str)
  extracted_data_new = ''.join(extracted_data_new)
  exMapS, exMapL = DivideMap(extracted_data_new)
  return convert_data, exMapS, exMapL, restored, size, int(id_seed)

def RestoreAndValidationStep1(id_seed, mapS, mapL, restored, size):
  index,_ = generateRandInd(len(restored), seed=id_seed)
  full_restored = fullyRestored(restored.copy(),List(mapL), index, List(mapS))
  full_restored = stackImage(full_restored, size[0], size[1])
  return full_restored.astype('uint8')


def stackImage(C2_Block, height, width):
  index1 = 0
  index2 = width//3
  z = list()
  for i in range(height):
    z.append(np.reshape(C2_Block[index1:index2],(width,3)))
    index1 = index2
    index2 += width//3
  return np.array(z)

def readTextFromImg(img):
  custom_config = r'--oem 3 --psm 6'
  text = pytesseract.image_to_string(img, config=custom_config)
  return text

def validationOCR(img, data):
  img = img[30:280]
  w = img.shape[1]
  h = img.shape[0]
  bpp = img.shape[2]


  img_bytes = img.tobytes()
  bpl = bpp * w
  
  pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'
  custom_config = r'--oem 3 --psm 6'
  text = pytesseract.image_to_string(img, config=custom_config)
  if text.find(data['number']) != -1:
    return True
  else:
    return False

def RealValidation(embedded_image):
  
  try:
    data, mapS, mapL, block_embed, size, seed = Extraction(embedded_image)
    val = RestoreAndValidationStep1(seed, mapS, mapL, block_embed, size)
  except Exception as e:
    print(e)
    val = False
  
  if isinstance(val, bool):
    print("Seed Not Found")
    return False
  else:
    result = validationOCR(embedded_image, data)
    if result:
      return True
    else:
      print("Number Not Found")

      return False

