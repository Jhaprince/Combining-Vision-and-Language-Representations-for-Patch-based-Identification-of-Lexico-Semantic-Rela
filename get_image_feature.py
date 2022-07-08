"""
   This file is made for extracting global feature of image
   previously in the original ECIR paper we were extracting the global features of text using glov 
"""




import pickle
import os
import glob
import  numpy as np

def unpickle_global_feature(pklpath):
  with open(pklpath, 'rb') as pickle_in:
    unpickled_list = pickle.load(pickle_in).cpu().data.numpy()
  return unpickled_list
  

def get_image_clip_feature(basepath,word):
  file=os.path.join(basepath,word)
  glob_vec=np.zeros(512)

  if os.path.exists(os.path.join(file,"image_feat.pkl")):
    print("yes")
    pklpath=os.path.join(file,"image_feat.pkl")
    glob_vec=unpickle_global_feature(pklpath)
    
    print(glob_vec.shape)
    
    

  return glob_vec.reshape(1,512)




 
