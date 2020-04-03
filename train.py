# Copyright 2018
# 
# Yaojie Liu, Amin Jourabloo, Xiaoming Liu, Michigan State University
# 
# All Rights Reserved.
# 
# This research is based upon work supported by the Office of the Director of 
# National Intelligence (ODNI), Intelligence Advanced Research Projects Activity
# (IARPA), via IARPA R&D Contract No. 2017-17020200004. The views and 
# conclusions contained herein are those of the authors and should not be 
# interpreted as necessarily representing the official policies or endorsements,
# either expressed or implied, of the ODNI, IARPA, or the U.S. Government. The 
# U.S. Government is authorized to reproduce and distribute reprints for 
# Governmental purposes not withstanding any copyright annotation thereon. 
# ==============================================================================
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
from datetime import datetime
import time

import tensorflow as tf
from model.dataset import Dataset
from model.config import Config
from model.model import Model

if __name__ == "__main__":
    
    # Configurations
    config = Config()
    # Train
    # Epoch-1 learning rate = 0.0003
    config.DATA_DIR = ["/home/umit/xDataset/deepFake-dat/Train_Fake_Much_1",
                       "/home/umit/xDataset/deepFake-dat/Train_Live_Much_1",
                       "/home/umit/xDataset/deepFake-dat/Train_Fake_Much_2",
                       "/home/umit/xDataset/deepFake-dat/Train_Live_Much_2",
                       "/home/umit/xDataset/deepFake-dat/Train_Fake_Much_3",
                       "/home/umit/xDataset/deepFake-dat/Train_Live_Much_3",
                       "/home/umit/xDataset/deepFake-dat/Train_Fake_Much_4",
                       "/home/umit/xDataset/deepFake-dat/Train_Live_Much_4",
                       "/home/umit/xDataset/deepFake-dat/Train_Fake_Much_5",
                       "/home/umit/xDataset/deepFake-dat/Train_Live_Much_5",
                       "/home/umit/xDataset/deepFake-dat/Train_Fake_Much_6"
                       "/home/umit/xDataset/deepFake-dat/Train_Live_Much_6",
                       "/home/umit/xDataset/deepFake-dat/Train_Fake_Much_7"]
    

    config.LOG_DIR = './log/model'
    config.MODE = 'training'
    config.STEPS_PER_EPOCH = 2000
    config.MAX_EPOCH = 1000
    config.LEARNING_RATE = 0.00001 #0.00005 #0.0001 #0.0005 #0.001
    config.BATCH_SIZE = 20
    # Validation
    config.DATA_DIR_VAL = ["/home/umit/xDataset/deepFake-dat/Train_Fake_Much_1",
                           "/home/umit/xDataset/deepFake-dat/Train_Live_Few_1"]
    config.STEPS_PER_EPOCH_VAL = 500
   
    config.display()

    # Get images and labels.
    dataset_train = Dataset(config,'train')
    #dataset_validation = Dataset(config,'validation')
    
    # Build a Graph
    model = Model(config)

    # # Train the model
    model.compile()
    model.train(dataset_train, val=None)
    
# Epoch 47-1000/1000: Map:0.116, Cls:0.0997, Route:1.78(0.108, 151.491), Uniq:nan, Counts:[1,1,0,0,6,2,0,5]     

#     Time taken for epoch 47 is 1144.3 sec
# Epoch 48-1000/1000: Map:0.116, Cls:0.0988, Route:1.78(0.091, 142.153), Uniq:nan, Counts:[2,1,0,1,4,1,2,8]     

#     Time taken for epoch 48 is 1123.81 sec
# Epoch 49-1000/1000: Map:0.115, Cls:0.0973, Route:1.77(0.061, 120.022), Uniq:nan, Counts:[3,0,0,1,4,1,2,6]     

#     Time taken for epoch 49 is 1121.01 sec
# Epoch 50-1000/1000: Map:0.114, Cls:0.0963, Route:1.76(0.097, 136.802), Uniq:nan, Counts:[3,0,1,0,3,2,2,6]     

#     Time taken for epoch 50 is 1122.96 sec
# Epoch 51-710/1000: Map:0.113, Cls:0.0956, Route:1.76(0.083, 135.795), Uniq:nan, Counts:[2,1,1,0,4,1,1,6] 

# Epoch 101-2000/2000: Map:0.102, Cls:0.084, Route:1.55(0.076, 120.952), Uniq:nan, Counts:[1,1,1,0,3,2,1,7]       

#     Time taken for epoch 101 is 2276.19 sec
# Epoch 102-2000/2000: Map:0.102, Cls:0.0846, Route:1.56(0.090, 118.274), Uniq:nan, Counts:[3,1,1,0,3,3,2,5]      

#     Time taken for epoch 102 is 2247.31 sec
# Epoch 103-2000/2000: Map:0.102, Cls:0.0837, Route:1.55(0.063, 113.360), Uniq:nan, Counts:[1,2,1,0,3,2,2,7]      

#     Time taken for epoch 103 is 2241.02 sec
# Epoch 104-2000/2000: Map:0.102, Cls:0.0829, Route:1.55(0.090, 122.335), Uniq:nan, Counts:[1,2,1,0,4,2,2,6]      

#     Time taken for epoch 104 is 2242.65 sec
# Epoch 105-2000/2000: Map:0.102, Cls:0.0824, Route:1.55(0.096, 115.555), Uniq:nan, Counts:[1,2,0,1,2,1,1,6]      

#     Time taken for epoch 105 is 2240.77 sec
# Epoch 106-2000/2000: Map:0.101, Cls:0.0815, Route:1.55(0.079, 114.446), Uniq:nan, Counts:[2,2,0,0,5,2,1,5]      

#     Time taken for epoch 106 is 2242.11 sec
# Epoch 107-2000/2000: Map:0.101, Cls:0.0809, Route:1.55(0.095, 135.246), Uniq:nan, Counts:[2,2,2,0,7,2,1,4]      

#     Time taken for epoch 107 is 2245.71 sec
# Epoch 108-2000/2000: Map:0.101, Cls:0.0805, Route:1.55(0.054, 119.668), Uniq:nan, Counts:[3,1,0,0,3,0,2,8]
# Time taken for epoch 108 is 2241.54 sec
# Epoch 109-2000/2000: Map:0.1, Cls:0.0799, Route:1.54(0.057, 106.944), Uniq:nan, Counts:[1,2,1,0,4,0,3,8]        

#     Time taken for epoch 109 is 2250.45 sec
# Epoch 110-2000/2000: Map:0.1, Cls:0.0791, Route:1.54(0.055, 104.063), Uniq:nan, Counts:[1,1,0,0,2,0,2,11]     

#     Time taken for epoch 110 is 2249.05 sec
# Epoch 111-2000/2000: Map:0.1, Cls:0.0786, Route:1.54(0.057, 105.384), Uniq:nan, Counts:[2,2,0,0,3,1,1,6]  

# Epoch 112-2000/2000: Map:0.0932, Cls:0.0893, Route:1.42(0.047, 84.716), Uniq:0.0509, Counts:[0,1,0,0,1,1,3,5]      

#     Time taken for epoch 112 is 2281.11 sec
# Epoch 113-2000/2000: Map:0.092, Cls:0.0863, Route:1.42(0.057, 103.698), Uniq:0.0514, Counts:[1,1,0,0,2,1,1,5]      

#     Time taken for epoch 113 is 2263.94 sec
# Epoch 114-2000/2000: Map:0.0917, Cls:0.0853, Route:1.42(0.039, 97.293), Uniq:0.0513, Counts:[1,1,0,0,3,0,1,4]       

#     Time taken for epoch 114 is 2264.64 sec
# Epoch 115-2000/2000: Map:0.0917, Cls:0.0843, Route:1.42(0.062, 119.617), Uniq:0.0513, Counts:[1,3,0,0,4,1,0,5]      

#     Time taken for epoch 115 is 2264.13 sec
# Epoch 116-2000/2000: Map:0.0913, Cls:0.083, Route:1.42(0.061, 95.227), Uniq:nan, Counts:[1,1,0,0,3,0,1,7]           

#     Time taken for epoch 116 is 2267.51 sec
# Epoch 117-2000/2000: Map:0.0911, Cls:0.0824, Route:1.42(0.039, 58.522), Uniq:nan, Counts:[1,0,0,0,2,1,0,7]       

#     Time taken for epoch 117 is 2259.17 sec
# Epoch 118-2000/2000: Map:0.0909, Cls:0.0817, Route:1.42(0.048, 104.512), Uniq:nan, Counts:[0,2,1,0,1,1,1,10]     

#     Time taken for epoch 118 is 2259.81 sec
# Epoch 119-2000/2000: Map:0.0907, Cls:0.0812, Route:1.41(0.072, 118.866), Uniq:nan, Counts:[1,1,2,0,3,2,0,5]      

#     Time taken for epoch 119 is 2261.45 sec
# Epoch 120-2000/2000: Map:0.0906, Cls:0.0806, Route:1.41(0.045, 61.072), Uniq:nan, Counts:[1,1,0,0,1,0,2,4]       

#     Time taken for epoch 120 is 2262.29 sec
# Epoch 121-1931/2000: Map:0.0904, Cls:0.0801, Route:1.41(0.043, 112.243), Uniq:nan, Counts:[0,2,0,0,1,1,1,7]   

# Time taken for epoch 120 is 2262.29 sec
# Epoch 121-2000/2000: Map:0.0904, Cls:0.08, Route:1.41(0.085, 103.970), Uniq:nan, Counts:[2,1,0,0,3,1,0,6]        

#     Time taken for epoch 121 is 2261.57 sec
# Epoch 122-2000/2000: Map:0.0901, Cls:0.0794, Route:1.41(0.086, 105.890), Uniq:nan, Counts:[1,2,0,1,2,1,2,3]      

#     Time taken for epoch 122 is 2264.73 sec
# Epoch 123-2000/2000: Map:0.0899, Cls:0.0789, Route:1.41(0.042, 93.505), Uniq:nan, Counts:[1,1,1,0,3,0,0,9]      

#     Time taken for epoch 123 is 2265.06 sec
# Epoch 124-2000/2000: Map:0.0897, Cls:0.0784, Route:1.41(0.061, 94.365), Uniq:nan, Counts:[0,1,0,1,2,1,1,6]       

#     Time taken for epoch 124 is 2265.46 sec
# Epoch 125-1316/2000: Map:0.0895, Cls:0.078, Route:1.41(0.040, 64.974), Uniq:nan, Counts:[0,1,0,0,3,0,2,4] 

# Epoch 125-2000/2000: Map:0.0895, Cls:0.0779, Route:1.41(0.054, 96.589), Uniq:nan, Counts:[2,2,0,0,2,1,1,6]      

#     Time taken for epoch 125 is 2270.42 sec
# Epoch 126-2000/2000: Map:0.0893, Cls:0.0776, Route:1.41(0.073, 107.502), Uniq:nan, Counts:[1,1,0,0,1,1,2,4]      

#     Time taken for epoch 126 is 2265.72 sec
# Epoch 127-2000/2000: Map:0.0891, Cls:0.0772, Route:1.41(0.072, 99.577), Uniq:nan, Counts:[1,2,1,0,2,0,0,7]       

#     Time taken for epoch 127 is 2272.57 sec
# Epoch 128-2000/2000: Map:0.0889, Cls:0.0767, Route:1.41(0.043, 119.279), Uniq:nan, Counts:[1,1,0,0,3,1,0,5]      

#     Time taken for epoch 128 is 2271.52 sec
# Epoch 129-2000/2000: Map:0.0886, Cls:0.0763, Route:1.41(0.040, 68.747), Uniq:nan, Counts:[0,1,1,0,4,0,3,4]       

#     Time taken for epoch 129 is 2265.1 sec
# Epoch 130-2000/2000: Map:0.0884, Cls:0.0758, Route:1.41(0.078, 106.493), Uniq:nan, Counts:[2,1,0,0,4,1,1,4]      

#     Time taken for epoch 130 is 2264.72 sec
# Epoch 131-2000/2000: Map:0.0882, Cls:0.0754, Route:1.41(0.062, 110.413), Uniq:nan, Counts:[1,1,1,0,3,2,1,7]      

#     Time taken for epoch 131 is 2267.13 sec
# Epoch 132-2000/2000: Map:0.0881, Cls:0.0751, Route:1.41(0.059, 70.406), Uniq:nan, Counts:[1,0,0,0,4,2,0,4]      

#     Time taken for epoch 132 is 2264.11 sec
# Epoch 133-2000/2000: Map:0.0879, Cls:0.0747, Route:1.41(0.041, 53.848), Uniq:nan, Counts:[0,1,0,0,1,0,0,5]       

#     Time taken for epoch 133 is 2266.92 sec
# Epoch 134-2000/2000: Map:0.0878, Cls:0.0743, Route:1.41(0.072, 120.976), Uniq:nan, Counts:[1,1,2,0,3,2,1,5]      

#     Time taken for epoch 134 is 2263.99 sec
# Epoch 135-2000/2000: Map:0.0876, Cls:0.074, Route:1.41(0.054, 104.289), Uniq:nan, Counts:[1,1,0,0,3,1,1,7]      

#     Time taken for epoch 135 is 2266.46 sec
# Epoch 136-2000/2000: Map:0.0874, Cls:0.0736, Route:1.41(0.058, 112.057), Uniq:nan, Counts:[1,1,0,0,4,3,0,6]      

#     Time taken for epoch 136 is 2268.9 sec
# Epoch 137-2000/2000: Map:0.0873, Cls:0.0732, Route:1.41(0.041, 93.818), Uniq:nan, Counts:[1,2,0,0,2,0,0,3]       

#     Time taken for epoch 137 is 2265.06 sec
# Epoch 138-2000/2000: Map:0.0871, Cls:0.0729, Route:1.41(0.031, 74.711), Uniq:nan, Counts:[1,1,0,1,1,0,1,6]       

#     Time taken for epoch 138 is 2268.69 sec
# Epoch 139-2000/2000: Map:0.087, Cls:0.0726, Route:1.41(0.071, 110.672), Uniq:nan, Counts:[2,3,0,0,3,1,0,5]      

#     Time taken for epoch 139 is 2266.66 sec
# Epoch 140-2000/2000: Map:0.0868, Cls:0.0723, Route:1.41(0.091, 105.360), Uniq:nan, Counts:[2,1,0,0,2,0,1,3]
# Epoch 141-2000/2000: Map:0.0818, Cls:0.0627, Route:1.4(0.059, 95.610), Uniq:0.0545, Counts:[1,1,0,0,2,1,3,7]       

#     Time taken for epoch 141 is 2351.94 sec
# Epoch 142-2000/2000: Map:0.0819, Cls:0.0626, Route:1.4(0.072, 92.004), Uniq:nan, Counts:[2,2,0,0,3,2,1,4]         

#     Time taken for epoch 142 is 2558.8 sec
# Epoch 143-2000/2000: Map:0.0817, Cls:0.0621, Route:1.39(0.051, 88.813), Uniq:nan, Counts:[2,1,1,0,2,1,2,5]       

#     Time taken for epoch 143 is 2474.52 sec
# Epoch 144-2000/2000: Map:0.0817, Cls:0.0617, Route:1.39(0.083, 100.473), Uniq:nan, Counts:[3,2,0,0,2,2,1,5]      

#     Time taken for epoch 144 is 2771.57 sec
# Epoch 145-2000/2000: Map:0.0815, Cls:0.0613, Route:1.39(0.031, 94.362), Uniq:nan, Counts:[0,2,0,0,3,0,0,5]       

#     Time taken for epoch 145 is 3203.42 sec
# Epoch 146-2000/2000: Map:0.0813, Cls:0.0611, Route:1.39(0.036, 98.731), Uniq:nan, Counts:[1,3,0,0,1,0,1,7]       

#     Time taken for epoch 146 is 3140.6 sec
# Epoch 147-2000/2000: Map:0.0813, Cls:0.0608, Route:1.39(0.061, 96.768), Uniq:nan, Counts:[1,3,0,0,2,1,2,5]       

#     Time taken for epoch 147 is 3050.15 sec
# Epoch 148-2000/2000: Map:0.0813, Cls:0.0607, Route:1.39(0.090, 116.673), Uniq:nan, Counts:[2,3,2,0,4,0,2,3]      

#     Time taken for epoch 148 is 2466.25 sec
# Epoch 149-2000/2000: Map:0.0812, Cls:0.0607, Route:1.39(0.106, 123.282), Uniq:nan, Counts:[0,2,0,0,2,1,1,4]      

#     Time taken for epoch 149 is 2954.1 sec
# Epoch 150-2000/2000: Map:0.0812, Cls:0.0604, Route:1.39(0.097, 141.759), Uniq:nan, Counts:[1,1,1,1,3,1,1,4]      

#     Time taken for epoch 150 is 2762.13 sec
# Epoch 151-2000/2000: Map:0.081, Cls:0.0602, Route:1.39(0.067, 119.753), Uniq:nan, Counts:[1,1,0,0,3,2,1,4]       

#     Time taken for epoch 151 is 3087.5 sec
# Epoch 152-2000/2000: Map:0.0809, Cls:0.0602, Route:1.39(0.091, 91.312), Uniq:nan, Counts:[2,1,0,1,2,2,2,5]      

#     Time taken for epoch 152 is 3350.51 sec
# Epoch 153-2000/2000: Map:0.0808, Cls:0.06, Route:1.39(0.066, 108.025), Uniq:nan, Counts:[1,2,0,0,3,1,2,6]        

#     Time taken for epoch 153 is 3166.3 sec
# Epoch 154-2000/2000: Map:0.0806, Cls:0.0598, Route:1.39(0.074, 90.791), Uniq:nan, Counts:[2,2,0,0,4,0,2,4] 
# Epoch 155-2000/2000: Map:0.0775, Cls:0.0552, Route:1.38(0.084, 105.581), Uniq:0.0588, Counts:[1,3,0,0,4,1,1,6]      

#     Time taken for epoch 155 is 2326.64 sec
# Epoch 156-2000/2000: Map:0.0782, Cls:0.0557, Route:1.39(0.079, 109.855), Uniq:0.0573, Counts:[2,3,1,0,4,0,1,4]      

#     Time taken for epoch 156 is 2321.37 sec
# Epoch 157-2000/2000: Map:0.0786, Cls:0.0564, Route:1.39(0.065, 85.438), Uniq:nan, Counts:[2,2,0,0,4,0,1,6]          

#     Time taken for epoch 157 is 2305.62 sec
# Epoch 158-2000/2000: Map:0.0789, Cls:0.0563, Route:1.39(0.058, 99.374), Uniq:nan, Counts:[2,2,0,1,3,1,0,5]       

#     Time taken for epoch 158 is 2319.04 sec
# Epoch 159-2000/2000: Map:0.0789, Cls:0.0562, Route:1.39(0.088, 109.190), Uniq:nan, Counts:[1,1,0,1,4,1,1,7]     

#     Time taken for epoch 159 is 2329.34 sec
# Epoch 160-1375/2000: Map:0.0788, Cls:0.056, Route:1.39(0.048, 96.595), Uniq:nan, Counts:[2,1,0,0,2,1,1,10] 
# Epoch 160-2000/2000: Map:0.0788, Cls:0.0561, Route:1.39(0.043, 67.837), Uniq:nan, Counts:[1,0,0,0,5,1,1,5]       

#     Time taken for epoch 160 is 2310.02 sec
# Epoch 161-2000/2000: Map:0.0787, Cls:0.056, Route:1.38(0.060, 92.278), Uniq:nan, Counts:[1,1,1,0,3,2,1,5]        

#     Time taken for epoch 161 is 2305.14 sec
# Epoch 162-2000/2000: Map:0.0785, Cls:0.0558, Route:1.38(0.032, 90.787), Uniq:nan, Counts:[1,2,0,0,3,0,0,8]       

#     Time taken for epoch 162 is 2298.73 sec
# Epoch 163-2000/2000: Map:0.0783, Cls:0.0556, Route:1.38(0.072, 115.163), Uniq:nan, Counts:[1,1,0,0,2,1,1,8]      

#     Time taken for epoch 163 is 2300.66 sec
# Epoch 164-2000/2000: Map:0.0783, Cls:0.0556, Route:1.38(0.045, 101.264), Uniq:nan, Counts:[1,1,0,0,1,2,2,4]     

#     Time taken for epoch 164 is 2294.3 sec
# Epoch 165-2000/2000: Map:0.0782, Cls:0.0554, Route:1.38(0.060, 116.590), Uniq:nan, Counts:[1,1,1,0,3,2,0,6]      

#     Time taken for epoch 165 is 2293.15 sec
# Epoch 166-2000/2000: Map:0.0781, Cls:0.0553, Route:1.38(0.066, 86.693), Uniq:nan, Counts:[1,0,0,0,2,0,1,4]       

#     Time taken for epoch 166 is 2310.25 sec
# Epoch 167-2000/2000: Map:0.0781, Cls:0.0552, Route:1.38(0.074, 95.514), Uniq:nan, Counts:[2,2,0,0,3,2,1,6]       

#     Time taken for epoch 167 is 2298.71 sec
# Epoch 168-2000/2000: Map:0.078, Cls:0.0551, Route:1.38(0.062, 98.340), Uniq:nan, Counts:[2,3,0,0,3,1,1,6]       

#     Time taken for epoch 168 is 2348.79 sec
# Epoch 169-2000/2000: Map:0.0779, Cls:0.0551, Route:1.38(0.052, 109.461), Uniq:nan, Counts:[2,2,0,0,4,2,1,8]      

#     Time taken for epoch 169 is 2286.56 sec
# Epoch 170-2000/2000: Map:0.0779, Cls:0.055, Route:1.38(0.056, 103.041), Uniq:nan, Counts:[1,1,0,0,2,2,1,4]      

#     Time taken for epoch 170 is 2294.77 sec
# Epoch 171-2000/2000: Map:0.0778, Cls:0.0549, Route:1.38(0.046, 102.965), Uniq:nan, Counts:[2,1,0,0,4,0,1,6]      

#     Time taken for epoch 171 is 2302.95 sec
# Epoch 172-2000/2000: Map:0.0777, Cls:0.0547, Route:1.38(0.081, 102.508), Uniq:nan, Counts:[3,2,0,0,3,1,0,4]      

#     Time taken for epoch 172 is 2304.62 sec
# Epoch 173-2000/2000: Map:0.0777, Cls:0.0547, Route:1.38(0.074, 98.295), Uniq:nan, Counts:[4,2,1,0,3,1,1,6]       

#     Time taken for epoch 173 is 2295.15 sec
# Epoch 174-2000/2000: Map:0.0777, Cls:0.0546, Route:1.38(0.059, 108.156), Uniq:nan, Counts:[1,3,0,0,4,1,2,5]      

#     Time taken for epoch 174 is 2331.52 sec
# Epoch 175-675/2000: Map:0.0777, Cls:0.0545, Route:1.38(0.046, 83.538), Uniq:nan, Counts:[1,2,0,0,3,1,2,6]  

# Epoch 176-2000/2000: Map:0.0739, Cls:0.0517, Route:1.42(0.079, 116.107), Uniq:nan, Counts:[2,2,0,0,4,1,1,5]         

#     Time taken for epoch 176 is 2304.39 sec
# Epoch 177-2000/2000: Map:0.0738, Cls:0.0507, Route:1.42(0.070, 156.568), Uniq:nan, Counts:[1,1,0,0,2,1,1,5]      

#     Time taken for epoch 177 is 2271.4 sec
# Epoch 178-2000/2000: Map:0.0739, Cls:0.051, Route:1.42(0.068, 104.556), Uniq:nan, Counts:[1,2,1,1,0,1,1,7]       

#     Time taken for epoch 178 is 2267.88 sec
# Epoch 179-2000/2000: Map:0.0736, Cls:0.0509, Route:1.41(0.048, 99.661), Uniq:nan, Counts:[1,1,1,0,2,0,1,6]       

#     Time taken for epoch 179 is 2266.34 sec
# Epoch 180-2000/2000: Map:0.0736, Cls:0.0509, Route:1.41(0.089, 99.611), Uniq:nan, Counts:[2,2,1,0,3,1,0,6]       

#     Time taken for epoch 180 is 2262.68 sec
# Epoch 181-2000/2000: Map:0.0735, Cls:0.0506, Route:1.41(0.036, 80.973), Uniq:nan, Counts:[1,3,0,0,3,0,2,6]       

#     Time taken for epoch 181 is 2261.92 sec
# Epoch 182-2000/2000: Map:0.0734, Cls:0.0505, Route:1.41(0.039, 90.345), Uniq:nan, Counts:[1,2,0,0,3,0,2,4]       

#     Time taken for epoch 182 is 2262.79 sec
# Epoch 183-2000/2000: Map:0.0732, Cls:0.0503, Route:1.41(0.058, 92.292), Uniq:nan, Counts:[2,2,0,0,2,1,1,6]       

#     Time taken for epoch 183 is 2267.6 sec
# Epoch 184-2000/2000: Map:0.073, Cls:0.05, Route:1.41(0.040, 96.485), Uniq:nan, Counts:[2,1,0,1,1,1,1,6]          

#     Time taken for epoch 184 is 2262 sec
# Epoch 185-2000/2000: Map:0.0729, Cls:0.0499, Route:1.4(0.053, 99.057), Uniq:nan, Counts:[3,2,0,0,3,0,0,3]        

#     Time taken for epoch 185 is 2268 sec
# Epoch 186-2000/2000: Map:0.0728, Cls:0.0499, Route:1.4(0.047, 90.862), Uniq:nan, Counts:[1,1,1,0,3,0,1,11]      

#     Time taken for epoch 186 is 2266.93 sec
# Epoch 187-2000/2000: Map:0.0727, Cls:0.0496, Route:1.4(0.051, 87.400), Uniq:nan, Counts:[1,1,0,0,3,2,1,5]       

#     Time taken for epoch 187 is 2266.46 sec
# Epoch 188-2000/2000: Map:0.0728, Cls:0.0496, Route:1.4(0.080, 106.352), Uniq:nan, Counts:[1,1,1,0,2,1,2,9]      

#     Time taken for epoch 188 is 2266.49 sec
# Epoch 189-2000/2000: Map:0.0727, Cls:0.0495, Route:1.4(0.088, 132.643), Uniq:nan, Counts:[1,1,0,0,3,2,1,6]      

#     Time taken for epoch 189 is 2267.3 sec
# Epoch 190-2000/2000: Map:0.0727, Cls:0.0494, Route:1.4(0.081, 155.813), Uniq:nan, Counts:[1,2,1,0,2,1,1,3]      

#     Time taken for epoch 190 is 2266.74 sec
# Epoch 191-2000/2000: Map:0.0727, Cls:0.0493, Route:1.4(0.053, 101.373), Uniq:nan, Counts:[3,2,0,0,3,1,0,8]      

#     Time taken for epoch 191 is 2263.22 sec
# Epoch 192-2000/2000: Map:0.0726, Cls:0.0491, Route:1.4(0.047, 89.585), Uniq:nan, Counts:[2,3,0,0,3,1,0,6]       

#     Time taken for epoch 192 is 2267.59 sec
# Epoch 193-2000/2000: Map:0.0726, Cls:0.0491, Route:1.4(0.055, 73.708), Uniq:nan, Counts:[1,1,0,0,2,2,0,5] 

# Epoch 225-2000/2000: Map:0.0637, Cls:0.0309, Route:1.36(0.038, 105.912), Uniq:nan, Counts:[2,3,0,0,2,0,1,6]       

#     Time taken for epoch 225 is 2268.28 sec
# Epoch 226-2000/2000: Map:0.064, Cls:0.0311, Route:1.36(0.060, 96.368), Uniq:nan, Counts:[4,1,0,0,2,2,0,8]        

#     Time taken for epoch 226 is 2232.1 sec
# Epoch 227-2000/2000: Map:0.0641, Cls:0.031, Route:1.37(0.045, 61.867), Uniq:nan, Counts:[0,1,0,0,2,1,2,6]       

#     Time taken for epoch 227 is 2230.57 sec
# Epoch 228-2000/2000: Map:0.064, Cls:0.0307, Route:1.37(0.051, 95.136), Uniq:nan, Counts:[1,2,0,0,3,1,0,9]       

#     Time taken for epoch 228 is 2243.11 sec
# Epoch 229-2000/2000: Map:0.0637, Cls:0.0303, Route:1.36(0.059, 102.527), Uniq:nan, Counts:[2,3,0,0,4,0,1,5]      

#     Time taken for epoch 229 is 2234.23 sec
# Epoch 230-2000/2000: Map:0.0636, Cls:0.0304, Route:1.36(0.048, 90.996), Uniq:nan, Counts:[2,2,0,0,0,1,1,7]       

#     Time taken for epoch 230 is 2232.15 sec
# Epoch 231-2000/2000: Map:0.0636, Cls:0.0303, Route:1.36(0.108, 98.419), Uniq:nan, Counts:[1,3,0,0,3,0,1,4]  

# Epoch 244-2000/2000: Map:0.0619, Cls:0.0287, Route:1.35(0.049, 98.751), Uniq:0.0579, Counts:[1,2,0,0,1,1,0,5]      

#     Time taken for epoch 244 is 2421.24 sec
# Epoch 245-2000/2000: Map:0.0621, Cls:0.029, Route:1.35(0.061, 121.889), Uniq:0.0571, Counts:[5,4,0,0,2,0,0,7]      

#     Time taken for epoch 245 is 2476.21 sec
# Epoch 246-2000/2000: Map:0.062, Cls:0.0289, Route:1.35(0.047, 111.607), Uniq:nan, Counts:[1,1,0,0,3,0,1,3]      

#     Time taken for epoch 246 is 2480.02 sec
# Epoch 247-2000/2000: Map:0.0619, Cls:0.0287, Route:1.35(0.077, 93.085), Uniq:nan, Counts:[1,2,1,0,1,2,1,6]      

#     Time taken for epoch 247 is 2504.82 sec
# Epoch 248-2000/2000: Map:0.062, Cls:0.0287, Route:1.35(0.063, 110.946), Uniq:nan, Counts:[2,1,0,0,2,1,1,6]      

#     Time taken for epoch 248 is 2527.18 sec
# Epoch 249-2000/2000: Map:0.062, Cls:0.0288, Route:1.35(0.081, 101.243), Uniq:nan, Counts:[4,2,0,0,1,2,0,6]       

#     Time taken for epoch 249 is 2378.35 sec
# Epoch 250-2000/2000: Map:0.062, Cls:0.0289, Route:1.35(0.072, 103.189), Uniq:nan, Counts:[1,3,0,0,1,1,1,5]      

#     Time taken for epoch 250 is 2438.41 sec
# Epoch 251-2000/2000: Map:0.0619, Cls:0.0289, Route:1.35(0.073, 97.494), Uniq:nan, Counts:[1,2,0,0,2,1,1,5]       

#     Time taken for epoch 251 is 2344.36 sec
# Epoch 252-2000/2000: Map:0.0619, Cls:0.0289, Route:1.35(0.053, 89.079), Uniq:nan, Counts:[1,3,0,0,3,1,2,4]       

#     Time taken for epoch 252 is 2427.33 sec
# Epoch 253-2000/2000: Map:0.0619, Cls:0.029, Route:1.35(0.060, 83.629), Uniq:nan, Counts:[1,2,0,0,4,1,0,3]      

#     Time taken for epoch 253 is 2480.83 sec
# Epoch 254-2000/2000: Map:0.0619, Cls:0.029, Route:1.35(0.060, 81.152), Uniq:nan, Counts:[1,2,0,0,3,0,2,6]       

#     Time taken for epoch 254 is 2455.52 sec
# Epoch 255-2000/2000: Map:0.0619, Cls:0.029, Route:1.35(0.016, 75.027), Uniq:nan, Counts:[0,1,0,0,2,0,2,10]       

#     Time taken for epoch 255 is 2498.25 sec
# Epoch 256-2000/2000: Map:0.0619, Cls:0.029, Route:1.35(0.062, 97.019), Uniq:nan, Counts:[0,2,0,1,3,1,0,6]               