import numpy as np
from datetime import datetime

def mk_data(data,samples,strt_indx,stp_indx,mode):
  
    feats = 27
    
    if mode=='train':
      target = np.zeros(samples)
    else:
      target = 0
    
    features = np.zeros([samples,feats])
    
    i=0
    
    for cust in np.arange(strt_indx,stp_indx):
    
      x=0
      ts = data[2][cust][1]
      
      if mode=='train':
        target[i] = data[6][cust][1]
      else:
        target = 0
      
      cumsum_ts = np.cumsum(ts)
      max_count=np.count_nonzero(np.where(ts==np.max(ts)))
      min_count=np.count_nonzero(np.where(ts==np.min(ts)))
      
      unique_elements, counts_elements = np.unique(data[1][cust][1], return_counts=True)
      count = np.count_nonzero(counts_elements)
    
      maxv= np.where(ts==np.max(ts))
      minv= np.where(ts==np.min(ts))
      l_eq=np.where(ts>data[4][cust][1])
    
      l= data[1][cust][1][-1]
      f= data[1][cust][1][0]
      ld = data[5][cust][1]
    
      maxd = data[1][cust][1][maxv[0][-1]]
      mind = data[1][cust][1][minv[0][-1]]
    
      date_diff= datetime.strptime(l,'%Y-%m-%d') - datetime.strptime(f,'%Y-%m-%d')
      date_diff2= datetime.strptime(ld,'%Y-%m-%d') - datetime.strptime(f,'%Y-%m-%d')
      date_diff3= datetime.strptime(ld,'%Y-%m-%d') - datetime.strptime(l,'%Y-%m-%d')
      date_diff4= datetime.strptime(ld,'%Y-%m-%d') - datetime.strptime(maxd,'%Y-%m-%d')
      date_diff5= datetime.strptime(ld,'%Y-%m-%d') - datetime.strptime(mind,'%Y-%m-%d')
    
      l_eq=np.where(ts>data[4][cust][1])
    
      if len(l_eq[0])>0:
        lda = data[1][cust][1][l_eq[0][-1]]
      else:
        lda = data[1][cust][1][0]
    
      date_diff6 =datetime.strptime(ld,'%Y-%m-%d') - datetime.strptime(lda,'%Y-%m-%d')
      loan_date=datetime.strptime(data[5][cust][1],'%Y-%m-%d')
    
      features[i,x] = np.max(ts)
      x+=1
      features[i,x] = np.min(ts)
      x+=1
      features[i,x] = count
      x+=1
      features[i,x] = max_count
      x+=1
      features[i,x] = min_count
      x+=1
      features[i,x] = np.mean(ts[np.where(ts>0)])    #1
      x+=1
      features[i,x] = np.mean(ts[np.where(ts<0)])    #2
      x+=1
      features[i,x] = np.median(ts[np.where(ts>0)])    #1
      x+=1
      features[i,x] = np.median(ts[np.where(ts<0)])    #2
      x+=1
      features[i,x] = np.std(ts[np.where(ts>0)])    
      x+=1
      features[i,x] = np.std(ts[np.where(ts<0)])
      x+=1
      features[i,x] = cumsum_ts[-1] 
      x+=1
      features[i,x] = cumsum_ts[0]   
      x+=1
      features[i,x] = np.count_nonzero(ts)
      x+=1
      features[i,x] = date_diff.days
      x+=1
      features[i,x] = date_diff2.days
      x+=1
      features[i,x] = date_diff3.days
      x+=1
      features[i,x] = date_diff4.days
      x+=1
      features[i,x] = date_diff5.days
      x+=1
      features[i,x] = date_diff6.days
      x+=1
      features[i,x] = np.count_nonzero(ts[np.where(ts>0)])
      x+=1
      features[i,x] = np.count_nonzero(ts[np.where(ts<0)])
      x+=1
      features[i,x] = np.count_nonzero(ts[np.where(ts>(data[4][cust][1]))])
      x+=1
      features[i,x] = loan_date.day #3 
      x+=1
      features[i,x] = loan_date.month #3 
      x+=1
      features[i,x] = loan_date.year #3 
      x+=1
      features[i,x] = data[4][cust][1] #3 
      
      i+=1
    
    features[np.isnan(features)] = 0
    features=np.square(features)
    
    if mode=='train':
      return features,target
    else:
      return features
