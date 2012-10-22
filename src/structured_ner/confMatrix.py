import numpy as N
import itertools

class confMatrix():
  def __init__(self, vals):
    self.x = vals
  
 

  def confusion_matrix(self):
    Ytrue = []
    Ypred = []

    
    for i in self.x:
      if len(i) != 0:
	Ytrue.append(i[1])
	Ypred.append(i[2])
	  



    classes = list(set(Ytrue))
    n = len(classes)

    #error = N.array([zip(Ytrue,Ypred).count(x) for x in itertools.product(classes,repeat=2)]).reshape(n,n)
    #print error

    error = N.array([z.count(x) for z in [zip(Ytrue,Ypred)] for x in itertools.product(classes,repeat=2)]).reshape(n,n)
    print error
    return error



 