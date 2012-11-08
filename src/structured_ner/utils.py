def entities_from_list(list):
    b = False
    current_start = 0
    current_tag = None

    entities = []

    for i in range(len(list)):
      if list[i] == 'O' or i == len(list)-1 or (b and list[i] != current_tag):
          if b:
              entities.append( (current_start, i-1, current_tag) )
              b = False
              current_start = 0
              current_tag = None

      if list[i] != 'O' and not b:
          b = True
          current_start = i
          current_tag = list[i]

      if i == len(list)-1 and list[i] != 'O':
          entities.append( (i, i, list[i]) )

    return entities


import numpy as N
import itertools

class ConfusionMatrix():

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

        return error




