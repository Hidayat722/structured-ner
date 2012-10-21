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
