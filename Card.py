# Card.py

class BeloteCard32(object):
    def __init__(self, id : int):
        self.id = id

        if 0 <= id <= 7:
            self.color = 'carreau'
        elif 8 <= id <= 15:
            self.color = 'coeur'
        elif 16 <= id <= 23:
            self.color = 'trèfle'
        elif 24 <= id <= 31:
            self.color = 'pique'

        base_indices = [0, 8, 16, 24]
        if id in base_indices :
            self.label = '7'
        elif id in [x + 1 for x in base_indices]:
            self.label = '8'
        elif id in [x + 2 for x in base_indices]:
            self.label = '9'
        elif id in [x + 3 for x in base_indices]:
            self.label = '10'
        elif id in [x + 4 for x in base_indices]:
            self.label = 'V'
        elif id in [x + 5 for x in base_indices]:
            self.label = 'D'
        elif id in [x + 6 for x in base_indices]:
            self.label = 'R'
        elif id in [x + 7 for x in base_indices]:
            self.label = 'A'

        if self.color == 'pique':
            if self.label == 'V':
                self.value = 20
            elif self.label == '9':
                self.value = 14
            elif self.label == 'A':
                self.value = 11
            elif self.label == '10':
                self.value = 10
            elif self.label == 'R':
                self.value = 4
            elif self.label == 'D':
                self.value = 3
            elif self.label in ['7','8']:
                self.value = 0
        else:
            if self.label == 'A':
                self.value = 11
            elif self.label == '10':
                self.value = 10
            elif self.label == 'R':
                self.value = 4
            elif self.label == 'D':
                self.value = 3
            elif self.label == 'V':
                self.value = 2
            elif self.label == '9':
                self.value = 0
            elif self.label in ['7','8']:
                self.value = 0

    def display(self):
        return (self.label,self.color,self.id)
    
class BeloteCard12(object):
    def __init__(self, id : int):
        self.id = id

        if 0 <= id <= 2:
            self.color = 'carreau'
        elif 3 <= id <= 5:
            self.color = 'coeur'
        elif 6 <= id <= 8:
            self.color = 'trèfle'
        elif 9 <= id <= 11:
            self.color = 'pique'

        base_indices = [0, 3, 6, 9]
        if id in base_indices :
            self.label = '9'
        elif id in [x + 1 for x in base_indices]:
            self.label = '10'
        elif id in [x + 2 for x in base_indices]:
            self.label = 'V'

        if self.color == 'pique':
            if self.label == 'V':
                self.value = 20
            elif self.label == '9':
                self.value = 14
            elif self.label == '10':
                self.value = 10
        else:
            if self.label == '10':
                self.value = 10
            elif self.label == 'V':
                self.value = 2
            elif self.label == '9':
                self.value = 0

    def display(self):
        return (self.label,self.color,self.id)