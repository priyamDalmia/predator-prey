class Character():
    def __init__(self, _id, pos_x, pos_y, hp):
        self._id = _id 
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.hp = hp

    def get_position(self):
        return (self.pos_x, self.pos_y)
    
    def set_position(self, x, y):
        self.pos_x = x
        self.pos_y = y

    def move_up(self):
        self.pos_x += 1      

    def move_down(self):
        self.pos_x -= 1

    def move_right(self):
        self.pos_y += 1

    def move_left(self):
        self.pos_y -= 1

class Predator(Character):
    def __init__(self, _id, pos_x, pos_y, hp):
        super().__init__(_id, pos_x, pos_y, hp)

class Prey(Character):
    def __init__(self, _id, pos_x, pos_y, hp):
        super().__init__(_id, pos_x, pos_y, hp)


