class Character():
    def __init__(self, _id, pos_x, pos_y, hp):
        self._id = _id 
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.hp = hp

    def get_position(self):
        return (self.pos_x, self.pos_y)

    def move_up(self):
        pass

    def move_down(self):
        pass

    def move_right(self):
        pass

    def move_left(self):
        pass

class Predator(Character):
    def __init__(self, _id, pos_x, pos_y, hp):
        super().__init__(_id, pos_x, pos_y, hp)

class Prey(Character):
    def __init__(self, _id, pos_x, pos_y, hp):
        super().__init__(_id, pos_x, pos_y, hp)


