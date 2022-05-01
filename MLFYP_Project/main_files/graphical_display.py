from graphics import *
import math

def main_draw(game):
    size_x = 500
    size_y = 500
    table_radius = size_x/3
    origin_x = size_x/2
    origin_y = size_y/2

    win = GraphWin("Table", size_x, size_y)
    c = Circle(Point(origin_x, origin_y), table_radius)
    c.draw(win)
    win.setBackground("white")
    win.plot(35, 128, "blue")

    #The parametric equation for a circle is:
    # x = cx + r * cos(a)
    # y = cy + r * sin(a)
    

    player_1= {"x": origin_x + table_radius * math.cos(1/2*math.pi), "y": origin_y + table_radius * math.sin(1/2*math.pi), "size": 32}
    player_2= {"x": origin_x + table_radius * math.cos(5/4*math.pi), "y": origin_y + table_radius * math.sin(5/4*math.pi), "size": 32}
    player_3= {"x": origin_x + table_radius * math.cos(7/4*math.pi), "y": origin_y + table_radius * math.sin(7/4*math.pi), "size": 32}

    player_1_graphic = Circle(Point(player_1['x'], player_1['y']), player_1['size'])
    player_1_graphic.draw(win)
    player_2_graphic = Circle(Point(player_2['x'], player_2['y']), player_2['size'])
    player_2_graphic.draw(win)
    player_3_graphic = Circle(Point(player_3['x'], player_3['y']), player_3['size'])
    player_3_graphic.draw(win)

    Player1 = Player(game.Player1.position, player_1_graphic, game.Player1.card_holding, game.Player1.action)
    Player2 = Player(game.Player2.position, player_2_graphic, game.Player2.card_holding, game.Player2.action)
    Player3 = Player(game.Player3.position, player_3_graphic, game.Player3.card_holding, game.Player3.action)

    player_list = [Player1, Player2, Player3]
    PokerTable = Table([], )

    for player in player_list:
        player.active_status = True
    
    Player1.set_is_current_move(True)

    for player in player_list:
        print(player.active_status)
        if player.is_active():
            player.graphic_object.setOutline('green')
        if player.is_current_move:
            player.graphic_object.setFill('red')

    Player3.graphic_object.active_status = False

    

    win.getMouse() # pause for click in window
    win.close()


class Table():

    def __init__(self, active_players, current_player, most_recent_action):
        self.active_players = active_players
        self.current_player = current_player
        self.most_recent_action = most_recent_action   
        self.player_list = player_list 

    def current_player(self, player):
        return self.current_player

    def set_active_player(player):
        present = False
        for p in self.active_players:
            if p == player:
                present = True
        if present = False:
            self.active_players.append(player)

        


class Player(Table):

    def __init__(self, position, graphic_object, card_holding, action):
        self.active_status = False
        self.is_current_move = False
        self.position = position
        self.graphic_object = graphic_object
        self.card_holding = card_holding    
        self.action = action

        
    def is_active(self):
        return self.active_status

    def set_active(self, status):
        self.active_status = status
        super().set_active_player(self)

    def set_is_current_move(self, is_move):
        self.is_current_move = is_move 
    
if __name__ == '__main__':
    
    main_draw()