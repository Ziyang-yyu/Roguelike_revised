
from time import sleep
from collections import namedtuple
import random
import agents
import numpy as np

global a_turn
a_turn = False

def startScene():
    print('''

 You heard a sound but the room is too dark, you cannot see anything, say hi to get a response

  ''')

def showInstructions():


  print('''

story description

========
Commands:
  go [left, right, up, down]
  set trap
  get [items]
  search treasure

''')

#treasure = ['ring', 'bracelet', 'watch', 'crown', 'necklace', 'rare coin', 'amulet', 'diamonds']
Treasure = namedtuple('Treasure', ['name', 'worth'])

treasure = [
    Treasure('ring', 50),
    Treasure('pin clutch', 100),
    Treasure('bracelet', 150),
    Treasure('watch', 200),
    Treasure('pocket watch', 300)
]

#treasure = [ 'platinum bracelet', 'jeweled crown', 'silver watch', 'gold ring', 'diamond pendent','amulet']
items = ['ladder', 'key', 'lock', 'letter']

#an inventory, which is initially empty
inventory = 0
a_inventory = 0

#traps = ['door locked from outside']
def trap_check(current_room, inv):
    if not a_turn:
        t = 'a_trap'
    else:
        t = 'trap'
    print('trap check')

    print(rooms[current_room]['hidden'])
    if rooms[current_room]['hidden'] == t:
        print('ooops! a trap here, some of the inventory is stolen')
        inv -= inv*random.randrange(10,50,5)/100
        rooms[current_room]['hidden'] = ''
    return inv


def showStatus():
  global a_turn
  global inventory
  global a_inventory
  #print the player's current status

  #print the current inventory
  if not a_turn:
     print('You are in ' + currentRoom)
     inventory = trap_check(currentRoom, inventory)
     print('Inventory : ' + str(inventory))
      #description is used to provide detail in the room.
     if "description" in rooms[currentRoom]:
         print(rooms[currentRoom]['description'])
        #print('---------------------------')


      #This lets the player know there are stairs and that they have the option to go up or down.
     if "stairs" in rooms[currentRoom]:
         print('You see ' + rooms[currentRoom]['stairs'])
        #print('---------------------------')

      #This lets the player know there is an item that can be picked up.
     if "item" in rooms[currentRoom]:
        print('You see a ' + rooms[currentRoom]['item'])

  else:

      print('The life rescuer is in ' + a_currentRoom)
      #print('OOPS! There is a trap')

      a_inventory = trap_check(a_currentRoom, a_inventory)
      print('Life rescuer\'s inventory : ' + str(a_inventory))

  print("---------------------------")

  #These are all things that will be printed when the player enters a room if they are true



#a dictionary linking a room to other rooms
rooms = {

            'The Attic' : {

                  'down' : 'The Main Bedroom',
                  'right': 'The Top Floor Door',
                  'hidden': 'treasure',
                  'state': 'open',
                  'description' : ''
            },

            'The Top Floor Door' : {

                  'right' : 'The Sunroom',
                  'left': 'The Attic',
                  'down':  'The First Floor Door',
                  'state': 'locked', # can be changed
                  'hidden':'',
                  'description' : ''

            },

            'The Sunroom' : {
                  'right': 'The Balcony',
                  'left': 'The Top Floor Door',
                  'down': 'The Laundry Room',
                  'hidden':'',
                  'description' : 'You see an unlocked drawer next to the bed',
                  'state': 'open',
                  'item' : 'drawer'
            },

            'The Balcony' : {
                  'left' : 'The Sunroom',
                  'down': 'Guest Bedroom',
                  'description':'',
                  'state': 'exit',
                  'hidden':''
            },

            'The Main Bedroom' : {
                'up': 'The Attic',
                'down': 'The Living Room',
                'right':'The First Floor Door',
                'description':'',
                'state': 'open',
                'hidden': 'treasure'
            },

            'The First Floor Door' : {
                'up': 'The Top Floor Door',
                'down': 'The Ground Floor Hallway',
                'left':'The Main Bedroom',
                'right': 'The Laundry Room',
                'state': 'open',
                'hidden':''

            },

            'The Laundry Room' : {
                'up': 'The Sunroom',
                'down': 'The Dining Room',
                'right':'The Guest Bedroom',
                'left': 'The First Floor Door',
                'description':'',
                'state': 'open',
                'hidden':''
            },

            'The Guest Bedroom' : {
                'up': 'The Balcony',
                'down': 'The Kitchen',
                'left': 'The Laundry Room',
                'description':'',
                'state': 'open',
                'hidden': ''

            },

            'The Living Room' : {
                'up': 'The Main Bedroom',
                'down': 'The Basement Storage Room',
                'right':'The Ground Floor Hallway',
                'description':'',
                'state': 'open',
                'hidden':''

            },

            'The Ground Floor Hallway' : {
                'up': 'The First Floor Door',
                'down': 'The Game Room',
                'right':'The Dining Room',
                'left': 'The Living Room',
                'description':'',
                'state': 'open',
                'hidden': ''
            },

            'The Dining Room' : {
                'up': 'The Laundry Room',
                'down': 'The Basement Door',
                'right':'The Kitchen',
                'left': 'The Ground Floor Hallway',
                'description':'',
                'state': 'open',
                'hidden':''

            },

            'The Kitchen' : {
                'up': 'The Guest Bedroom',
                'down': 'The Garage',
                'left': 'The Dining Room',
                'description':'',
                'state': 'open',
                'hidden': 'treasure'

            },

            'The Basement Storage Room' : {
                'up': 'The Main Bedroom',
                'right':'The Game Room',
                'description':'',
                'state': 'open',
                'hidden': 'treasure'

            },

            'The Game Room' : {
                'up': 'The Ground Floor Hallway',
                'right': 'The Basement Door',
                'left': 'The Basement Storage Room',
                'description':'',
                'state': 'open',
                'hidden': 'treasure'

            },

            'The Basement Door' : {
                'up': 'The Dining Room',
                'right': 'The Garage',
                'left': 'The Game Room',
                'state': 'locked',
                'description':'',
                'hidden':''

            },
            'The Garage' : {
                'up': 'The Kitchen',
                'left': 'The Basement Door',
                'description':'',
                'state': 'open',
                'hidden': ''

            }
         }

#start the player in the Hall
currentRoom = 'The Living Room'
possible_start = [s for s in list(rooms.keys()) if rooms[s]['state'] == 'open']

a_currentRoom = random.choice(possible_start)
showInstructions()
showStatus()

def switch_turn():
    global a_turn
    if a_turn == False:
        a_turn= True
    else:
        a_turn= False

#st = np.array([[(0,255),(0,10),(100,100),(0,0)],[(100,100),(100,100),(0,255),(100,100)],[(0,10),(0,255),(100,100),(0,10)],[(101,0),(0,0),(0,255),(0,255)]])

def state_converter(rooms_dict):
    global a_currentRoom
    global currentRoom
    print(a_currentRoom)
    states = np.array([])

    for key, value in rooms_dict.items():
        sub=np.array([0,0,0])

        if key == currentRoom:
            sub[2]='5' #human
        if key == a_currentRoom:

            sub[0]='2' #agent

        if value['hidden'] == 'treasure':
            sub[1]=255
        elif value['hidden'] == 'trap': # set by human player
            sub[1]=50
        elif value['state'] == 'locked':
            sub=np.array([1,1,1])
        elif value['hidden'] == 'a_trap': # set by agent
            sub[1]=20

        states=np.append(states,sub)
    states = states.reshape(4,4,3)

    return states

st = state_converter(rooms)

while True:
  print('start')

  move = ''

  #st = state_converter(rooms)
  print(st)
  while move == '':


      if not a_turn:

          move = input(">")

      else:

         agent_coord=np.where(st==2)
         human_coord=np.where(st==5)
         treasure_coord=np.where(st==255)


         #move = agents.random_action()
         move = agents.bfs_action(agent_coord, human_coord,  treasure_coord, st, a_inventory)

          # TODO: run simulations to find the best move
          # TODO: convert game to states

          #move = random.choice(ACTIONS)
         print("The life rescuer> "+move)

  move = move.lower().split()

# List of actions
  #if they type 'go' first
  if move[0] == 'go':
    if not a_turn:
        #check that they are allowed to move wherever they want to go
        if move[1] in rooms[currentRoom]:
          #set the current room to the new room
          next = rooms[currentRoom][move[1]]
          if rooms[next]['state'] !='open':
              print('Door locked, you cannot go there')
          else:
              currentRoom = next

        #there is no door (link) to the new room
        else:

            print('There is no door here')
            #sleep(1)
    else:
        #check that they are allowed to move wherever they want to go
        if move[1] in rooms[a_currentRoom]:
          #set the current room to the new room
          next = rooms[a_currentRoom][move[1]]
          if rooms[next]['state'] !='open':
              print('Door locked, you cannot go there')
          else:
              a_currentRoom = next
        #there is no door (link) to the new room
        else:

            print('There is no door here')
            #sleep(1)


  #if they type 'get' first
  elif move[0] == 'get' :
    #if the room contains an item, and the item is the one they want to get
    if "item" in rooms[currentRoom] and move[1] in rooms[currentRoom]['item']:

      #add the item to their inventory if it has two names (ex shiny sword)
      inventory += [' '.join(move[1:])]
      #The code commented out below is only needed if your items all have one name only
      #inventory += [move[1]]

      #display a helpful message
      print(move[1] + ' was picked up!')

      #delete the item from the room

      rooms[currentRoom]['item'] = ''
    #otherwise, if the item isn't there to get
    else:
      #tell them they can't get it

      print('Can\'t pick up ' + move[1] + '!')

     #If they type search first it will pull a random treasure from the room if the room has hidden in it.
     #If the room has trap, it will pull a random trap and kill the player.

  elif move[0] == 'search':
      t = random.choice(treasure)
      if not a_turn:

    #If the room has a hidden treasure, a player can search for it.
        if move[1] in rooms[currentRoom]['hidden']:

            inventory += t.worth
            print("You found a ", t.name, "! Nice job!")
            rooms[currentRoom]['hidden'] = ''
        else:
            print('Nothing found')
      else:
        if move[1] in rooms[a_currentRoom]['hidden']:

            a_inventory += t.worth
            print("The life rescuer found a ", t.name, "! Nice job!")
            rooms[a_currentRoom]['hidden'] = ''
        else:
            print('Nothing found')

      #sleep(2)
  elif move[0] == 'set' :
      if not a_turn:
          if rooms[currentRoom]['hidden'] != 'trap':

              reply =  input("A trap costs 100 to set, once the trap is set, you can\
              randomly take an item from the inventory list of the person who enters this\
              room later. However, if that person has no inventory at all, nothing will be\
              taken. Think twice, and enter 'y' if you wish to set the trap, else press any key to continue>")
              if reply.lower() == 'y':
                  if inventory < 100:
                      print('You do not have enough money to set a trap')
                      continue
                  inventory -= 100
                  rooms[currentRoom]['hidden'] = 'trap'
              else:
                  continue
          else:
              print('There is already a trap, you cannot set another trap!')
      else:
          if rooms[a_currentRoom]['hidden'] != 'a_trap':
              if a_inventory >= 100:
                  rooms[currentRoom]['hidden'] = 'a_trap'
                  print('A trap is set, stay alert!')
                  a_inventory -= 100
          else:
              continue

  st = state_converter(rooms)
  showStatus()

  switch_turn()


      # TODO: complete trap setting, dont show the location
"""
  else:


  # user turn
       while True:
          msg = input("")
          ints = class_predict(msg)
          ans = get_response(ints, intents)
          print(ans)


      if not a_turn:

          ints = Journalist.class_predict(' '.join(move))
          ans = agent.get_response(ints, agent.intents)
          print("The life rescuer: "+ans)

      else:
          c =  agent_nn.agents[0].forward(states).argmax()

          if c == 0:
              move = "Go right"
          elif c == 1:
              move = "Go left"
          elif c == 2:
               move = "Go up"
          elif c == 3:
               move = "Go down"
          states,b,done, search = e.step(c,states, search)
          #move = random.choice(ACTIONS)
          print("Agent > "+move)

      print('Nothing is found')

"""
# TODO: no cheating
# online (not knowing the game) or offline (konwing the game)
# genetic algorithm vs reinforcement learning vs dfs
# to coherent
# logic + proof
# pre-train on different mazes
# more actions
